from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from .const import S3GEN_SR, S3GEN_SIL
from .s3gen import S3Token2Wav


@dataclass
class _DecodeWindow:
    speech_tokens: torch.Tensor
    noise: torch.Tensor
    context_samples: int
    stable_end_token: int


@dataclass
class S3GenBatchingMetrics:
    """Cumulative evidence for bucketed S3Gen decode operations."""

    bucket_width_tokens: int = 8
    decode_calls: int = 0
    decoded_rows: int = 0
    max_batch_size: int = 0
    mixed_length_batch_calls: int = 0
    token_slots: int = 0
    valid_token_slots: int = 0
    mel_frame_slots: int = 0
    valid_mel_frames: int = 0
    incremental_calls: int = 0
    full_prefix_calls: int = 0
    bucket_calls: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bucket_width_tokens < 1:
            raise ValueError("S3Gen bucket width must be positive")

    def record(
        self,
        *,
        kind: str,
        bucket_tokens: int,
        token_lengths: list[int],
        bucket_mel_frames: int,
        mel_lengths: list[int],
    ) -> None:
        batch_size = len(token_lengths)
        self.decode_calls += 1
        self.decoded_rows += batch_size
        self.max_batch_size = max(self.max_batch_size, batch_size)
        self.mixed_length_batch_calls += int(len(set(token_lengths)) > 1)
        self.token_slots += batch_size * bucket_tokens
        self.valid_token_slots += sum(token_lengths)
        self.mel_frame_slots += batch_size * bucket_mel_frames
        self.valid_mel_frames += sum(mel_lengths)
        if kind == "incremental":
            self.incremental_calls += 1
        elif kind == "full_prefix":
            self.full_prefix_calls += 1
        else:
            raise ValueError(f"unknown S3Gen decode kind: {kind}")
        self.bucket_calls[bucket_tokens] = self.bucket_calls.get(bucket_tokens, 0) + 1

    def as_dict(self) -> dict[str, object]:
        token_padding = self.token_slots - self.valid_token_slots
        mel_padding = self.mel_frame_slots - self.valid_mel_frames
        return {
            "bucketWidthTokens": self.bucket_width_tokens,
            "decodeCalls": self.decode_calls,
            "decodedRows": self.decoded_rows,
            "meanBatchSize": round(self.decoded_rows / self.decode_calls, 4)
            if self.decode_calls
            else 0.0,
            "maxBatchSize": self.max_batch_size,
            "mixedLengthBatchCalls": self.mixed_length_batch_calls,
            "incrementalCalls": self.incremental_calls,
            "fullPrefixCalls": self.full_prefix_calls,
            "tokenSlots": self.token_slots,
            "validTokenSlots": self.valid_token_slots,
            "paddedTokenSlots": token_padding,
            "tokenPaddingPercent": round(
                100.0 * token_padding / self.token_slots,
                4,
            )
            if self.token_slots
            else 0.0,
            "melFrameSlots": self.mel_frame_slots,
            "validMelFrames": self.valid_mel_frames,
            "paddedMelFrames": mel_padding,
            "melPaddingPercent": round(
                100.0 * mel_padding / self.mel_frame_slots,
                4,
            )
            if self.mel_frame_slots
            else 0.0,
            "bucketCalls": {
                str(bucket): calls
                for bucket, calls in sorted(self.bucket_calls.items())
            },
        }


class S3GenStreamer:
    """Incrementally decode S3 speech tokens into waveform chunks.

    S3Gen uses a small lookahead window when converting speech tokens to mels.
    The streamer retains a bounded left-context window, reuses stable diffusion
    noise for overlapping windows, and crossfades adjacent waveform chunks.
    Setting ``left_context_tokens`` to zero restores full-prefix decoding for
    quality comparisons and regression benchmarks.
    """

    def __init__(
        self,
        s3gen: S3Token2Wav,
        ref_dict: dict,
        *,
        n_cfm_timesteps: Optional[int] = None,
        crossfade_ms: float = 12.0,
        left_context_tokens: int = 25,
    ):
        self.s3gen = s3gen
        self.ref_dict = ref_dict
        self.n_cfm_timesteps = n_cfm_timesteps or (2 if s3gen.meanflow else 10)
        self.crossfade_samples = max(0, int(S3GEN_SR * crossfade_ms / 1000.0))
        self.left_context_tokens = max(0, int(left_context_tokens))

        self.token_buffer: list[torch.Tensor] = []
        self.noised_mels: torch.Tensor | None = None
        self.hift_cache_source = torch.zeros(
            1, 1, 0, device=s3gen.device, dtype=s3gen.dtype
        )
        self.pending_tail: torch.Tensor | None = None
        self.emitted_samples = 0
        self.decoded_tokens = 0
        self.generated_tokens = 0
        self.decoded_chunks = 0
        self.finished = False

    def append(self, speech_token: torch.Tensor) -> None:
        if self.finished:
            raise RuntimeError("cannot append tokens after finish()")
        speech_token = torch.atleast_2d(speech_token).to(
            device=self.s3gen.device, dtype=torch.long
        )
        self.token_buffer.append(speech_token)
        self.generated_tokens += speech_token.shape[-1]

    def flush(self, *, finalize: bool = False) -> torch.Tensor | None:
        chunk = self._decode_available(finalize=finalize)
        return self._emit_smoothed(chunk, finalize=finalize)

    def finish(self) -> torch.Tensor | None:
        if not self.finished:
            silence = torch.tensor(
                [[S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]],
                dtype=torch.long,
                device=self.s3gen.device,
            )
            self.token_buffer.append(silence)
            self.finished = True
        return self.flush(finalize=True)

    def _ensure_noise(self, mel_frames: int) -> torch.Tensor:
        if mel_frames <= 0:
            raise ValueError("mel_frames must be positive")

        shape = (1, 80, mel_frames)
        if self.noised_mels is None:
            self.noised_mels = torch.randn(
                *shape, dtype=self.s3gen.dtype, device=self.s3gen.device
            )
        elif self.noised_mels.shape[-1] < mel_frames:
            extra = torch.randn(
                1,
                80,
                mel_frames - self.noised_mels.shape[-1],
                dtype=self.s3gen.dtype,
                device=self.s3gen.device,
            )
            self.noised_mels = torch.cat([self.noised_mels, extra], dim=-1)
        return self.noised_mels[:, :, :mel_frames]

    def _decode_available(self, *, finalize: bool) -> torch.Tensor | None:
        if self.left_context_tokens > 0:
            return self._decode_incremental(finalize=finalize)
        return self._decode_full_prefix(finalize=finalize)

    def _prepare_incremental_window(
        self,
        *,
        finalize: bool,
    ) -> _DecodeWindow | None:
        if not self.token_buffer:
            return None

        speech_tokens = torch.cat(self.token_buffer, dim=1)
        stable_end_token = speech_tokens.shape[-1]
        if not finalize:
            lookahead = self.s3gen.flow.pre_lookahead_len
            if stable_end_token <= lookahead:
                return None
            stable_end_token -= lookahead

        if stable_end_token <= self.decoded_tokens:
            return None

        context_start_token = max(
            0,
            self.decoded_tokens - self.left_context_tokens,
        )
        token_mel_ratio = self.s3gen.flow.token_mel_ratio
        noise = self._ensure_noise(stable_end_token * token_mel_ratio)
        noise = noise[
            :,
            :,
            context_start_token * token_mel_ratio : stable_end_token
            * token_mel_ratio,
        ]
        samples_per_token = S3GEN_SR // self.s3gen.flow.input_frame_rate
        context_samples = (
            self.decoded_tokens - context_start_token
        ) * samples_per_token
        return _DecodeWindow(
            speech_tokens=speech_tokens[:, context_start_token:],
            noise=noise,
            context_samples=context_samples,
            stable_end_token=stable_end_token,
        )

    def _decode_incremental(self, *, finalize: bool) -> torch.Tensor | None:
        window = self._prepare_incremental_window(finalize=finalize)
        if window is None:
            return None

        output_mels = self.s3gen(
            speech_tokens=window.speech_tokens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=self.ref_dict,
            n_cfm_timesteps=self.n_cfm_timesteps,
            finalize=finalize,
            skip_vocoder=True,
            noised_mels=window.noise,
        ).to(dtype=self.s3gen.dtype)

        empty_source = torch.zeros(
            1,
            1,
            0,
            dtype=self.s3gen.dtype,
            device=self.s3gen.device,
        )
        wav, _ = self.s3gen.hift_inference(output_mels, empty_source)
        wav[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade

        overlap = self.crossfade_samples if window.context_samples > 0 else 0
        trim_samples = max(0, window.context_samples - overlap)
        if trim_samples >= wav.shape[-1]:
            return None

        self.decoded_tokens = window.stable_end_token
        self.decoded_chunks += 1
        return wav[:, trim_samples:]

    def _decode_full_prefix(self, *, finalize: bool) -> torch.Tensor | None:
        if not self.token_buffer:
            return None

        speech_tokens = torch.cat(self.token_buffer, dim=1)
        effective_tokens = speech_tokens.shape[-1]
        if not finalize:
            lookahead = self.s3gen.flow.pre_lookahead_len
            if effective_tokens <= lookahead:
                return None
            effective_tokens -= lookahead

        if effective_tokens <= 0:
            return None

        noised_mels = self._ensure_noise(
            effective_tokens * self.s3gen.flow.token_mel_ratio
        )
        output_mels = self.s3gen(
            speech_tokens=speech_tokens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=self.ref_dict,
            n_cfm_timesteps=self.n_cfm_timesteps,
            finalize=finalize,
            skip_vocoder=True,
            noised_mels=noised_mels,
        ).to(dtype=self.s3gen.dtype)

        wav, source = self.s3gen.hift_inference(output_mels, self.hift_cache_source)
        self.hift_cache_source = source.detach()
        wav[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade

        if wav.shape[-1] <= self.emitted_samples:
            return None

        self.decoded_chunks += 1
        return wav[:, self.emitted_samples :]

    def _emit_smoothed(
        self, chunk: torch.Tensor | None, *, finalize: bool
    ) -> torch.Tensor | None:
        if chunk is None or chunk.shape[-1] == 0:
            return None

        if self.crossfade_samples <= 0:
            self.emitted_samples += chunk.shape[-1]
            return chunk

        chunk_len = chunk.shape[-1]
        if not finalize and chunk_len <= self.crossfade_samples:
            return None

        if finalize:
            output = chunk
            if self.pending_tail is not None:
                output = self._join_with_crossfade(self.pending_tail, chunk)
            self.pending_tail = None
            self.emitted_samples += chunk_len
            return output

        emit_len = chunk_len - self.crossfade_samples
        body = chunk[:, :emit_len]
        new_tail = chunk[:, emit_len:].detach().clone()
        output = body

        if self.pending_tail is not None:
            output = self._join_with_crossfade(self.pending_tail, body)

        self.pending_tail = new_tail
        self.emitted_samples += emit_len
        return output

    def _join_with_crossfade(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        overlap = min(self.crossfade_samples, left.shape[-1], right.shape[-1])
        if overlap <= 0:
            return torch.cat([left, right], dim=1)

        fade_out = torch.linspace(
            1.0, 0.0, overlap, device=right.device, dtype=right.dtype
        ).unsqueeze(0)
        fade_in = 1.0 - fade_out
        crossed = left[:, -overlap:] * fade_out + right[:, :overlap] * fade_in

        parts = []
        if left.shape[-1] > overlap:
            parts.append(left[:, :-overlap])
        parts.append(crossed)
        if right.shape[-1] > overlap:
            parts.append(right[:, overlap:])
        return torch.cat(parts, dim=1)


class S3GenBatchStreamer:
    """Batch compatible :class:`S3GenStreamer` decode operations.

    Individual stream state remains isolated. Streams with compatible prefix
    and vocoder-cache lengths are grouped into one S3Gen and HiFT pass, then
    split before each stream applies its own crossfade bookkeeping.
    """

    def __init__(
        self,
        streamers: Iterable[S3GenStreamer],
        *,
        bucket_width_tokens: int = 8,
        metrics: S3GenBatchingMetrics | None = None,
    ):
        self.streamers = list(streamers)
        if not self.streamers:
            raise ValueError("at least one S3GenStreamer is required")
        self.s3gen = self.streamers[0].s3gen
        if any(streamer.s3gen is not self.s3gen for streamer in self.streamers):
            raise ValueError("all streamers must share one S3Gen model")
        self.bucket_width_tokens = int(bucket_width_tokens)
        if self.bucket_width_tokens < 1:
            raise ValueError("S3Gen bucket width must be positive")
        self.metrics = metrics or S3GenBatchingMetrics(self.bucket_width_tokens)
        if self.metrics.bucket_width_tokens != self.bucket_width_tokens:
            raise ValueError("S3Gen metrics and streamer bucket widths must match")

    def flush(
        self,
        indices: Iterable[int],
        *,
        finalize: bool = False,
    ) -> list[tuple[int, torch.Tensor]]:
        full_prefix_groups: dict[tuple[int, int, int], list[int]] = defaultdict(
            list,
        )
        incremental_groups: dict[
            tuple[int, int, int],
            list[tuple[int, _DecodeWindow]],
        ] = defaultdict(list)
        for index in indices:
            streamer = self.streamers[index]
            if not streamer.token_buffer:
                continue
            if streamer.left_context_tokens > 0:
                window = streamer._prepare_incremental_window(finalize=finalize)
                if window is None:
                    continue
                incremental_groups[
                    (
                        id(streamer.ref_dict),
                        self._bucket_length(window.speech_tokens.shape[-1]),
                        streamer.n_cfm_timesteps,
                    )
                ].append((index, window))
                continue

            token_count = sum(token.shape[-1] for token in streamer.token_buffer)
            effective_tokens = token_count
            if not finalize:
                effective_tokens -= self.s3gen.flow.pre_lookahead_len
            if effective_tokens <= 0:
                continue
            full_prefix_groups[
                (
                    id(streamer.ref_dict),
                    self._bucket_length(token_count),
                    streamer.n_cfm_timesteps,
                )
            ].append(index)

        outputs: list[tuple[int, torch.Tensor]] = []
        for group in full_prefix_groups.values():
            outputs.extend(self._decode_full_prefix_group(group, finalize=finalize))
        for group in incremental_groups.values():
            outputs.extend(self._decode_incremental_group(group, finalize=finalize))
        return outputs

    def finish(self, indices: Iterable[int]) -> list[tuple[int, torch.Tensor]]:
        pending = []
        for index in indices:
            streamer = self.streamers[index]
            if streamer.finished:
                continue
            silence = torch.tensor(
                [[S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]],
                dtype=torch.long,
                device=self.s3gen.device,
            )
            streamer.token_buffer.append(silence)
            streamer.finished = True
            pending.append(index)
        return self.flush(pending, finalize=True)

    def _bucket_length(self, token_count: int) -> int:
        return (
            (int(token_count) + self.bucket_width_tokens - 1)
            // self.bucket_width_tokens
            * self.bucket_width_tokens
        )

    @staticmethod
    def _pad_last_dim(
        tensors: list[torch.Tensor],
        target_length: int,
        *,
        value: float | int,
    ) -> torch.Tensor:
        return torch.cat(
            [
                F.pad(
                    tensor,
                    (0, target_length - tensor.shape[-1]),
                    value=value,
                )
                for tensor in tensors
            ],
            dim=0,
        )

    def _mel_lengths(
        self,
        token_lengths: list[int],
        *,
        finalize: bool,
    ) -> list[int]:
        lookahead = 0 if finalize else self.s3gen.flow.pre_lookahead_len
        ratio = self.s3gen.flow.token_mel_ratio
        return [(length - lookahead) * ratio for length in token_lengths]

    def _mask_padded_mels(
        self,
        output_mels: torch.Tensor,
        mel_lengths: list[int],
    ) -> torch.Tensor:
        lengths = torch.tensor(
            mel_lengths,
            dtype=torch.long,
            device=output_mels.device,
        )
        positions = torch.arange(
            output_mels.shape[-1],
            device=output_mels.device,
        )
        mask = positions[None, :] < lengths[:, None]
        return output_mels * mask[:, None, :].to(dtype=output_mels.dtype)

    def _samples_per_mel_frame(self) -> int:
        mel_frame_rate = (
            self.s3gen.flow.input_frame_rate * self.s3gen.flow.token_mel_ratio
        )
        if S3GEN_SR % mel_frame_rate:
            raise RuntimeError("S3Gen mel frame rate does not divide sample rate")
        return S3GEN_SR // mel_frame_rate

    def _decode_full_prefix_group(
        self,
        indices: list[int],
        *,
        finalize: bool,
    ) -> list[tuple[int, torch.Tensor]]:
        streamers = [self.streamers[index] for index in indices]
        token_rows = [
            torch.cat(streamer.token_buffer, dim=1) for streamer in streamers
        ]
        token_lengths = [row.shape[-1] for row in token_rows]
        bucket_tokens = self._bucket_length(max(token_lengths))
        speech_tokens = self._pad_last_dim(
            token_rows,
            bucket_tokens,
            value=S3GEN_SIL,
        )
        speech_token_lens = torch.tensor(
            token_lengths,
            dtype=torch.long,
            device=self.s3gen.device,
        )
        mel_lengths = self._mel_lengths(token_lengths, finalize=finalize)
        bucket_mel_frames = self._mel_lengths(
            [bucket_tokens],
            finalize=finalize,
        )[0]
        noises = self._pad_last_dim(
            [
                streamer._ensure_noise(mel_frames)
                for streamer, mel_frames in zip(streamers, mel_lengths)
            ],
            bucket_mel_frames,
            value=0.0,
        )

        output_mels = self.s3gen(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=streamers[0].ref_dict,
            n_cfm_timesteps=streamers[0].n_cfm_timesteps,
            finalize=finalize,
            skip_vocoder=True,
            noised_mels=noises,
        ).to(dtype=self.s3gen.dtype)
        output_mels = self._mask_padded_mels(output_mels, mel_lengths)
        cache_lengths = [
            streamer.hift_cache_source.shape[-1] for streamer in streamers
        ]
        cache_source = self._pad_last_dim(
            [streamer.hift_cache_source for streamer in streamers],
            max(cache_lengths),
            value=0.0,
        )
        cache_source_lens = torch.tensor(
            cache_lengths,
            dtype=torch.long,
            device=self.s3gen.device,
        )
        wavs, sources = self.s3gen.hift_inference(
            output_mels,
            cache_source,
            cache_source_lens=cache_source_lens,
        )
        wavs[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade
        samples_per_mel = self._samples_per_mel_frame()
        valid_samples = [length * samples_per_mel for length in mel_lengths]
        self.metrics.record(
            kind="full_prefix",
            bucket_tokens=bucket_tokens,
            token_lengths=token_lengths,
            bucket_mel_frames=bucket_mel_frames,
            mel_lengths=mel_lengths,
        )

        outputs: list[tuple[int, torch.Tensor]] = []
        for row, (index, streamer) in enumerate(zip(indices, streamers)):
            sample_count = valid_samples[row]
            if wavs.shape[-1] < sample_count or sources.shape[-1] < sample_count:
                raise RuntimeError("S3Gen vocoder returned fewer samples than requested")
            streamer.hift_cache_source = sources[
                row : row + 1,
                :,
                :sample_count,
            ].detach()
            wav = wavs[row : row + 1, :sample_count]
            chunk = None
            if wav.shape[-1] > streamer.emitted_samples:
                streamer.decoded_chunks += 1
                chunk = wav[:, streamer.emitted_samples :]
            emitted = streamer._emit_smoothed(chunk, finalize=finalize)
            if emitted is not None:
                outputs.append((index, emitted))
        return outputs

    def _decode_incremental_group(
        self,
        items: list[tuple[int, _DecodeWindow]],
        *,
        finalize: bool,
    ) -> list[tuple[int, torch.Tensor]]:
        indices = [index for index, _ in items]
        windows = [window for _, window in items]
        streamers = [self.streamers[index] for index in indices]
        token_lengths = [window.speech_tokens.shape[-1] for window in windows]
        bucket_tokens = self._bucket_length(max(token_lengths))
        speech_tokens = self._pad_last_dim(
            [window.speech_tokens for window in windows],
            bucket_tokens,
            value=S3GEN_SIL,
        )
        speech_token_lens = torch.tensor(
            token_lengths,
            dtype=torch.long,
            device=self.s3gen.device,
        )
        mel_lengths = [window.noise.shape[-1] for window in windows]
        bucket_mel_frames = self._mel_lengths(
            [bucket_tokens],
            finalize=finalize,
        )[0]
        noises = self._pad_last_dim(
            [window.noise for window in windows],
            bucket_mel_frames,
            value=0.0,
        )
        output_mels = self.s3gen(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=streamers[0].ref_dict,
            n_cfm_timesteps=streamers[0].n_cfm_timesteps,
            finalize=finalize,
            skip_vocoder=True,
            noised_mels=noises,
        ).to(dtype=self.s3gen.dtype)
        output_mels = self._mask_padded_mels(output_mels, mel_lengths)
        empty_source = torch.zeros(
            len(streamers),
            1,
            0,
            dtype=self.s3gen.dtype,
            device=self.s3gen.device,
        )
        wavs, _ = self.s3gen.hift_inference(output_mels, empty_source)
        wavs[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade
        samples_per_mel = self._samples_per_mel_frame()
        valid_samples = [length * samples_per_mel for length in mel_lengths]
        self.metrics.record(
            kind="incremental",
            bucket_tokens=bucket_tokens,
            token_lengths=token_lengths,
            bucket_mel_frames=bucket_mel_frames,
            mel_lengths=mel_lengths,
        )

        outputs: list[tuple[int, torch.Tensor]] = []
        for row, (index, streamer, window) in enumerate(
            zip(indices, streamers, windows),
        ):
            overlap = (
                streamer.crossfade_samples if window.context_samples > 0 else 0
            )
            trim_samples = max(0, window.context_samples - overlap)
            sample_count = valid_samples[row]
            if wavs.shape[-1] < sample_count:
                raise RuntimeError("S3Gen vocoder returned fewer samples than requested")
            wav = wavs[row : row + 1, :sample_count]
            if trim_samples >= wav.shape[-1]:
                continue
            streamer.decoded_tokens = window.stable_end_token
            streamer.decoded_chunks += 1
            emitted = streamer._emit_smoothed(
                wav[:, trim_samples:],
                finalize=finalize,
            )
            if emitted is not None:
                outputs.append((index, emitted))
        return outputs
