from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import torch

from .const import S3GEN_SR, S3GEN_SIL
from .s3gen import S3Token2Wav


class S3GenStreamer:
    """Incrementally decode S3 speech tokens into waveform chunks.

    S3Gen uses a small lookahead window when converting speech tokens to mels.
    The streamer buffers that lookahead, reuses stable diffusion noise across
    repeated prefix decodes, and keeps HiFT source cache/crossfade state to
    reduce discontinuities at chunk boundaries.
    """

    def __init__(
        self,
        s3gen: S3Token2Wav,
        ref_dict: dict,
        *,
        n_cfm_timesteps: Optional[int] = None,
        crossfade_ms: float = 12.0,
    ):
        self.s3gen = s3gen
        self.ref_dict = ref_dict
        self.n_cfm_timesteps = n_cfm_timesteps or (2 if s3gen.meanflow else 10)
        self.crossfade_samples = max(0, int(S3GEN_SR * crossfade_ms / 1000.0))

        self.token_buffer: list[torch.Tensor] = []
        self.noised_mels: torch.Tensor | None = None
        self.hift_cache_source = torch.zeros(
            1, 1, 0, device=s3gen.device, dtype=s3gen.dtype
        )
        self.pending_tail: torch.Tensor | None = None
        self.emitted_samples = 0
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

    def __init__(self, streamers: Iterable[S3GenStreamer]):
        self.streamers = list(streamers)
        if not self.streamers:
            raise ValueError("at least one S3GenStreamer is required")
        self.s3gen = self.streamers[0].s3gen
        if any(streamer.s3gen is not self.s3gen for streamer in self.streamers):
            raise ValueError("all streamers must share one S3Gen model")

    def flush(
        self,
        indices: Iterable[int],
        *,
        finalize: bool = False,
    ) -> list[tuple[int, torch.Tensor]]:
        groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        for index in indices:
            streamer = self.streamers[index]
            if not streamer.token_buffer:
                continue
            token_count = sum(token.shape[-1] for token in streamer.token_buffer)
            effective_tokens = token_count
            if not finalize:
                effective_tokens -= self.s3gen.flow.pre_lookahead_len
            if effective_tokens <= 0:
                continue
            groups[
                (
                    id(streamer.ref_dict),
                    token_count,
                    streamer.emitted_samples,
                    streamer.hift_cache_source.shape[-1],
                )
            ].append(index)

        outputs: list[tuple[int, torch.Tensor]] = []
        for group in groups.values():
            outputs.extend(self._decode_group(group, finalize=finalize))
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

    def _decode_group(
        self,
        indices: list[int],
        *,
        finalize: bool,
    ) -> list[tuple[int, torch.Tensor]]:
        streamers = [self.streamers[index] for index in indices]
        speech_tokens = torch.cat(
            [torch.cat(streamer.token_buffer, dim=1) for streamer in streamers],
            dim=0,
        )
        speech_token_lens = torch.full(
            (len(streamers),),
            speech_tokens.shape[-1],
            dtype=torch.long,
            device=self.s3gen.device,
        )
        effective_tokens = speech_tokens.shape[-1]
        if not finalize:
            effective_tokens -= self.s3gen.flow.pre_lookahead_len
        mel_frames = effective_tokens * self.s3gen.flow.token_mel_ratio
        noises = torch.cat(
            [streamer._ensure_noise(mel_frames) for streamer in streamers],
            dim=0,
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
        cache_source = torch.cat(
            [streamer.hift_cache_source for streamer in streamers],
            dim=0,
        )
        wavs, sources = self.s3gen.hift_inference(output_mels, cache_source)
        wavs[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade

        outputs: list[tuple[int, torch.Tensor]] = []
        for row, (index, streamer) in enumerate(zip(indices, streamers)):
            streamer.hift_cache_source = sources[row : row + 1].detach()
            wav = wavs[row : row + 1]
            chunk = None
            if wav.shape[-1] > streamer.emitted_samples:
                streamer.decoded_chunks += 1
                chunk = wav[:, streamer.emitted_samples :]
            emitted = streamer._emit_smoothed(chunk, finalize=finalize)
            if emitted is not None:
                outputs.append((index, emitted))
        return outputs
