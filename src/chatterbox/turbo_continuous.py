from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .models.s3gen import (
    S3GenBatchingMetrics,
    S3GenBatchStreamer,
    S3GenStreamer,
)
from .models.s3tokenizer import SPEECH_VOCAB_SIZE
from .models.t3 import T3ContinuousBatchDecoder, T3ContinuousRequest
from .streaming import StreamingAudioChunk


@dataclass(frozen=True)
class ContinuousAudioOutput:
    request_id: str
    chunk: StreamingAudioChunk


@dataclass(frozen=True)
class ContinuousStepResult:
    """One bounded speech-token step and any audio it made ready."""

    audio: tuple[ContinuousAudioOutput, ...]
    active_request_ids: tuple[str, ...]
    waiting_request_ids: tuple[str, ...]
    finished_request_ids: tuple[str, ...]
    cancelled_request_ids: tuple[str, ...]
    first_speech_token_request_ids: tuple[str, ...]
    t3_batch_size: int


@dataclass
class _ContinuousRequestState:
    request_id: str
    text_source: Any
    accepted_snapshot: Any
    t3_request: T3ContinuousRequest
    streamer: S3GenStreamer
    pending_since: float | None = None
    chunk_index: int = 0
    next_sample: int = 0
    first_speech_token_reported: bool = False
    next_flush_token: int = 0
    pending_tokens: list[torch.Tensor] = field(default_factory=list)
    pending_valid: list[torch.Tensor] = field(default_factory=list)
    last_waiting: torch.Tensor | None = None
    last_finished: torch.Tensor | None = None


class ChatterboxTurboContinuousEngine:
    """Request-keyed Turbo engine whose batch membership can change per step.

    The engine deliberately executes one bounded T3 speech-token step at a
    time. A caller can therefore add a newly ready request or remove a
    cancelled request before the next step without restarting the T3 history
    or S3Gen streamer of requests already in flight.
    """

    def __init__(
        self,
        *,
        t3: Any,
        s3gen: Any,
        tokenizer: Any,
        t3_conditionals: Any,
        s3gen_conditionals: dict,
        sample_rate: int,
        normalize_text: Callable[..., str],
        chunk_tokens: int = 24,
        max_gen_len: int = 1000,
        crossfade_ms: float = 12.0,
        decoder_left_context_tokens: int = 25,
        s3gen_bucket_width_tokens: int = 8,
        min_update_chars: int = 16,
        max_update_latency_seconds: float = 0.12,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        use_cuda_graph: bool = True,
    ) -> None:
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        if min_update_chars < 1:
            raise ValueError("min_update_chars must be positive")
        if max_update_latency_seconds <= 0:
            raise ValueError("max_update_latency_seconds must be positive")

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Turbo tokenizer must define a pad or EOS token")

        self.t3 = t3
        self.s3gen = s3gen
        self.tokenizer = tokenizer
        self.t3_conditionals = t3_conditionals
        self.s3gen_conditionals = s3gen_conditionals
        self.sample_rate = int(sample_rate)
        self.normalize_text = normalize_text
        self.chunk_tokens = int(chunk_tokens)
        self.crossfade_ms = float(crossfade_ms)
        self.decoder_left_context_tokens = int(decoder_left_context_tokens)
        self.s3gen_bucket_width_tokens = int(s3gen_bucket_width_tokens)
        if self.s3gen_bucket_width_tokens < 1:
            raise ValueError("S3Gen bucket width must be positive")
        self._s3gen_batching_metrics = S3GenBatchingMetrics(
            self.s3gen_bucket_width_tokens,
        )
        self.min_update_chars = int(min_update_chars)
        self.max_update_latency_seconds = float(max_update_latency_seconds)
        self.cuda_graph_requested = bool(use_cuda_graph)
        # Dynamic membership invalidates a captured graph's batch-shaped cache.
        # Task 1 uses the safe eager path and exposes that fallback explicitly.
        self.cuda_graph_enabled = False
        self.cuda_graph_fallback_count = int(self.cuda_graph_requested)
        self._requests: dict[str, _ContinuousRequestState] = {}
        self._decoder = T3ContinuousBatchDecoder(
            t3,
            t3_conditionals,
            pad_token_id=int(pad_token_id),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_gen_len=max_gen_len,
        )
        self.step_count = 0
        self.max_active_requests = 0
        self._steps_since_host_transfer = 0
        self.host_transfer_count = 0
        self.host_transferred_values = 0

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def add_request(self, request_id: str, text_source: Any) -> None:
        if request_id in self._requests:
            raise ValueError(f"duplicate continuous request id: {request_id}")
        snapshot = text_source.snapshot()
        if snapshot.cancelled:
            raise RuntimeError(f"request is already cancelled: {request_id}")
        if not snapshot.text.strip():
            raise ValueError("continuous Turbo request requires initial text")
        text_tokens, text_attention_mask = self._tokenize(snapshot)
        self._requests[request_id] = _ContinuousRequestState(
            request_id=request_id,
            text_source=text_source,
            accepted_snapshot=snapshot,
            t3_request=T3ContinuousRequest(
                request_id=request_id,
                text_tokens=text_tokens,
                text_attention_mask=text_attention_mask,
                text_version=int(snapshot.version),
                input_done=bool(snapshot.input_done),
            ),
            streamer=S3GenStreamer(
                self.s3gen,
                self.s3gen_conditionals,
                n_cfm_timesteps=2,
                crossfade_ms=self.crossfade_ms,
                left_context_tokens=self.decoder_left_context_tokens,
            ),
            next_flush_token=self.chunk_tokens,
        )

    @torch.inference_mode()
    def step(self) -> ContinuousStepResult:
        cancelled = self._refresh_requests()
        active_states = [
            state
            for state in self._requests.values()
            if not state.t3_request.waiting and not state.t3_request.finished
        ]
        active_ids = tuple(state.request_id for state in active_states)
        self.max_active_requests = max(self.max_active_requests, len(active_states))

        if not active_states:
            self._decoder.step([])
            self.step_count += 1
            return ContinuousStepResult(
                audio=(),
                active_request_ids=(),
                waiting_request_ids=self._waiting_request_ids(),
                finished_request_ids=(),
                cancelled_request_ids=tuple(cancelled),
                first_speech_token_request_ids=(),
                t3_batch_size=0,
            )

        t3_step = self._decoder.step(
            [state.t3_request for state in active_states],
        )
        if t3_step is None:
            raise RuntimeError("continuous T3 returned no step for active requests")
        state_by_id = {state.request_id: state for state in active_states}
        for index, request_id in enumerate(t3_step.request_ids):
            state = state_by_id[request_id]
            state.pending_tokens.append(t3_step.tokens[index : index + 1])
            state.pending_valid.append(t3_step.valid[index : index + 1])
            state.last_waiting = t3_step.waiting[index]
            state.last_finished = t3_step.finished[index]

        self._steps_since_host_transfer += 1
        self.step_count += 1
        if self._steps_since_host_transfer < self.chunk_tokens:
            return ContinuousStepResult(
                audio=(),
                active_request_ids=active_ids,
                waiting_request_ids=self._waiting_request_ids(),
                finished_request_ids=(),
                cancelled_request_ids=tuple(cancelled),
                first_speech_token_request_ids=(),
                t3_batch_size=len(active_states),
            )

        boundary_states = [
            state for state in self._requests.values() if state.pending_tokens
        ]
        host_rows = self._transfer_boundary_state(boundary_states)
        first_speech_token_ids: list[str] = []
        flush_ids: list[str] = []
        finish_ids: list[str] = []

        for state, row in zip(boundary_states, host_rows):
            for token, token_value, is_valid in zip(
                state.pending_tokens,
                row["tokens"],
                row["valid"],
            ):
                if not is_valid:
                    continue
                if not state.first_speech_token_reported:
                    state.first_speech_token_reported = True
                    first_speech_token_ids.append(state.request_id)
                if token_value < SPEECH_VOCAB_SIZE:
                    state.streamer.append(token[:, None])

            state.t3_request.update_host_state(
                waiting=bool(row["waiting"]),
                finished=bool(row["finished"]),
            )
            if state.t3_request.finished:
                finish_ids.append(state.request_id)
            elif state.streamer.generated_tokens >= state.next_flush_token:
                flush_ids.append(state.request_id)
                while state.next_flush_token <= state.streamer.generated_tokens:
                    state.next_flush_token += self.chunk_tokens
            state.pending_tokens.clear()
            state.pending_valid.clear()
            state.last_waiting = None
            state.last_finished = None

        self._steps_since_host_transfer = 0

        finish_set = set(finish_ids)
        flush_ids = [request_id for request_id in flush_ids if request_id not in finish_set]
        decode_ids = [state.request_id for state in self._requests.values()]
        decode_states = [self._requests[request_id] for request_id in decode_ids]
        decode_index = {request_id: index for index, request_id in enumerate(decode_ids)}
        batch_streamer = S3GenBatchStreamer(
            [state.streamer for state in decode_states],
            bucket_width_tokens=self.s3gen_bucket_width_tokens,
            metrics=self._s3gen_batching_metrics,
        )

        audio: list[ContinuousAudioOutput] = []
        if finish_ids:
            for index, wav in batch_streamer.finish(
                [decode_index[request_id] for request_id in finish_ids],
            ):
                state = decode_states[index]
                audio.append(
                    ContinuousAudioOutput(
                        request_id=state.request_id,
                        chunk=self._make_chunk(state, wav, is_final=True),
                    ),
                )
        if flush_ids:
            for index, wav in batch_streamer.flush(
                [decode_index[request_id] for request_id in flush_ids],
            ):
                state = decode_states[index]
                audio.append(
                    ContinuousAudioOutput(
                        request_id=state.request_id,
                        chunk=self._make_chunk(state, wav, is_final=False),
                    ),
                )

        for request_id in finish_ids:
            self._requests.pop(request_id, None)

        return ContinuousStepResult(
            audio=tuple(audio),
            active_request_ids=active_ids,
            waiting_request_ids=self._waiting_request_ids(),
            finished_request_ids=tuple(finish_ids),
            cancelled_request_ids=tuple(cancelled),
            first_speech_token_request_ids=tuple(first_speech_token_ids),
            t3_batch_size=len(active_states),
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "stepCount": self.step_count,
            "requestCount": len(self._requests),
            "maxActiveRequests": self.max_active_requests,
            "speechTokenSteps": self.step_count,
            "hostTransferCount": self.host_transfer_count,
            "hostTransferredValues": self.host_transferred_values,
            "perSpeechTokenHostTransfers": 0,
            "cudaGraphRequested": self.cuda_graph_requested,
            "cudaGraphEnabled": self.cuda_graph_enabled,
            "cudaGraphFallbackCount": self.cuda_graph_fallback_count,
            "t3": self._decoder.metrics(),
            "s3gen": self._s3gen_batching_metrics.as_dict(),
        }

    def _refresh_requests(self) -> list[str]:
        now = time.monotonic()
        cancelled: list[str] = []
        for request_id, state in list(self._requests.items()):
            snapshot = state.text_source.snapshot()
            if snapshot.cancelled:
                cancelled.append(request_id)
                self._requests.pop(request_id, None)
                continue
            if snapshot.version == state.accepted_snapshot.version:
                state.pending_since = None
                continue
            if state.pending_since is None:
                state.pending_since = now
            growth = len(snapshot.text) - len(state.accepted_snapshot.text)
            should_accept = (
                snapshot.input_done
                or growth >= self.min_update_chars
                or now - state.pending_since >= self.max_update_latency_seconds
            )
            if not should_accept:
                continue
            text_tokens, text_attention_mask = self._tokenize(snapshot)
            state.accepted_snapshot = snapshot
            state.pending_since = None
            state.t3_request.text_tokens = text_tokens
            state.t3_request.text_attention_mask = text_attention_mask
            state.t3_request.text_version = int(snapshot.version)
            state.t3_request.input_done = bool(snapshot.input_done)
            state.t3_request.resume()
        return cancelled

    def _transfer_boundary_state(
        self,
        states: list[_ContinuousRequestState],
    ) -> list[dict[str, Any]]:
        """Copy tokens and status once at a PCM-sized scheduling boundary."""
        if not states:
            return []
        token_lengths = [len(state.pending_tokens) for state in states]
        tokens = torch.cat(
            [torch.cat(state.pending_tokens) for state in states],
        ).to(dtype=torch.long)
        valid = torch.cat(
            [torch.cat(state.pending_valid) for state in states],
        ).to(dtype=torch.long)
        waiting = torch.stack([state.last_waiting for state in states]).to(
            dtype=torch.long,
        )
        finished = torch.stack([state.last_finished for state in states]).to(
            dtype=torch.long,
        )
        payload = torch.cat([tokens, valid, waiting, finished])
        host = payload.detach().cpu().tolist()
        token_total = sum(token_lengths)
        token_values = host[:token_total]
        valid_values = host[token_total : 2 * token_total]
        waiting_values = host[2 * token_total : 2 * token_total + len(states)]
        finished_values = host[2 * token_total + len(states) :]

        rows = []
        offset = 0
        for index, length in enumerate(token_lengths):
            rows.append(
                {
                    "tokens": token_values[offset : offset + length],
                    "valid": valid_values[offset : offset + length],
                    "waiting": waiting_values[index],
                    "finished": finished_values[index],
                },
            )
            offset += length
        self.host_transfer_count += 1
        self.host_transferred_values += len(host)
        return rows

    def _tokenize(self, snapshot: Any) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalize_text(
            snapshot.text,
            add_terminal_punctuation=bool(snapshot.input_done),
        )
        tokenized = self.tokenizer(
            normalized,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        return (
            tokenized.input_ids[0].to(device=self.t3.device, dtype=torch.long),
            tokenized.attention_mask[0].to(
                device=self.t3.device,
                dtype=torch.long,
            ),
        )

    def _make_chunk(
        self,
        state: _ContinuousRequestState,
        wav: torch.Tensor,
        *,
        is_final: bool,
    ) -> StreamingAudioChunk:
        # Task 6 replaces this synchronous transfer with GPU-side PCM16 and a
        # pinned-memory copy stream. Keep the existing behavior for Task 1.
        audio = wav.detach().to(device="cpu", dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        start_sample = state.next_sample
        end_sample = start_sample + audio.shape[-1]
        chunk = StreamingAudioChunk(
            audio=audio,
            sample_rate=self.sample_rate,
            index=state.chunk_index,
            is_final=is_final,
            start_sample=start_sample,
            end_sample=end_sample,
            generated_tokens=state.streamer.generated_tokens,
            watermarked=False,
        )
        state.chunk_index += 1
        state.next_sample = end_sample
        return chunk

    def _waiting_request_ids(self) -> tuple[str, ...]:
        return tuple(
            state.request_id
            for state in self._requests.values()
            if state.t3_request.waiting
        )
