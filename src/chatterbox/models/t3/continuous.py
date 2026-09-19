from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


@dataclass
class T3ContinuousRequest:
    """Request-owned state consumed by :class:`T3ContinuousBatchDecoder`."""

    request_id: str
    text_tokens: Tensor
    text_attention_mask: Tensor
    text_version: int
    input_done: bool
    history: Tensor | None = None
    history_attention_mask: Tensor | None = None
    waiting_device: Tensor | None = None
    finished_device: Tensor | None = None
    waiting: bool = False
    finished: bool = False

    @torch.inference_mode()
    def resume(self) -> None:
        self.waiting = False
        if self.history is not None and self.history_attention_mask is not None:
            self.history = self.history[self.history_attention_mask.bool()]
            self.history_attention_mask = torch.ones_like(self.history)
        if self.waiting_device is not None:
            self.waiting_device.zero_()

    def update_host_state(self, *, waiting: bool, finished: bool) -> None:
        self.waiting = waiting
        self.finished = finished
        if waiting and self.history is not None:
            self.history = self.history[self.history_attention_mask.bool()]
            self.history_attention_mask = torch.ones_like(self.history)


@dataclass(frozen=True)
class T3ContinuousStep:
    request_ids: tuple[str, ...]
    tokens: Tensor
    valid: Tensor
    waiting: Tensor
    finished: Tensor


class T3ContinuousBatchDecoder:
    """Run one bounded Turbo speech-token step over changeable membership.

    Request histories live outside the batched KV cache. Membership changes
    rebuild one batch from those histories. Append-only live-text changes crop
    the cache to the common unchanged text prefix, append the new text tail,
    and replay speech history. Between changes the normal Hugging Face KV cache
    advances one speech token at a time.
    """

    def __init__(
        self,
        t3,
        t3_cond,
        *,
        pad_token_id: int,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
    ) -> None:
        self.t3 = t3
        self.t3_cond = t3_cond
        self.pad_token_id = int(pad_token_id)
        self.max_gen_len = int(max_gen_len)
        self.logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            self.logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            self.logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            self.logits_processors.append(
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
            )

        self._signature: tuple[tuple[str, int], ...] | None = None
        self._outputs = None
        self._past_key_values = None
        self._attention_mask: Tensor | None = None
        self._last_positions: Tensor | None = None
        self._sampling_history: Tensor | None = None
        self._sampling_history_attention_mask: Tensor | None = None
        self._text_tokens: Tensor | None = None
        self._text_attention_mask: Tensor | None = None
        self._text_lengths: tuple[int, ...] | None = None
        self._len_cond: int | None = None
        self._rebuilt = False
        self.step_count = 0
        self.rebuild_count = 0
        self.full_rebuild_count = 0
        self.incremental_update_count = 0
        self.incremental_fallback_count = 0
        self.prefix_change_fallback_count = 0
        self.unsupported_cache_fallback_count = 0
        self.reused_text_prefix_tokens = 0
        self.reused_cache_prefix_columns = 0
        self.replayed_text_tokens = 0
        self.replayed_speech_tokens = 0
        self.last_reused_text_prefix_length = 0
        self.max_reused_text_prefix_length = 0
        self.rebuild_host_dispatch_time_ms = 0.0
        self.full_rebuild_gpu_time_ms = 0.0
        self.incremental_update_gpu_time_ms = 0.0
        self._pending_rebuild_timings: list[tuple[str, Any, Any]] = []
        self._pending_replayed_speech_counts: list[tuple[Tensor, Any]] = []
        self.membership_change_count = 0
        self.max_batch_size = 0
        self.sampling_batch_call_count = 0
        self.sampled_row_count = 0

    @property
    def device(self):
        return self.t3.device

    @torch.inference_mode()
    def step(
        self,
        requests: Sequence[T3ContinuousRequest],
    ) -> T3ContinuousStep | None:
        requests = list(requests)
        if not requests:
            self.reset()
            return None
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("continuous T3 request ids must be unique")
        start_token = self.t3.hp.start_speech_token
        for request in requests:
            if request.history is None:
                request.history = torch.tensor(
                    [start_token],
                    dtype=torch.long,
                    device=self.device,
                )
                request.history_attention_mask = torch.ones_like(request.history)
            if request.waiting_device is None:
                request.waiting_device = torch.tensor(
                    request.waiting,
                    dtype=torch.bool,
                    device=self.device,
                )
            if request.finished_device is None:
                request.finished_device = torch.tensor(
                    request.finished,
                    dtype=torch.bool,
                    device=self.device,
                )

        signature = tuple(
            (request.request_id, request.text_version) for request in requests
        )
        membership_changed = False
        if self._signature is not None:
            membership_changed = tuple(
                request_id for request_id, _ in signature
            ) != tuple(request_id for request_id, _ in self._signature)
            if membership_changed:
                self.membership_change_count += 1

        if self._outputs is None or self._signature is None or membership_changed:
            self._rebuild(requests)
        elif signature != self._signature:
            plan, fallback_reason = self._incremental_plan(requests)
            if plan is None or not self._cache_supports_crop():
                self.incremental_fallback_count += 1
                if fallback_reason == "prefix_changed":
                    self.prefix_change_fallback_count += 1
                else:
                    self.unsupported_cache_fallback_count += 1
                self._rebuild(requests)
            elif not self._incremental_rebuild(requests, *plan):
                self.incremental_fallback_count += 1
                self.unsupported_cache_fallback_count += 1
                self._rebuild(requests)
        self._signature = signature

        if self._rebuilt:
            rows = torch.arange(len(requests), device=self.device)
            hidden = self._outputs[0][rows, self._last_positions]
        else:
            hidden = self._outputs[0][:, -1, :]
        speech_logits = self.t3.speech_head(hidden)

        waiting_before = torch.stack(
            [request.waiting_device for request in requests],
        )
        finished_before = torch.stack(
            [request.finished_device for request in requests],
        )
        active = ~waiting_before & ~finished_before

        next_tokens = self._sample_batch(
            speech_logits,
            active=active,
        )
        emitted_stop = active & (next_tokens == self.t3.hp.stop_speech_token)
        valid = active & ~emitted_stop
        input_done = torch.tensor(
            [request.input_done for request in requests],
            dtype=torch.bool,
            device=self.device,
        )
        generated_counts = torch.stack(
            [request.history_attention_mask.sum() - 1 for request in requests],
        )
        reached_limit = valid & (generated_counts + 1 >= self.max_gen_len)
        waiting = waiting_before | (emitted_stop & ~input_done)
        finished = finished_before | (emitted_stop & input_done) | reached_limit

        self._sampling_history = torch.cat(
            [self._sampling_history, next_tokens[:, None]],
            dim=1,
        )
        self._sampling_history_attention_mask = torch.cat(
            [
                self._sampling_history_attention_mask,
                valid.to(dtype=torch.long)[:, None],
            ],
            dim=1,
        )
        for index, request in enumerate(requests):
            request.history = self._sampling_history[index]
            request.history_attention_mask = self._sampling_history_attention_mask[
                index
            ]
            request.waiting_device = waiting[index]
            request.finished_device = finished[index]

        self.step_count += 1
        self.max_batch_size = max(self.max_batch_size, len(requests))
        continuing = (valid & ~finished).to(dtype=torch.long)[:, None]
        self._attention_mask = torch.cat(
            [self._attention_mask, continuing],
            dim=1,
        )
        position_ids = self._attention_mask.cumsum(-1)[:, -1:] - 1
        position_ids.masked_fill_(continuing == 0, 0)
        speech_embed = self.t3.speech_emb(next_tokens[:, None])
        self._outputs = self.t3.tfmr(
            inputs_embeds=speech_embed,
            attention_mask=self._attention_mask,
            position_ids=position_ids,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        self._past_key_values = self._outputs.past_key_values
        self._rebuilt = False
        return T3ContinuousStep(
            request_ids=tuple(request_ids),
            tokens=next_tokens,
            valid=valid,
            waiting=waiting,
            finished=finished,
        )

    def reset(self) -> None:
        self._signature = None
        self._invalidate_cache()

    def metrics(self) -> dict[str, int | float]:
        self._collect_rebuild_timings()
        return {
            "stepCount": self.step_count,
            "rebuildCount": self.rebuild_count,
            "fullRebuildCount": self.full_rebuild_count,
            "incrementalUpdateCount": self.incremental_update_count,
            "incrementalFallbackCount": self.incremental_fallback_count,
            "prefixChangeFallbackCount": self.prefix_change_fallback_count,
            "unsupportedCacheFallbackCount": self.unsupported_cache_fallback_count,
            "reusedTextPrefixTokens": self.reused_text_prefix_tokens,
            "reusedCachePrefixColumns": self.reused_cache_prefix_columns,
            "replayedTextTokens": self.replayed_text_tokens,
            "replayedSpeechTokens": self.replayed_speech_tokens,
            "lastReusedTextPrefixLength": self.last_reused_text_prefix_length,
            "maxReusedTextPrefixLength": self.max_reused_text_prefix_length,
            "rebuildHostDispatchTimeMs": round(
                self.rebuild_host_dispatch_time_ms,
                4,
            ),
            "fullRebuildGpuTimeMs": round(self.full_rebuild_gpu_time_ms, 4),
            "incrementalUpdateGpuTimeMs": round(
                self.incremental_update_gpu_time_ms,
                4,
            ),
            "pendingRebuildTimings": len(self._pending_rebuild_timings),
            "membershipChangeCount": self.membership_change_count,
            "maxBatchSize": self.max_batch_size,
            "samplingBatchCalls": self.sampling_batch_call_count,
            "sampledRows": self.sampled_row_count,
            "samplingPythonRowIterations": 0,
        }

    def _processed_sampling_logits(self, speech_logits: Tensor) -> Tensor:
        """Apply the sampling policy once to the complete GPU batch.

        Invalid history positions are replaced with the row's first valid
        token. Repetition penalty depends on the set of token ids that has
        appeared, so repeating an already-valid id preserves the unpadded
        per-request policy without introducing a padding token.
        """
        if (
            self._sampling_history is None
            or self._sampling_history_attention_mask is None
        ):
            raise RuntimeError("continuous T3 sampling history is not initialized")
        first_valid_token = self._sampling_history[:, :1]
        sampling_ids = torch.where(
            self._sampling_history_attention_mask.bool(),
            self._sampling_history,
            first_valid_token,
        )
        return self.logits_processors(sampling_ids, speech_logits)

    def _sample_batch(self, speech_logits: Tensor, *, active: Tensor) -> Tensor:
        processed_logits = self._processed_sampling_logits(speech_logits)
        probabilities = F.softmax(processed_logits, dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        self.sampling_batch_call_count += 1
        self.sampled_row_count += int(speech_logits.shape[0])
        return torch.where(
            active,
            sampled,
            torch.full_like(sampled, self.t3.hp.stop_speech_token),
        )

    def _batch_text(
        self,
        requests: Sequence[T3ContinuousRequest],
    ) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        batch_size = len(requests)
        max_text_length = max(request.text_tokens.shape[-1] for request in requests)
        text_tokens = torch.full(
            (batch_size, max_text_length),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        text_attention_mask = torch.zeros_like(text_tokens)
        for index, request in enumerate(requests):
            length = request.text_tokens.shape[-1]
            text_tokens[index, :length] = request.text_tokens.to(
                device=self.device,
                dtype=torch.long,
            )
            text_attention_mask[index, :length] = request.text_attention_mask.to(
                device=self.device,
                dtype=torch.long,
            )
        return (
            text_tokens,
            text_attention_mask,
            tuple(request.text_tokens.shape[-1] for request in requests),
        )

    def _batch_speech(
        self,
        requests: Sequence[T3ContinuousRequest],
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(requests)
        start_token = self.t3.hp.start_speech_token
        max_speech_length = max(request.history.shape[-1] for request in requests)
        speech_tokens = torch.full(
            (batch_size, max_speech_length),
            start_token,
            dtype=torch.long,
            device=self.device,
        )
        speech_attention_mask = torch.zeros_like(speech_tokens)
        for index, request in enumerate(requests):
            length = request.history.shape[-1]
            speech_tokens[index, :length] = request.history.to(
                device=self.device,
                dtype=torch.long,
            )
            speech_attention_mask[index, :length] = request.history_attention_mask.to(
                device=self.device,
                dtype=torch.long,
            )
        return speech_tokens, speech_attention_mask

    def _rebuild(self, requests: Sequence[T3ContinuousRequest]) -> None:
        timing = self._begin_rebuild_timing()
        batch_size = len(requests)
        text_tokens, text_attention_mask, text_lengths = self._batch_text(requests)
        speech_tokens, speech_attention_mask = self._batch_speech(requests)

        embeds, len_cond = self.t3.prepare_input_embeds(
            t3_cond=self.t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
            cfg_weight=0.0,
        )
        attention_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    len_cond,
                    dtype=torch.long,
                    device=self.device,
                ),
                text_attention_mask,
                speech_attention_mask,
            ],
            dim=1,
        )
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        outputs = self.t3.tfmr(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        self._outputs = outputs
        self._past_key_values = outputs.past_key_values
        self._attention_mask = attention_mask
        self._sampling_history = speech_tokens
        self._sampling_history_attention_mask = speech_attention_mask
        self._text_tokens = text_tokens
        self._text_attention_mask = text_attention_mask
        self._text_lengths = text_lengths
        self._len_cond = int(len_cond)
        speech_physical_positions = torch.arange(
            speech_tokens.shape[1],
            dtype=torch.long,
            device=self.device,
        ).expand(batch_size, -1)
        last_speech_positions = speech_physical_positions.masked_fill(
            speech_attention_mask == 0,
            -1,
        ).amax(dim=1)
        self._last_positions = (
            len_cond + text_tokens.shape[1] + last_speech_positions
        )
        self._rebuilt = True
        self.rebuild_count += 1
        self.full_rebuild_count += 1
        self._finish_rebuild_timing("full", timing)

    def _incremental_plan(
        self,
        requests: Sequence[T3ContinuousRequest],
    ) -> tuple[tuple[Tensor, Tensor, tuple[int, ...], int] | None, str | None]:
        if (
            self._text_tokens is None
            or self._text_attention_mask is None
            or self._text_lengths is None
            or len(self._text_lengths) != len(requests)
        ):
            return None, "cache_unsupported"

        prefix_length = self._text_tokens.shape[1]
        for index, request in enumerate(requests):
            old_length = self._text_lengths[index]
            new_length = request.text_tokens.shape[-1]
            if new_length < old_length:
                return None, "prefix_changed"
            new_tokens = request.text_tokens.to(
                device=self.device,
                dtype=torch.long,
            )
            new_mask = request.text_attention_mask.to(
                device=self.device,
                dtype=torch.long,
            )
            if not torch.equal(
                new_tokens[:old_length],
                self._text_tokens[index, :old_length],
            ) or not torch.equal(
                new_mask[:old_length],
                self._text_attention_mask[index, :old_length],
            ):
                return None, "prefix_changed"
            if new_length > old_length:
                prefix_length = min(prefix_length, old_length)

        text_tokens, text_attention_mask, text_lengths = self._batch_text(requests)
        return (
            text_tokens,
            text_attention_mask,
            text_lengths,
            prefix_length,
        ), None

    def _incremental_rebuild(
        self,
        requests: Sequence[T3ContinuousRequest],
        text_tokens: Tensor,
        text_attention_mask: Tensor,
        text_lengths: tuple[int, ...],
        prefix_length: int,
    ) -> bool:
        timing = self._begin_rebuild_timing()
        batch_size = len(requests)
        speech_tokens, speech_attention_mask = self._batch_speech(requests)
        embeds, len_cond = self.t3.prepare_input_embeds(
            t3_cond=self.t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
            cfg_weight=0.0,
        )
        if self._len_cond is None or int(len_cond) != self._len_cond:
            return False

        prefix_cache_length = int(len_cond) + prefix_length
        past_key_values = self._crop_cache(prefix_cache_length)
        if past_key_values is None:
            return False

        attention_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    len_cond,
                    dtype=torch.long,
                    device=self.device,
                ),
                text_attention_mask,
                speech_attention_mask,
            ],
            dim=1,
        )
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        outputs = self.t3.tfmr(
            inputs_embeds=embeds[:, prefix_cache_length:],
            attention_mask=attention_mask,
            position_ids=position_ids[:, prefix_cache_length:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        self._outputs = outputs
        self._past_key_values = outputs.past_key_values
        self._attention_mask = attention_mask
        self._sampling_history = speech_tokens
        self._sampling_history_attention_mask = speech_attention_mask
        self._text_tokens = text_tokens
        self._text_attention_mask = text_attention_mask
        self._text_lengths = text_lengths
        self._len_cond = int(len_cond)
        speech_physical_positions = torch.arange(
            speech_tokens.shape[1],
            dtype=torch.long,
            device=self.device,
        ).expand(batch_size, -1)
        last_speech_positions = speech_physical_positions.masked_fill(
            speech_attention_mask == 0,
            -1,
        ).amax(dim=1)
        full_last_positions = (
            len_cond + text_tokens.shape[1] + last_speech_positions
        )
        self._last_positions = full_last_positions - prefix_cache_length
        self._rebuilt = True

        reused_text_tokens = sum(
            min(length, prefix_length) for length in text_lengths
        )
        replayed_text_tokens = sum(
            max(0, length - prefix_length) for length in text_lengths
        )
        replayed_speech_tokens = speech_attention_mask.sum() - batch_size
        self.reused_text_prefix_tokens += reused_text_tokens
        self.reused_cache_prefix_columns += batch_size * prefix_cache_length
        self.replayed_text_tokens += replayed_text_tokens
        self.last_reused_text_prefix_length = prefix_length
        self.max_reused_text_prefix_length = max(
            self.max_reused_text_prefix_length,
            prefix_length,
        )
        self.rebuild_count += 1
        self.incremental_update_count += 1
        completion_event = self._finish_rebuild_timing("incremental", timing)
        if completion_event is None:
            self.replayed_speech_tokens += max(
                0,
                int(replayed_speech_tokens.item()),
            )
        else:
            self._pending_replayed_speech_counts.append(
                (replayed_speech_tokens, completion_event),
            )
        return True

    def _cache_supports_crop(self) -> bool:
        cache = self._past_key_values
        if cache is None:
            return False
        if callable(getattr(cache, "crop", None)):
            return True
        return isinstance(cache, tuple) and all(
            isinstance(layer, tuple)
            and len(layer) >= 2
            and torch.is_tensor(layer[0])
            and torch.is_tensor(layer[1])
            for layer in cache
        )

    def _crop_cache(self, length: int):
        cache = self._past_key_values
        crop = getattr(cache, "crop", None)
        if callable(crop):
            crop(length)
            return cache
        if not self._cache_supports_crop():
            return None
        return tuple(
            (
                layer[0][..., :length, :],
                layer[1][..., :length, :],
                *layer[2:],
            )
            for layer in cache
        )

    def _begin_rebuild_timing(self) -> tuple[float, Any | None]:
        host_start = time.perf_counter()
        if torch.device(self.device).type != "cuda":
            return host_start, None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return host_start, start

    def _finish_rebuild_timing(
        self,
        kind: str,
        timing: tuple[float, Any | None],
    ) -> Any | None:
        host_start, cuda_start = timing
        host_elapsed_ms = (time.perf_counter() - host_start) * 1_000
        self.rebuild_host_dispatch_time_ms += host_elapsed_ms
        if cuda_start is None:
            if kind == "full":
                self.full_rebuild_gpu_time_ms += host_elapsed_ms
            else:
                self.incremental_update_gpu_time_ms += host_elapsed_ms
            return None
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_end.record()
        self._pending_rebuild_timings.append((kind, cuda_start, cuda_end))
        return cuda_end

    def _collect_rebuild_timings(self) -> None:
        pending = []
        for kind, start, end in self._pending_rebuild_timings:
            if not end.query():
                pending.append((kind, start, end))
                continue
            elapsed_ms = start.elapsed_time(end)
            if kind == "full":
                self.full_rebuild_gpu_time_ms += elapsed_ms
            else:
                self.incremental_update_gpu_time_ms += elapsed_ms
        self._pending_rebuild_timings = pending
        pending_counts = []
        for count, event in self._pending_replayed_speech_counts:
            if not event.query():
                pending_counts.append((count, event))
                continue
            self.replayed_speech_tokens += max(0, int(count.item()))
        self._pending_replayed_speech_counts = pending_counts

    def _invalidate_cache(self) -> None:
        self._outputs = None
        self._past_key_values = None
        self._attention_mask = None
        self._last_positions = None
        self._sampling_history = None
        self._sampling_history_attention_mask = None
        self._text_tokens = None
        self._text_attention_mask = None
        self._text_lengths = None
        self._len_cond = None
        self._rebuilt = False
