from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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

    Request histories live outside the batched KV cache. When membership or
    live text changes, the decoder rebuilds one batch from those histories.
    Between changes it retains and advances the normal Hugging Face KV cache.
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
        self._rebuilt = False
        self.step_count = 0
        self.rebuild_count = 0
        self.membership_change_count = 0
        self.max_batch_size = 0

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
        if signature != self._signature or self._outputs is None:
            if self._signature is not None and tuple(
                request_id for request_id, _ in signature
            ) != tuple(request_id for request_id, _ in self._signature):
                self.membership_change_count += 1
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

        sampled_tokens = []
        for index, request in enumerate(requests):
            generated_ids = request.history.unsqueeze(0)
            processed_logits = self.logits_processors(
                generated_ids,
                speech_logits[index : index + 1],
            )
            probabilities = F.softmax(processed_logits, dim=-1)
            sampled_tokens.append(
                torch.multinomial(probabilities, num_samples=1)[0, 0],
            )
        next_tokens = torch.stack(sampled_tokens)
        next_tokens = torch.where(
            active,
            next_tokens,
            torch.full_like(next_tokens, self.t3.hp.stop_speech_token),
        )
        emitted_stop = active & (next_tokens == self.t3.hp.stop_speech_token)
        valid = active & ~emitted_stop
        input_done = torch.tensor(
            [request.input_done for request in requests],
            dtype=torch.bool,
            device=self.device,
        )
        generated_counts = torch.stack(
            [
                request.history_attention_mask.sum() - 1
                for request in requests
            ],
        )
        reached_limit = valid & (generated_counts + 1 >= self.max_gen_len)
        waiting = waiting_before | (emitted_stop & ~input_done)
        finished = finished_before | (emitted_stop & input_done) | reached_limit

        for index, request in enumerate(requests):
            request.history = torch.cat(
                [request.history, next_tokens[index : index + 1]],
            )
            request.history_attention_mask = torch.cat(
                [
                    request.history_attention_mask,
                    valid[index : index + 1].to(dtype=torch.long),
                ],
            )
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

    def metrics(self) -> dict[str, int]:
        return {
            "stepCount": self.step_count,
            "rebuildCount": self.rebuild_count,
            "membershipChangeCount": self.membership_change_count,
            "maxBatchSize": self.max_batch_size,
        }

    def _rebuild(self, requests: Sequence[T3ContinuousRequest]) -> None:
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
            speech_attention_mask[index, :length] = (
                request.history_attention_mask.to(
                    device=self.device,
                    dtype=torch.long,
                )
            )

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
        speech_physical_positions = torch.arange(
            max_speech_length,
            dtype=torch.long,
            device=self.device,
        ).expand(batch_size, -1)
        last_speech_positions = speech_physical_positions.masked_fill(
            speech_attention_mask == 0,
            -1,
        ).amax(dim=1)
        self._last_positions = (
            len_cond + max_text_length + last_speech_positions
        )
        self._rebuilt = True
        self.rebuild_count += 1

    def _invalidate_cache(self) -> None:
        self._outputs = None
        self._past_key_values = None
        self._attention_mask = None
        self._last_positions = None
        self._rebuilt = False
