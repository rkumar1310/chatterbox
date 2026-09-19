from __future__ import annotations

from dataclasses import dataclass, field
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
    history: list[int] = field(default_factory=list)
    waiting: bool = False
    finished: bool = False


@dataclass(frozen=True)
class T3ContinuousToken:
    request_id: str
    token: Tensor
    token_value: int
    valid: bool
    waiting: bool
    finished: bool


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
    ) -> list[T3ContinuousToken]:
        requests = list(requests)
        if not requests:
            self.reset()
            return []
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("continuous T3 request ids must be unique")
        if any(request.waiting or request.finished for request in requests):
            raise ValueError("continuous T3 step received an inactive request")

        start_token = self.t3.hp.start_speech_token
        for request in requests:
            if not request.history:
                request.history.append(start_token)

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

        sampled_tokens = []
        for index, request in enumerate(requests):
            generated_ids = torch.as_tensor(
                request.history,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            processed_logits = self.logits_processors(
                generated_ids,
                speech_logits[index : index + 1],
            )
            if torch.all(processed_logits == -float("inf")):
                raise RuntimeError(
                    f"all Turbo logits are -inf for request {request.request_id}",
                )
            probabilities = F.softmax(processed_logits, dim=-1)
            sampled_tokens.append(
                torch.multinomial(probabilities, num_samples=1)[0, 0],
            )
        next_tokens = torch.stack(sampled_tokens)
        next_token_values = next_tokens.detach().cpu().tolist()

        valid_values = []
        results = []
        for index, request in enumerate(requests):
            token_value = int(next_token_values[index])
            valid = token_value != self.t3.hp.stop_speech_token
            if not valid:
                if request.input_done:
                    request.finished = True
                else:
                    request.waiting = True
            else:
                request.history.append(token_value)
                if len(request.history) - 1 >= self.max_gen_len:
                    request.finished = True
            valid_values.append(valid)
            results.append(
                T3ContinuousToken(
                    request_id=request.request_id,
                    token=next_tokens[index : index + 1, None],
                    token_value=token_value,
                    valid=valid,
                    waiting=request.waiting,
                    finished=request.finished,
                ),
            )

        self.step_count += 1
        self.max_batch_size = max(self.max_batch_size, len(requests))
        if any(request.waiting or request.finished for request in requests):
            self._invalidate_cache()
            return results

        continuing = torch.tensor(
            valid_values,
            dtype=torch.long,
            device=self.device,
        )[:, None]
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
        return results

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
        max_speech_length = max(len(request.history) for request in requests)
        speech_tokens = torch.full(
            (batch_size, max_speech_length),
            start_token,
            dtype=torch.long,
            device=self.device,
        )
        speech_attention_mask = torch.zeros_like(speech_tokens)
        speech_lengths = []
        for index, request in enumerate(requests):
            length = len(request.history)
            speech_lengths.append(length)
            speech_tokens[index, :length] = torch.as_tensor(
                request.history,
                dtype=torch.long,
                device=self.device,
            )
            speech_attention_mask[index, :length] = 1

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
        self._last_positions = torch.tensor(
            [
                len_cond + max_text_length + length - 1
                for length in speech_lengths
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._rebuilt = True
        self.rebuild_count += 1

    def _invalidate_cache(self) -> None:
        self._outputs = None
        self._past_key_values = None
        self._attention_mask = None
        self._last_positions = None
        self._rebuilt = False
