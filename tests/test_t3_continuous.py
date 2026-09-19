from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from chatterbox.models.t3.continuous import (
    T3ContinuousBatchDecoder,
    T3ContinuousRequest,
)


class _FakeOutput:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden
        self.past_key_values = object()

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.hidden


class _FakeTransformer:
    def __init__(self, owner: "_FakeT3") -> None:
        self.owner = owner

    def __call__(self, *, inputs_embeds: torch.Tensor, **_kwargs) -> _FakeOutput:
        batch, length, _hidden = inputs_embeds.shape
        output = torch.zeros(batch, length, 1)
        output[:, :, 0] = float(self.owner.next_token)
        return _FakeOutput(output)


class _FakeT3:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.hp = SimpleNamespace(start_speech_token=0, stop_speech_token=9)
        self.next_token = 2
        self.tfmr = _FakeTransformer(self)

    def prepare_input_embeds(
        self,
        *,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        **_kwargs,
    ) -> tuple[torch.Tensor, int]:
        batch = text_tokens.shape[0]
        conditional_length = 1
        total_length = (
            conditional_length + text_tokens.shape[1] + speech_tokens.shape[1]
        )
        return torch.zeros(batch, total_length, 1), conditional_length

    def speech_head(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden.shape[0], 10), -1000.0)
        token_ids = hidden[:, 0].to(dtype=torch.long)
        logits.scatter_(1, token_ids[:, None], 1000.0)
        return logits

    def speech_emb(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.to(dtype=torch.float32).unsqueeze(-1)


def _request(request_id: str, *, version: int = 1, input_done: bool = True):
    return T3ContinuousRequest(
        request_id=request_id,
        text_tokens=torch.tensor([1, 2]),
        text_attention_mask=torch.tensor([1, 1]),
        text_version=version,
        input_done=input_done,
    )


class T3ContinuousBatchDecoderTests(unittest.TestCase):
    def test_late_request_joins_without_restarting_existing_history(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        first = _request("first")
        late = _request("late")

        decoder.step([first])
        decoder.step([first, late])

        self.assertEqual(first.history, [0, 2, 2])
        self.assertEqual(late.history, [0, 2])
        self.assertEqual(decoder.metrics()["membershipChangeCount"], 1)
        self.assertEqual(decoder.metrics()["maxBatchSize"], 2)

    def test_waiting_request_resumes_after_new_text_version(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        request = _request("live", input_done=False)
        t3.next_token = t3.hp.stop_speech_token

        first_result = decoder.step([request])[0]
        self.assertTrue(first_result.waiting)
        self.assertFalse(first_result.finished)

        request.waiting = False
        request.input_done = True
        request.text_version += 1
        second_result = decoder.step([request])[0]
        self.assertTrue(second_result.finished)
        self.assertFalse(second_result.valid)
        self.assertGreaterEqual(decoder.metrics()["rebuildCount"], 2)


if __name__ == "__main__":
    unittest.main()
