from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

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
    def test_vectorized_sampling_matches_per_request_policy(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            temperature=0.7,
            top_k=6,
            top_p=0.8,
            repetition_penalty=1.2,
        )
        decoder._sampling_history = torch.tensor(
            [
                [0, 1, 2, 0],
                [0, 3, 0, 0],
                [0, 1, 4, 5],
            ],
        )
        decoder._sampling_history_attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
        )
        logits = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.9, 0.1, 1.0],
            ],
        )

        reference_rows = []
        for index in range(logits.shape[0]):
            history = decoder._sampling_history[index][
                decoder._sampling_history_attention_mask[index].bool()
            ].unsqueeze(0)
            reference_rows.append(
                decoder.logits_processors(
                    history,
                    logits[index : index + 1],
                ),
            )
        reference = torch.cat(reference_rows)

        vectorized = decoder._processed_sampling_logits(logits)

        self.assertTrue(
            torch.equal(torch.isfinite(vectorized), torch.isfinite(reference))
        )
        self.assertTrue(
            torch.allclose(
                torch.nan_to_num(vectorized),
                torch.nan_to_num(reference),
            ),
        )
        self.assertTrue(
            torch.allclose(
                F.softmax(vectorized, dim=-1),
                F.softmax(reference, dim=-1),
            ),
        )

    def test_sampling_uses_one_multinomial_call_for_the_batch(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        batch_size = 8
        decoder._sampling_history = torch.zeros(batch_size, 3, dtype=torch.long)
        decoder._sampling_history_attention_mask = torch.ones(
            batch_size,
            3,
            dtype=torch.long,
        )
        logits = torch.arange(10, dtype=torch.float32).repeat(batch_size, 1)
        active = torch.tensor([True, True, False, True, True, False, True, True])

        with patch("torch.multinomial", wraps=torch.multinomial) as multinomial:
            sampled = decoder._sample_batch(logits, active=active)

        self.assertEqual(multinomial.call_count, 1)
        self.assertEqual(sampled.tolist(), [9, 9, 9, 9, 9, 9, 9, 9])
        self.assertEqual(decoder.metrics()["samplingBatchCalls"], 1)
        self.assertEqual(decoder.metrics()["sampledRows"], batch_size)
        self.assertEqual(decoder.metrics()["samplingPythonRowIterations"], 0)

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

        self.assertEqual(
            first.history[first.history_attention_mask.bool()].tolist(),
            [0, 2, 2],
        )
        self.assertEqual(
            late.history[late.history_attention_mask.bool()].tolist(),
            [0, 2],
        )
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

        first_result = decoder.step([request])
        self.assertTrue(first_result.waiting.item())
        self.assertFalse(first_result.finished.item())

        request.resume()
        request.input_done = True
        request.text_version += 1
        second_result = decoder.step([request])
        self.assertTrue(second_result.finished.item())
        self.assertFalse(second_result.valid.item())
        self.assertGreaterEqual(decoder.metrics()["rebuildCount"], 2)


if __name__ == "__main__":
    unittest.main()
