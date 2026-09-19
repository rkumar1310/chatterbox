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


class _FakeCache:
    def __init__(self, length: int = 0) -> None:
        self.length = length
        self.crops = []

    def crop(self, length: int) -> None:
        self.length = length
        self.crops.append(length)


class _FakeOutput:
    def __init__(self, hidden: torch.Tensor, past_key_values=None) -> None:
        self.hidden = hidden
        self.past_key_values = past_key_values or _FakeCache(hidden.shape[1])

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.hidden


class _FakeTransformer:
    def __init__(self, owner: "_FakeT3") -> None:
        self.owner = owner
        self.calls = []

    def __call__(self, *, inputs_embeds: torch.Tensor, **kwargs) -> _FakeOutput:
        batch, length, _hidden = inputs_embeds.shape
        output = torch.zeros(batch, length, 1)
        output[:, :, 0] = float(self.owner.next_token)
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = _FakeCache()
        cache.length += length
        self.calls.append(
            {
                "inputLength": length,
                "cacheLength": cache.length,
                "attentionMask": kwargs["attention_mask"].clone(),
                "positionIds": kwargs["position_ids"].clone(),
            },
        )
        return _FakeOutput(output, cache)


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


class _SequenceCache:
    def __init__(self, sequence: torch.Tensor | None = None) -> None:
        self.sequence = sequence

    def crop(self, length: int) -> None:
        self.sequence = self.sequence[:, :length]


class _SequenceOutput:
    def __init__(self, hidden: torch.Tensor, cache: _SequenceCache) -> None:
        self.hidden = hidden
        self.past_key_values = cache

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.hidden


class _SequenceTransformer:
    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        **_kwargs,
    ) -> _SequenceOutput:
        cache = past_key_values or _SequenceCache()
        if cache.sequence is None:
            cache.sequence = inputs_embeds.clone()
        else:
            cache.sequence = torch.cat(
                [cache.sequence, inputs_embeds],
                dim=1,
            )
        masked = cache.sequence[:, :, 0] * attention_mask
        cumulative = masked.cumsum(dim=1)
        hidden = cumulative[:, -inputs_embeds.shape[1] :, None]
        return _SequenceOutput(hidden, cache)


class _SequenceT3:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.hp = SimpleNamespace(start_speech_token=0, stop_speech_token=9)
        self.tfmr = _SequenceTransformer()

    def prepare_input_embeds(
        self,
        *,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        **_kwargs,
    ) -> tuple[torch.Tensor, int]:
        batch = text_tokens.shape[0]
        condition = torch.ones(batch, 1, 1)
        text = text_tokens.to(dtype=torch.float32).unsqueeze(-1)
        speech = self.speech_emb(speech_tokens)
        return torch.cat([condition, text, speech], dim=1), 1

    def speech_head(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden.shape[0], 10), -1000.0)
        token_ids = hidden[:, 0].to(dtype=torch.long).remainder(8) + 1
        logits.scatter_(1, token_ids[:, None], 1000.0)
        return logits

    def speech_emb(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.to(dtype=torch.float32).unsqueeze(-1) + 100.0


def _request(request_id: str, *, version: int = 1, input_done: bool = True):
    return T3ContinuousRequest(
        request_id=request_id,
        text_tokens=torch.tensor([1, 2]),
        text_attention_mask=torch.tensor([1, 1]),
        text_version=version,
        input_done=input_done,
    )


class T3ContinuousBatchDecoderTests(unittest.TestCase):
    def test_append_only_update_reuses_text_prefix_and_replays_speech(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        request = _request("live")

        decoder.step([request])
        request.text_tokens = torch.tensor([1, 2, 3])
        request.text_attention_mask = torch.ones(3, dtype=torch.long)
        request.text_version = 2
        decoder.step([request])

        metrics = decoder.metrics()
        self.assertEqual(metrics["fullRebuildCount"], 1)
        self.assertEqual(metrics["incrementalUpdateCount"], 1)
        self.assertEqual(metrics["incrementalFallbackCount"], 0)
        self.assertEqual(metrics["lastReusedTextPrefixLength"], 2)
        self.assertEqual(metrics["reusedTextPrefixTokens"], 2)
        self.assertEqual(metrics["replayedTextTokens"], 1)
        self.assertEqual(metrics["replayedSpeechTokens"], 1)
        self.assertEqual(t3.tfmr.calls[-2]["inputLength"], 3)
        self.assertEqual(t3.tfmr.calls[-2]["cacheLength"], 6)

    def test_changed_text_prefix_falls_back_to_full_rebuild(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        request = _request("live")

        decoder.step([request])
        request.text_tokens = torch.tensor([1, 7, 3])
        request.text_attention_mask = torch.ones(3, dtype=torch.long)
        request.text_version = 2
        decoder.step([request])

        metrics = decoder.metrics()
        self.assertEqual(metrics["fullRebuildCount"], 2)
        self.assertEqual(metrics["incrementalUpdateCount"], 0)
        self.assertEqual(metrics["incrementalFallbackCount"], 1)
        self.assertEqual(metrics["prefixChangeFallbackCount"], 1)
        self.assertEqual(t3.tfmr.calls[-2]["inputLength"], 6)

    def test_batch_reuses_only_the_common_physical_text_prefix(self) -> None:
        t3 = _FakeT3()
        decoder = T3ContinuousBatchDecoder(
            t3,
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        short = _request("short")
        long = _request("long")
        long.text_tokens = torch.tensor([4, 5, 6, 7])
        long.text_attention_mask = torch.ones(4, dtype=torch.long)

        decoder.step([short, long])
        short.text_tokens = torch.tensor([1, 2, 3])
        short.text_attention_mask = torch.ones(3, dtype=torch.long)
        short.text_version = 2
        decoder.step([short, long])

        metrics = decoder.metrics()
        self.assertEqual(metrics["incrementalUpdateCount"], 1)
        self.assertEqual(metrics["lastReusedTextPrefixLength"], 2)
        self.assertEqual(metrics["reusedTextPrefixTokens"], 4)
        self.assertEqual(metrics["replayedTextTokens"], 3)
        self.assertEqual(metrics["replayedSpeechTokens"], 2)
        self.assertEqual(t3.tfmr.calls[-2]["inputLength"], 4)

    def test_incremental_and_full_rebuild_generate_identical_tokens(self) -> None:
        incremental = T3ContinuousBatchDecoder(
            _SequenceT3(),
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        reference = T3ContinuousBatchDecoder(
            _SequenceT3(),
            object(),
            pad_token_id=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        incremental_request = _request("live")
        reference_request = _request("live")

        first_incremental = incremental.step([incremental_request])
        first_reference = reference.step([reference_request])
        self.assertEqual(
            first_incremental.tokens.tolist(),
            first_reference.tokens.tolist(),
        )

        for request in (incremental_request, reference_request):
            request.text_tokens = torch.tensor([1, 2, 3])
            request.text_attention_mask = torch.ones(3, dtype=torch.long)
            request.text_version = 2
        reference._invalidate_cache()
        reference._signature = None

        second_incremental = incremental.step([incremental_request])
        second_reference = reference.step([reference_request])
        third_incremental = incremental.step([incremental_request])
        third_reference = reference.step([reference_request])

        self.assertEqual(
            second_incremental.tokens.tolist(),
            second_reference.tokens.tolist(),
        )
        self.assertEqual(
            third_incremental.tokens.tolist(),
            third_reference.tokens.tolist(),
        )
        self.assertEqual(
            incremental_request.history.tolist(),
            reference_request.history.tolist(),
        )
        self.assertEqual(incremental.metrics()["incrementalUpdateCount"], 1)
        self.assertEqual(reference.metrics()["fullRebuildCount"], 2)

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
        self.assertEqual(decoder.metrics()["fullRebuildCount"], 2)
        self.assertEqual(decoder.metrics()["incrementalUpdateCount"], 0)

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
        self.assertEqual(decoder.metrics()["fullRebuildCount"], 1)
        self.assertEqual(decoder.metrics()["incrementalUpdateCount"], 1)


if __name__ == "__main__":
    unittest.main()
