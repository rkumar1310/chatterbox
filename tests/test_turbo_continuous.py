from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

import chatterbox.turbo_continuous as continuous


@dataclass(frozen=True)
class _Snapshot:
    text: str
    version: int = 1
    input_done: bool = True
    cancelled: bool = False


class _Source:
    def __init__(self, text: str) -> None:
        self.value = _Snapshot(text)

    def snapshot(self):
        return self.value

    def cancel(self) -> None:
        self.value = _Snapshot(
            self.value.text,
            version=self.value.version + 1,
            input_done=self.value.input_done,
            cancelled=True,
        )


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **_kwargs):
        length = max(1, len(text.split()))
        return SimpleNamespace(
            input_ids=torch.ones(1, length, dtype=torch.long),
            attention_mask=torch.ones(1, length, dtype=torch.long),
        )


class _T3:
    device = torch.device("cpu")


class _Decoder:
    def __init__(self, *_args, **_kwargs) -> None:
        self.counts = {}
        self.max_batch = 0

    def step(self, requests):
        if not requests:
            return None
        self.max_batch = max(self.max_batch, len(requests))
        tokens = []
        valid = []
        finished = []
        for request in requests:
            count = self.counts.get(request.request_id, 0) + 1
            self.counts[request.request_id] = count
            limit = 4 if request.request_id == "long" else 5
            is_finished = count >= limit
            tokens.append(9 if is_finished else 2)
            valid.append(not is_finished)
            finished.append(is_finished)
        return SimpleNamespace(
            request_ids=tuple(request.request_id for request in requests),
            tokens=torch.tensor(tokens),
            valid=torch.tensor(valid),
            waiting=torch.zeros(len(requests), dtype=torch.bool),
            finished=torch.tensor(finished),
        )

    def metrics(self):
        return {"maxBatchSize": self.max_batch}


class _Streamer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.generated_tokens = 0

    def append(self, token) -> None:
        self.generated_tokens += token.shape[-1]


class _BatchStreamer:
    def __init__(self, streamers, **_kwargs) -> None:
        self.streamers = list(streamers)

    def flush(self, indices, **_kwargs):
        return [(index, torch.ones(1, 8)) for index in indices]

    def finish(self, indices):
        return [(index, torch.ones(1, 4)) for index in indices]


class TurboContinuousEngineTests(unittest.TestCase):
    def test_host_state_is_transferred_only_at_pcm_boundaries(self) -> None:
        with patch.multiple(
            continuous,
            T3ContinuousBatchDecoder=_Decoder,
            S3GenStreamer=_Streamer,
            S3GenBatchStreamer=_BatchStreamer,
        ):
            engine = continuous.ChatterboxTurboContinuousEngine(
                t3=_T3(),
                s3gen=SimpleNamespace(),
                tokenizer=_Tokenizer(),
                t3_conditionals=object(),
                s3gen_conditionals={},
                sample_rate=24_000,
                normalize_text=lambda text, **_kwargs: text,
                chunk_tokens=2,
            )
            engine.add_request("long", _Source("a long request"))

            first = engine.step()
            self.assertEqual(first.active_request_ids, ("long",))
            self.assertEqual(first.audio, ())
            self.assertEqual(engine.metrics()["hostTransferCount"], 0)

            engine.add_request("late", _Source("a later request"))
            joined = engine.step()
            self.assertEqual(joined.active_request_ids, ("long", "late"))
            self.assertEqual(engine.metrics()["hostTransferCount"], 1)
            self.assertEqual(
                [output.request_id for output in joined.audio],
                ["long"],
            )

            engine.step()
            fourth = engine.step()
            self.assertIn("long", fourth.finished_request_ids)
            self.assertIn("late", [output.request_id for output in fourth.audio])

            engine.step()
            sixth = engine.step()
            self.assertIn("late", sixth.finished_request_ids)
            self.assertEqual(engine.request_ids, ())
            metrics = engine.metrics()
            self.assertEqual(metrics["speechTokenSteps"], 6)
            self.assertEqual(metrics["hostTransferCount"], 3)
            self.assertEqual(metrics["perSpeechTokenHostTransfers"], 0)
            self.assertEqual(metrics["t3"]["maxBatchSize"], 2)

    def test_cancelled_request_leaves_without_a_host_token_transfer(self) -> None:
        with patch.multiple(
            continuous,
            T3ContinuousBatchDecoder=_Decoder,
            S3GenStreamer=_Streamer,
            S3GenBatchStreamer=_BatchStreamer,
        ):
            source = _Source("cancel this request")
            engine = continuous.ChatterboxTurboContinuousEngine(
                t3=_T3(),
                s3gen=SimpleNamespace(),
                tokenizer=_Tokenizer(),
                t3_conditionals=object(),
                s3gen_conditionals={},
                sample_rate=24_000,
                normalize_text=lambda text, **_kwargs: text,
                chunk_tokens=2,
            )
            engine.add_request("cancel", source)
            engine.step()
            source.cancel()

            result = engine.step()

            self.assertEqual(result.cancelled_request_ids, ("cancel",))
            self.assertEqual(result.audio, ())
            self.assertEqual(engine.request_ids, ())
            self.assertEqual(engine.metrics()["hostTransferCount"], 0)


if __name__ == "__main__":
    unittest.main()
