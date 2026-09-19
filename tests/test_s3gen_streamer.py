import unittest
from types import SimpleNamespace

import torch

from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.models.s3gen.streamer import (
    S3GenBatchingMetrics,
    S3GenBatchStreamer,
    S3GenStreamer,
)


class _FakeS3Gen:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.meanflow = True
        self.flow = SimpleNamespace(
            pre_lookahead_len=3,
            token_mel_ratio=2,
            input_frame_rate=25,
        )
        self.trim_fade = torch.empty(0)
        self.flow_calls = []
        self.hift_calls = []

    def __call__(
        self,
        *,
        speech_tokens,
        speech_token_lens,
        finalize,
        noised_mels,
        **_kwargs,
    ):
        self.flow_calls.append(
            {
                "speech_tokens": speech_tokens.clone(),
                "speech_token_lens": speech_token_lens.clone(),
                "finalize": finalize,
                "noised_mels": noised_mels.clone(),
            },
        )
        return torch.ones(
            speech_tokens.shape[0],
            80,
            noised_mels.shape[-1],
            dtype=self.dtype,
        )

    def hift_inference(
        self,
        speech_feat,
        cache_source,
        cache_source_lens=None,
    ):
        self.hift_calls.append(
            {
                "speech_feat": speech_feat.clone(),
                "cache_source": cache_source.clone(),
                "cache_source_lens": None
                if cache_source_lens is None
                else cache_source_lens.clone(),
            },
        )
        samples = speech_feat.shape[-1] * 480
        rows = torch.arange(
            1,
            speech_feat.shape[0] + 1,
            dtype=self.dtype,
        )[:, None]
        wavs = rows.expand(-1, samples).clone()
        return wavs, wavs[:, None, :].clone()


def _streamer(
    s3gen,
    ref_dict,
    *,
    left_context_tokens: int,
) -> S3GenStreamer:
    return S3GenStreamer(
        s3gen,
        ref_dict,
        crossfade_ms=0,
        left_context_tokens=left_context_tokens,
    )


class S3GenStreamerWindowTest(unittest.TestCase):
    def setUp(self):
        flow = SimpleNamespace(
            pre_lookahead_len=3,
            token_mel_ratio=2,
            input_frame_rate=25,
        )
        s3gen = SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            flow=flow,
            meanflow=True,
        )
        self.streamer = S3GenStreamer(
            s3gen,
            {},
            left_context_tokens=25,
        )

    def test_first_window_retains_lookahead_but_only_decodes_stable_noise(self):
        self.streamer.append(torch.arange(12))

        window = self.streamer._prepare_incremental_window(finalize=False)

        self.assertIsNotNone(window)
        self.assertEqual(window.stable_end_token, 9)
        self.assertEqual(window.speech_tokens.shape[-1], 12)
        self.assertEqual(window.noise.shape[-1], 18)
        self.assertEqual(window.context_samples, 0)

    def test_later_window_is_bounded_to_left_context_and_new_tokens(self):
        self.streamer.append(torch.arange(48))
        self.streamer.decoded_tokens = 33

        window = self.streamer._prepare_incremental_window(finalize=False)

        self.assertIsNotNone(window)
        self.assertEqual(window.stable_end_token, 45)
        self.assertEqual(window.speech_tokens.shape[-1], 40)
        self.assertEqual(window.noise.shape[-1], 74)
        self.assertEqual(window.context_samples, 24_000)


class S3GenBatchStreamerBucketTest(unittest.TestCase):
    def test_incremental_mixed_lengths_share_bucket_and_crop_padding(self):
        s3gen = _FakeS3Gen()
        ref_dict = {}
        streamers = [
            _streamer(s3gen, ref_dict, left_context_tokens=25),
            _streamer(s3gen, ref_dict, left_context_tokens=25),
        ]
        streamers[0].append(torch.arange(12))
        streamers[1].append(torch.arange(13))
        metrics = S3GenBatchingMetrics(bucket_width_tokens=8)
        batch = S3GenBatchStreamer(
            streamers,
            bucket_width_tokens=8,
            metrics=metrics,
        )

        first = batch.flush([0, 1])

        self.assertEqual(len(s3gen.flow_calls), 1)
        flow_call = s3gen.flow_calls[0]
        self.assertEqual(flow_call["speech_tokens"].shape, (2, 16))
        self.assertEqual(flow_call["speech_token_lens"].tolist(), [12, 13])
        self.assertEqual(flow_call["noised_mels"].shape, (2, 80, 26))
        self.assertTrue(
            torch.all(flow_call["speech_tokens"][0, 12:] == S3GEN_SIL),
        )
        self.assertTrue(
            torch.all(flow_call["speech_tokens"][1, 13:] == S3GEN_SIL),
        )
        hift_mels = s3gen.hift_calls[0]["speech_feat"]
        self.assertTrue(torch.all(hift_mels[0, :, 18:] == 0))
        self.assertTrue(torch.all(hift_mels[1, :, 20:] == 0))
        self.assertEqual([wav.shape[-1] for _, wav in first], [8_640, 9_600])
        self.assertEqual([streamer.decoded_tokens for streamer in streamers], [9, 10])

        final = batch.finish([0, 1])

        self.assertEqual(len(s3gen.flow_calls), 2)
        self.assertEqual(
            s3gen.flow_calls[1]["speech_token_lens"].tolist(),
            [15, 16],
        )
        self.assertEqual(
            [wav.shape[-1] for _, wav in final],
            [5_760, 5_760],
        )
        self.assertTrue(all(streamer.finished for streamer in streamers))
        self.assertEqual([streamer.decoded_tokens for streamer in streamers], [15, 16])
        self.assertEqual(
            metrics.as_dict(),
            {
                "bucketWidthTokens": 8,
                "decodeCalls": 2,
                "decodedRows": 4,
                "meanBatchSize": 2.0,
                "maxBatchSize": 2,
                "mixedLengthBatchCalls": 2,
                "incrementalCalls": 2,
                "fullPrefixCalls": 0,
                "tokenSlots": 64,
                "validTokenSlots": 56,
                "paddedTokenSlots": 8,
                "tokenPaddingPercent": 12.5,
                "melFrameSlots": 116,
                "validMelFrames": 100,
                "paddedMelFrames": 16,
                "melPaddingPercent": 13.7931,
                "bucketCalls": {"16": 2},
            },
        )

    def test_full_prefix_masks_variable_vocoder_cache_lengths(self):
        s3gen = _FakeS3Gen()
        ref_dict = {}
        streamers = [
            _streamer(s3gen, ref_dict, left_context_tokens=0),
            _streamer(s3gen, ref_dict, left_context_tokens=0),
        ]
        streamers[0].append(torch.arange(9))
        streamers[1].append(torch.arange(12))
        batch = S3GenBatchStreamer(streamers, bucket_width_tokens=8)

        first = batch.flush([0, 1])
        streamers[0].append(torch.arange(4))
        streamers[1].append(torch.arange(4))
        second = batch.flush([0, 1])

        self.assertEqual([wav.shape[-1] for _, wav in first], [5_760, 8_640])
        self.assertEqual([wav.shape[-1] for _, wav in second], [3_840, 3_840])
        second_hift = s3gen.hift_calls[1]
        self.assertEqual(
            second_hift["cache_source_lens"].tolist(),
            [5_760, 8_640],
        )
        self.assertEqual(second_hift["cache_source"].shape[-1], 8_640)
        self.assertTrue(
            torch.all(second_hift["cache_source"][0, :, 5_760:] == 0),
        )
        self.assertEqual(
            [streamer.hift_cache_source.shape[-1] for streamer in streamers],
            [9_600, 12_480],
        )


if __name__ == "__main__":
    unittest.main()
