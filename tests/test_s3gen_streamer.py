import unittest
from types import SimpleNamespace

import torch

from chatterbox.models.s3gen.streamer import S3GenStreamer


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


if __name__ == "__main__":
    unittest.main()
