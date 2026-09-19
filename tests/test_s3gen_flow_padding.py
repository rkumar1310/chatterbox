import unittest

import torch
from torch import nn

from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
from chatterbox.models.s3gen.utils.mask import make_pad_mask


class _PaddedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def output_size(self) -> int:
        return 4

    def forward(self, inputs, lengths):
        self.input = inputs.detach().clone()
        masks = ~make_pad_mask(lengths, inputs.shape[1]).unsqueeze(1)
        return inputs, masks


class _CapturingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask = None

    def forward(self, *, mu, mask, **_kwargs):
        self.mask = mask.detach().clone()
        return mu, None


class S3GenFlowPaddingTest(unittest.TestCase):
    def test_inference_masks_to_padded_tensor_width(self):
        encoder = _PaddedEncoder()
        decoder = _CapturingDecoder()
        flow = CausalMaskedDiffWithXvec(
            input_size=4,
            output_size=4,
            spk_embed_dim=3,
            vocab_size=32,
            token_mel_ratio=1,
            pre_lookahead_len=0,
            encoder=encoder,
            decoder=decoder,
        )
        speech_tokens = torch.tensor(
            [
                [3, 4, 5, 6, 7, 0, 0, 0],
                [8, 9, 10, 11, 12, 13, 14, 0],
            ],
        )

        output, _ = flow.inference(
            token=speech_tokens,
            token_len=torch.tensor([5, 7]),
            prompt_token=torch.tensor([[1, 2]]),
            prompt_token_len=torch.tensor([2]),
            prompt_feat=torch.zeros(1, 2, 4),
            prompt_feat_len=torch.tensor([2]),
            embedding=torch.tensor([[1.0, 0.0, 0.0]]),
            finalize=True,
            noised_mels=torch.zeros(2, 4, 8),
            n_timesteps=1,
        )

        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(encoder.input.shape, (2, 10, 4))
        self.assertTrue(torch.all(encoder.input[0, 7:] == 0))
        self.assertTrue(torch.all(encoder.input[1, 9:] == 0))
        self.assertEqual(decoder.mask.shape, (2, 1, 10))
        self.assertEqual(decoder.mask[0, 0].tolist(), [1] * 7 + [0] * 3)
        self.assertEqual(decoder.mask[1, 0].tolist(), [1] * 9 + [0])


if __name__ == "__main__":
    unittest.main()
