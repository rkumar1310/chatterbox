from types import SimpleNamespace

import pytest
import torch

from chatterbox.models.s3gen.hifigan import HiFTGenerator, SineGen, SourceModuleHnNSF


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sine_generator_preserves_input_dtype(dtype):
    generator = SineGen(samp_rate=24_000, harmonic_num=4)
    f0 = torch.full((2, 1, 32), 220.0, dtype=dtype)

    sine_waves, voiced, noise = generator(f0)

    assert sine_waves.dtype == dtype
    assert voiced.dtype == dtype
    assert noise.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_source_module_accepts_matching_low_precision_weights(dtype):
    source = SourceModuleHnNSF(
        sampling_rate=24_000,
        upsample_scale=1,
        harmonic_num=4,
    ).to(dtype=dtype)
    f0 = torch.full((2, 32, 1), 220.0, dtype=dtype)

    merged, noise, voiced = source(f0)

    assert merged.dtype == dtype
    assert noise.dtype == dtype
    assert voiced.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_istft_uses_float32_fallback_and_restores_output_dtype(dtype):
    generator = SimpleNamespace(
        istft_params={"n_fft": 16, "hop_len": 4},
        stft_window=torch.hann_window(16),
    )
    magnitude = torch.ones((1, 9, 8), dtype=dtype)
    phase = torch.zeros_like(magnitude)

    waveform = HiFTGenerator._istft(generator, magnitude, phase)

    assert waveform.dtype == dtype
    assert torch.isfinite(waveform).all()
