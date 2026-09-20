import pytest
import torch

from chatterbox.models.s3gen.hifigan import SineGen, SourceModuleHnNSF


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
