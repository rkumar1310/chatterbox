# Nano audio streaming

This independent clone extends Chatterbox at revision `5de7a54aa4e5e2baadb0182dde554908b48b85c2` with the `chatterbox.nano_streaming` package; no Reflex repository is needed to install or use it.

## Install from this clone

The recorded dependency lock reproduces the tested Python 3.13 environment on this Mac; other platforms should resolve the upstream dependencies and validate separately.

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r streaming-requirements.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

Other repositories can install this clone with `pip install /absolute/path/to/chatterbox-nano-streaming`, or install a wheel built here, then import the same public API.

```python
from chatterbox.nano_streaming import StreamConfig, load_nano

streamer, provenance = load_nano(device="cpu", config=StreamConfig())
for chunk in streamer.stream("Yes, I can hear you clearly now."):
    # Deliver each chunk immediately to an audio sink.
    audio_samples = chunk.samples.numpy()
    sample_rate = chunk.sample_rate
```

Use `device="mps"` on supported Macs or `device="cuda"` on CUDA machines, and pass `audio_prompt_path="voice.wav"` to `stream()` for a speaker reference; complete text or 768-wide Nano input vectors are required before generation starts.

## Generate recordings and run tests

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/nano_streaming -q
.venv/bin/python -m chatterbox.nano_streaming --text 'Yes, I can hear you clearly now.' --max-gen-len 256 --output artifacts/my-stream
```

Add `--device mps` for Metal or `--assets /path/to/cached/nano/assets` to reuse local weights; the first default load downloads Nano model revision `71ccd1d0081b430592cea481f4307e764e07bc64`.

Use an empty output directory; the CLI writes individual chunks immediately, joins them into `streamed.wav`, generates `native.wav`, and records timing, token parity, source and asset hashes in `report.json`.

## Fixes and boundaries

Cached native T3 sampling emits audio every five generated speech tokens; S3Gen receives the speaker reference together with up to 25 prior generated tokens and their matching 50 acoustic frames as conditioning, then predicts only the new frames.

The wrapper trims the native partial-flow attention mask to the cropped frame count, preserves old acoustic frames, aligns Gaussian noise to global token positions, separates acoustic and vocoder random states from speech-token sampling, and flushes final audio with contiguous sample positions.

The native three-token lookahead and 40 ms overlap remain, with no additional 500 ms buffer; first audio still incurs model computation time.

HiFT processes accumulated acoustic frames, so total decoder cost grows with utterance length; this implementation does not guarantee real-time generation or constant memory usage.

The loader verifies the native source files against the packaged `upstream-source.json` manifest for editable and wheel installs, without requiring a Git checkout at runtime; modified upstream versions must be revalidated before updating the manifest.

The original voice reference and model weights stay unchanged, and CUDA support has not been exercised on this Mac.

## Longer validation and listening examples

The GitHub branch is `rkumar1310/chatterbox:nano-streaming`; clone this branch and install it directly using the commands above.

The five new simple sentences, two one-word responses, medium paragraph and long story are defined in `examples/nano_streaming/cases.json`; run them sequentially with:

```sh
.venv/bin/python scripts/validate_nano_streaming.py --device mps --output artifacts/new-validation --export-listening examples/nano_streaming
```

Each case has a fixed seed and generation cap, and produces streamed chunks, joined audio, a matched native recording and an automatic verification report; model processes run one at a time.

The committed `examples/nano_streaming/audio` directory contains the paired final WAVs, and `examples/nano_streaming/listen.html` provides audio players for local listening; recordings still require human review for artifacts and intelligibility.
