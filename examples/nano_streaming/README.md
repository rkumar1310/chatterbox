# Nano streaming listening examples

New recordings generated on the device shown below, with unchanged Nano weights, five-token chunks, 25-token history and no added vocoder delay.

| Case | Text | Duration | Streamed audio | Native audio |
| --- | --- | ---: | --- | --- |
| 01-simple-weather | The weather is pleasant today. | 1.80 s | [Streamed](audio/01-simple-weather-streamed.wav) | [Native](audio/01-simple-weather-native.wav) |
| 02-simple-door | Please close the door gently. | 1.72 s | [Streamed](audio/02-simple-door-streamed.wav) | [Native](audio/02-simple-door-native.wav) |
| 03-simple-meeting | Your meeting starts at nine. | 1.60 s | [Streamed](audio/03-simple-meeting-streamed.wav) | [Native](audio/03-simple-meeting-native.wav) |
| 04-simple-keys | I left the keys on the table. | 1.64 s | [Streamed](audio/04-simple-keys-streamed.wav) | [Native](audio/04-simple-keys-native.wav) |
| 05-simple-tea | Would you like a cup of tea? | 1.48 s | [Streamed](audio/05-simple-tea-streamed.wav) | [Native](audio/05-simple-tea-native.wav) |
| 06-tiny-yes | Yes. | 0.68 s | [Streamed](audio/06-tiny-yes-streamed.wav) | [Native](audio/06-tiny-yes-native.wav) |
| 07-tiny-no | No. | 0.72 s | [Streamed](audio/07-tiny-no-streamed.wav) | [Native](audio/07-tiny-no-native.wav) |

Open `listen.html` locally for paired audio players; recordings and `results.json` are versioned so listening feedback can name an exact case and timestamp.

All recorded cases passed native speech-token parity, EOS completion, history bounds, first audio before completion, finite samples, sample continuity, chunk-to-WAV PCM equality, native duration equality and file hash checks.

Listening review remains pending; these checks do not measure phonetic accuracy, intelligibility or freedom from acoustic artifacts.

Timings follow a per-case warm-up and exclude loading, preprocessing and speaker preparation; first audio means the first nonempty waveform chunk returned to the consumer.

See `results.json` for device, FP32 dtype, seed, input length, chunk count, timing and sampled Metal driver memory; CPU uses two threads and batch size one.

Run `python scripts/validate_nano_streaming.py --device mps --output artifacts/new-validation --export-listening examples/nano_streaming` to regenerate all nine cases sequentially.

Use `--only 01-simple-weather` for one case or `--resume` to reuse completed recordings whose inputs, settings, hashes and source still match.
