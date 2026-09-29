# Nano streaming listening examples

New recordings generated on the device shown below, with unchanged Nano weights, five-token chunks, 25-token history and no added vocoder delay.

| Case | Device | Text | Duration | Streamed audio | Native audio |
| --- | --- | --- | ---: | --- | --- |
| 01-simple-weather | MPS | The weather is pleasant today. | 1.80 s | [Streamed](audio/01-simple-weather-streamed.wav) | [Native](audio/01-simple-weather-native.wav) |
| 02-simple-door | MPS | Please close the door gently. | 1.72 s | [Streamed](audio/02-simple-door-streamed.wav) | [Native](audio/02-simple-door-native.wav) |
| 03-simple-meeting | MPS | Your meeting starts at nine. | 1.60 s | [Streamed](audio/03-simple-meeting-streamed.wav) | [Native](audio/03-simple-meeting-native.wav) |
| 04-simple-keys | MPS | I left the keys on the table. | 1.64 s | [Streamed](audio/04-simple-keys-streamed.wav) | [Native](audio/04-simple-keys-native.wav) |
| 05-simple-tea | MPS | Would you like a cup of tea? | 1.48 s | [Streamed](audio/05-simple-tea-streamed.wav) | [Native](audio/05-simple-tea-native.wav) |
| 06-tiny-yes | MPS | Yes. | 0.68 s | [Streamed](audio/06-tiny-yes-streamed.wav) | [Native](audio/06-tiny-yes-native.wav) |
| 07-tiny-no | MPS | No. | 0.72 s | [Streamed](audio/07-tiny-no-streamed.wav) | [Native](audio/07-tiny-no-native.wav) |
| 08-medium-directions | MPS | When you leave the station, turn left and walk past the small bakery. Cross the road at the traffic lights, then follow the path beside the river. The library is the red brick building at the end of the path, just before the wooden bridge. | 12.32 s | [Streamed](audio/08-medium-directions-streamed.wav) | [Native](audio/08-medium-directions-native.wav) |
| 09-long-story | CPU | On Saturday morning, Maya packed a small bag and set out for the coast. The train moved slowly through green fields, quiet villages, and a forest full of tall trees. When she arrived, the sky was clear and a gentle wind carried the sound of waves across the platform. She walked to the harbour, bought a warm loaf of bread, and sat beside an old fishing boat. A friendly dog wandered over and rested at her feet. After lunch, she followed a narrow path up the hill and stopped to look at the sea. Far below, children were building a castle in the sand while their parents watched from the shore. Maya took a photograph, put away her phone, and stayed until the evening light turned the water gold. It had been a simple day, but she knew she would remember it for a long time. | 32.16 s | [Streamed](audio/09-long-story-streamed.wav) | [Native](audio/09-long-story-native.wav) |

Open `listen.html` locally for paired audio players; recordings and `results.json` are versioned so listening feedback can name an exact case and timestamp.

All recorded cases passed native speech-token parity, EOS completion, history bounds, finite samples, sample continuity, chunk-to-WAV PCM equality, native duration equality and file hash checks; audio precedes EOS when the utterance reaches the five-token chunk threshold.

Listening review remains pending; these checks do not measure phonetic accuracy, intelligibility or freedom from acoustic artifacts.

Timings follow a per-case warm-up and exclude loading, preprocessing and speaker preparation; first audio means the first nonempty waveform chunk returned to the consumer.

See `results.json` for device, FP32 dtype, seed, input length, chunk count, timing and sampled Metal driver memory; CPU uses two threads and batch size one.

Run `python scripts/validate_nano_streaming.py --device cpu --output artifacts/new-validation --export-listening examples/nano_streaming` to regenerate all nine cases sequentially on CPU; use `--device mps` to test Metal explicitly.

Use `--only 01-simple-weather` for one case or `--resume` to reuse completed recordings whose inputs, settings, hashes and source still match.

The weather sentence also passed a fresh CPU run: [streamed audio](audio/10-cpu-weather-streamed.wav) and [native audio](audio/10-cpu-weather-native.wav), with settings and timings in `cpu-results.json`.
