# Clone streaming validation

Validation ran in this independent Chatterbox clone using its own Python 3.13 environment, two CPU threads, batch size one and unchanged Nano weights.

All 28 tests passed with the editable install and again with the built wheel; the suite covers token sampling, live chunks, decoder history, masks, random states, stitching and source verification.

| Run | Audio chunks | Audio duration | First audio after warm-up | Total streaming time |
| --- | ---: | ---: | ---: | ---: |
| CPU | 23 | 4.52 s | 1.393 s | 36.384 s |
| MPS | 3 | 0.68 s | 0.668 s | 3.520 s |

The CPU run used the full 110-code sample sentence, produced 23 chunks and 108480 samples, retained at most 25 generated history tokens, and reproduced the previous final streamed WAV byte for byte.

The MPS run used `Hello.` as a short integration check from the installed wheel, with T3, S3Gen, the vocoder and voice encoder on Metal and no operator CPU fallback enabled.

Both runs reached EOS, matched native T3 speech tokens, yielded audio before generation finished, kept all model parameters frozen, and passed PCM concatenation and WAV hash checks.

These timings exclude model loading, text preprocessing and reference preparation, follow a warm-up with token generation and one partial decode, and use FP32; neither run demonstrates real-time speed.

The CPU recording preserves the previously reviewed audio exactly; broader listening quality, custom voice references and CUDA remain outside this validation.

Local WAVs and detailed reports are in `artifacts/live-streaming-cpu` and `artifacts/live-streaming-mps`, and the reusable package is in `artifacts/wheels`; generated artifacts and model weights are excluded from the source patch.

## Expanded recordings on 29 September 2026

Eight new cases ran sequentially on MPS: five simple sentences with different seeds, two one-word responses and a directions paragraph; the longer story ran fully on CPU, with one simple sentence also checked on CPU.

The unchanged model uses FP32, batch size one, two CPU threads, temperature 0.8, top-k 1000, top-p 0.95, repetition penalty 1.2 and two decoder steps; chunks contain five tokens, generated history is capped at 25 tokens, overlap is 40 ms and added vocoder lookahead is zero.

| Case | Device | Audio | Chunks | First audio | Streaming time |
| --- | --- | ---: | ---: | ---: | ---: |
| 01-simple-weather | MPS | 1.80 s | 9 | 0.645 s | 14.718 s |
| 02-simple-door | MPS | 1.72 s | 9 | 0.584 s | 11.662 s |
| 03-simple-meeting | MPS | 1.60 s | 8 | 0.641 s | 15.393 s |
| 04-simple-keys | MPS | 1.64 s | 8 | 0.535 s | 11.209 s |
| 05-simple-tea | MPS | 1.48 s | 7 | 2.245 s | 16.438 s |
| 06-tiny-yes | MPS | 0.68 s | 3 | 0.570 s | 4.140 s |
| 07-tiny-no | MPS | 0.72 s | 4 | 0.548 s | 4.116 s |
| 08-medium-directions | MPS | 12.32 s | 62 | 0.568 s | 215.000 s |
| 09-long-story | CPU | 32.16 s | 161 | 1.172 s | 555.124 s |
| 01-simple-weather | CPU | 1.80 s | 9 | 1.151 s | 10.163 s |

All ten recordings reached EOS and passed native speech-token parity, fixed history bounds, contiguous chunks, chunk-to-WAV PCM equality, native audio duration equality, finite and non-silent output, and SHA-256 checks; all emitted audio before EOS.

All 28 unit tests passed again after the live runs, covering initial EOS, generation caps, final tails, invalid input, frozen parameters, decoder masks, conditioning alignment, raw input vectors and random-state preservation.

The largest sampled MPS driver allocation during streaming was 2.690 GiB; this is a sampled driver-memory value, not a process or whole-system peak.

First audio is the first nonempty returned waveform chunk, measured after a per-case warm-up; loading, tokenization and reference preparation are excluded, and the per-case generation caps and seeds are recorded in `examples/nano_streaming/cases.json`.

Generation remains slower than real time on this 8 GB Mac and worsens for the longer recording, consistent with the remaining accumulated-frame vocoder work; these are observed timings under local machine load, not a controlled hardware benchmark.

The matched recordings are committed under `examples/nano_streaming/audio`, with a local listening page and readable JSON results; full chunks, logs and provenance reports remain local under `artifacts/new-validation-mps` and `artifacts/new-validation-cpu`.

The long-story MPS run was deliberately stopped after emitting 15.44 seconds over 559.4 seconds of generation as performance deteriorated; its partial WAV and log are preserved locally, and it has not passed EOS or native-baseline checks.

Human listening review is pending for the new recordings, including pipe sounds, metallic tone, clicks, missing or repeated words, drift and sentence endings; automated checks do not establish artifact-free quality or phonetic correctness.
