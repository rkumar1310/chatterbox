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

The CPU recording preserves the previously reviewed audio exactly; broader listening quality, long utterances, custom voice references and CUDA remain outside this validation.

Local WAVs and detailed reports are in `artifacts/live-streaming-cpu` and `artifacts/live-streaming-mps`, and the reusable package is in `artifacts/wheels`; generated artifacts and model weights are excluded from the source patch.
