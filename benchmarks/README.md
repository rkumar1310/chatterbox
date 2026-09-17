# Chatterbox Turbo batched-streaming benchmark

Run the A10 sweep from the repository root:

```shell
modal run benchmarks/modal_turbo_streaming_batch.py \
  --concurrencies 1,2,4,6,8,9,10 \
  --chunk-tokens 24 \
  --repetitions 3
```

The benchmark downloads `ResembleAI/chatterbox-turbo` into a persistent Modal
volume, warms all inference stages, then measures synchronized static
microbatches on an NVIDIA A10.

A run is marked acceptable only when every request:

- produces finite audio and a final chunk;
- has p95 real-time factor below 1.0;
- has p95 time to first audio below 1 second;
- plays from the first chunk with no simulated buffer underrun; and
- has no detected chunk-boundary amplitude outlier.

These are deliberately strict transport-level checks, not a replacement for
subjective listening tests. The included prompts are short English voice-agent
responses. Test production voices, prompt lengths, arrival patterns and longer
soak runs before setting a hard admission limit.

## A10 result (2026-09-17)

Using 24 speech tokens per audio chunk:

| Simultaneous streams | p95 first audio | p95 RTF | Underruns | Aggregate audio/wall | Peak allocated VRAM | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 6 | 0.650 s | 0.753 | 0 | 7.05x | 3.90 GiB | Pass |
| 8 | 0.760 s | 0.889 | 0 | 8.33x | 4.34 GiB | Pass |
| 9 | 0.782 s | 0.947 | 0 | 8.50x | 4.52 GiB | Pass |
| 10 | 0.884 s | 1.041 | 10 | 8.11x | 4.72 GiB | Fail |

The 9-stream case passed three additional consecutive trials. Across those
trials, p95 first audio was 0.754–0.782 seconds, p95 RTF was 0.881–0.947, and
there were zero playback underruns. Nine is therefore the measured ceiling for
this workload; use eight as the initial production admission limit to retain
headroom. Compute throughput, not A10 memory, is the limiting resource.
