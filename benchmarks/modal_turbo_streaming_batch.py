"""Benchmark Chatterbox Turbo batched streaming on a Modal A10.

Run from the repository root:

    modal run benchmarks/modal_turbo_streaming_batch.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "chatterbox-turbo-batched-streaming-a10"
MODEL_CACHE = "/root/.cache/huggingface"
MINUTES = 60

TEXTS = [
    "Thanks for calling. I found the account and I can help resolve that billing question now.",
    "Absolutely. Give me one moment to check the latest information, and then I will walk you through the next step.",
    "I understand why that was confusing. [chuckle] The good news is that the change has already been applied.",
    "Your appointment is confirmed for Thursday afternoon. I can also send a reminder before the scheduled time.",
    "Let me verify the delivery address first. Once that is confirmed, I can arrange the replacement immediately.",
    "That payment did go through successfully. The receipt should arrive in your inbox within the next few minutes.",
    "I have reviewed the request and everything looks correct. We can continue whenever you are ready.",
    "There are two available options. I will explain both briefly, and you can choose whichever one works better.",
]

repo_root = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "git", "libsndfile1")
    .run_commands("python -m pip install uv")
    .add_local_file(
        str(repo_root / "pyproject.toml"), "/opt/chatterbox/pyproject.toml", copy=True
    )
    .add_local_file(
        str(repo_root / "README.md"), "/opt/chatterbox/README.md", copy=True
    )
    .add_local_file(str(repo_root / "LICENSE"), "/opt/chatterbox/LICENSE", copy=True)
    .add_local_dir(str(repo_root / "src"), "/opt/chatterbox/src", copy=True)
    .run_commands(
        "cd /opt/chatterbox && uv pip install --system .",
        "python -m compileall -q /opt/chatterbox/src",
    )
    .env({"HF_HOME": MODEL_CACHE})
)

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    "chatterbox-turbo-batched-streaming-hf-cache",
    create_if_missing=True,
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


@app.function(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: cache},
    timeout=45 * MINUTES,
    startup_timeout=20 * MINUTES,
)
def run_sweep(
    concurrencies: list[int],
    chunk_tokens: int,
    repetitions: int,
) -> dict[str, Any]:
    import gc

    import numpy as np
    import torch

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    torch.set_float32_matmul_precision("high")
    loaded_at = time.monotonic()
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    torch.cuda.synchronize()
    model_load_seconds = time.monotonic() - loaded_at
    cache.commit()

    # Warm every stage, including the new batched S3Gen/vocoder path.
    torch.manual_seed(7)
    list(
        model.stream_batch(
            ["Hello. This is a short streaming warmup."],
            chunk_tokens=chunk_tokens,
            max_gen_len=300,
        ),
    )
    torch.cuda.synchronize()

    measurements = []
    for concurrency in concurrencies:
        for trial in range(repetitions):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            texts = [TEXTS[index % len(TEXTS)] for index in range(concurrency)]
            streams = [
                {
                    "ready": [],
                    "durations": [],
                    "chunks": [],
                    "final": False,
                }
                for _ in texts
            ]

            torch.manual_seed(1000 + concurrency)
            started_at = time.monotonic()
            for request_index, chunk in model.stream_batch(
                texts,
                chunk_tokens=chunk_tokens,
                max_gen_len=600,
            ):
                ready = time.monotonic() - started_at
                stream = streams[request_index]
                stream["ready"].append(ready)
                stream["durations"].append(chunk.duration_seconds)
                stream["chunks"].append(chunk.audio.numpy().reshape(-1))
                stream["final"] = stream["final"] or chunk.is_final
            torch.cuda.synchronize()
            wall_seconds = time.monotonic() - started_at

            request_results = []
            for index, stream in enumerate(streams):
                if not stream["chunks"]:
                    raise RuntimeError(f"request {index} produced no audio")
                waveform = np.concatenate(stream["chunks"])
                audio_seconds = len(waveform) / model.sr
                first_audio = stream["ready"][0]
                playback_cursor = first_audio
                underrun_seconds = 0.0
                underruns = 0
                for ready, duration in zip(stream["ready"], stream["durations"]):
                    if ready > playback_cursor + 0.005:
                        underruns += 1
                        underrun_seconds += ready - playback_cursor
                        playback_cursor = ready
                    playback_cursor += duration
                differences = np.abs(np.diff(waveform))
                boundary_jumps = []
                offset = 0
                for chunk in stream["chunks"][:-1]:
                    offset += len(chunk)
                    boundary_jumps.append(
                        float(abs(waveform[offset] - waveform[offset - 1]))
                    )
                p995_delta = (
                    float(np.quantile(differences, 0.995)) if differences.size else 0.0
                )
                request_results.append(
                    {
                        "requestIndex": index,
                        "firstAudioSeconds": round(first_audio, 4),
                        "completionSeconds": round(stream["ready"][-1], 4),
                        "audioSeconds": round(audio_seconds, 4),
                        "rtf": round(stream["ready"][-1] / audio_seconds, 4),
                        "chunkCount": len(stream["chunks"]),
                        "playbackUnderruns": underruns,
                        "underrunSeconds": round(underrun_seconds, 4),
                        "finite": bool(np.isfinite(waveform).all()),
                        "peak": round(float(np.max(np.abs(waveform))), 4),
                        "rms": round(float(np.sqrt(np.mean(np.square(waveform)))), 6),
                        "maxBoundaryJump": round(max(boundary_jumps, default=0.0), 6),
                        "boundaryOutliers": sum(
                            jump > max(0.25, p995_delta * 4.0)
                            for jump in boundary_jumps
                        ),
                        "receivedFinalChunk": stream["final"],
                    }
                )

            first_audio_values = [item["firstAudioSeconds"] for item in request_results]
            rtf_values = [item["rtf"] for item in request_results]
            total_audio = sum(item["audioSeconds"] for item in request_results)
            measurement = {
                "concurrency": concurrency,
                "trial": trial + 1,
                "wallSeconds": round(wall_seconds, 4),
                "totalAudioSeconds": round(total_audio, 4),
                "aggregateAudioPerWall": round(total_audio / wall_seconds, 4),
                "firstAudioMedianSeconds": round(
                    statistics.median(first_audio_values), 4
                ),
                "firstAudioP95Seconds": round(_percentile(first_audio_values, 0.95), 4),
                "firstAudioMaxSeconds": round(max(first_audio_values), 4),
                "rtfMedian": round(statistics.median(rtf_values), 4),
                "rtfP95": round(_percentile(rtf_values, 0.95), 4),
                "rtfMax": round(max(rtf_values), 4),
                "playbackUnderruns": sum(
                    item["playbackUnderruns"] for item in request_results
                ),
                "underrunSeconds": round(
                    sum(item["underrunSeconds"] for item in request_results), 4
                ),
                "boundaryOutliers": sum(
                    item["boundaryOutliers"] for item in request_results
                ),
                "allFinite": all(item["finite"] for item in request_results),
                "allFinal": all(item["receivedFinalChunk"] for item in request_results),
                "peakTorchAllocatedMiB": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 1
                ),
                "peakTorchReservedMiB": round(
                    torch.cuda.max_memory_reserved() / 1024**2, 1
                ),
                "requests": request_results,
            }
            measurement["acceptable"] = bool(
                measurement["allFinite"]
                and measurement["allFinal"]
                and measurement["rtfP95"] < 1.0
                and measurement["firstAudioP95Seconds"] < 1.0
                and measurement["playbackUnderruns"] == 0
                and measurement["boundaryOutliers"] == 0
            )
            measurements.append(measurement)
            print(json.dumps(measurement, sort_keys=True))

    return {
        "gpu": torch.cuda.get_device_name(0),
        "model": "ResembleAI/chatterbox-turbo",
        "modelLoadSeconds": round(model_load_seconds, 4),
        "chunkTokens": chunk_tokens,
        "measurements": measurements,
    }


@app.local_entrypoint()
def main(
    concurrencies: str = "1,2,4,6,8,9,10",
    chunk_tokens: int = 24,
    repetitions: int = 1,
    output: str = "chatterbox-turbo-batched-streaming-a10.json",
) -> None:
    values = [int(value) for value in concurrencies.split(",") if value.strip()]
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    result = run_sweep.remote(values, chunk_tokens, repetitions)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
