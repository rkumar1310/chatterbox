"""Benchmark Chatterbox Turbo's live batch path on a Modal A10.

This intentionally supplies complete text before generation so first speech
token latency and RTF measure the model runtime rather than network or LLM text
delivery.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "chatterbox-turbo-live-pipeline-a10"
MODEL_CACHE = "/root/.cache/huggingface"
MINUTES = 60
TEXT = (
    "Hello! I am checking the live Chatterbox streaming pipeline. The assistant "
    "should start speaking quickly and continue smoothly while the rest of this "
    "sentence arrives."
)

repo_root = Path(__file__).resolve().parents[1]
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "git", "libsndfile1")
    .run_commands("python -m pip install uv")
    .add_local_file(
        str(repo_root / "pyproject.toml"),
        "/opt/chatterbox/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        str(repo_root / "README.md"),
        "/opt/chatterbox/README.md",
        copy=True,
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


def completed_source(text: str):
    from chatterbox.streaming import LiveTextStream

    source = LiveTextStream()
    source.append(text)
    source.finish()
    return source


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "max": round(ordered[-1], 4),
    }


@app.function(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: cache},
    timeout=45 * MINUTES,
    startup_timeout=20 * MINUTES,
)
def run_benchmark(
    concurrencies: list[int],
    chunk_tokens: int,
    repetitions: int,
    use_cuda_graph: bool,
    decoder_left_context_tokens: int,
) -> dict[str, Any]:
    import gc

    import torch

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    torch.set_float32_matmul_precision("high")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    cache.commit()

    torch.manual_seed(1917)
    list(
        model.stream_live_batch(
            [completed_source("Hello. The benchmark is warm.")],
            chunk_tokens=chunk_tokens,
            max_gen_len=300,
            use_cuda_graph=use_cuda_graph,
            decoder_left_context_tokens=decoder_left_context_tokens,
        ),
    )
    torch.cuda.synchronize()

    measurements = []
    for concurrency in concurrencies:
        for trial in range(1, repetitions + 1):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sources = [completed_source(TEXT) for _ in range(concurrency)]
            timings = [dict() for _ in range(concurrency)]
            chunks = [[] for _ in range(concurrency)]

            def on_timing(stage: str, request_index: int, seconds: float) -> None:
                timings[request_index].setdefault(stage, seconds)

            torch.manual_seed(10_000 + concurrency)
            started_at = time.perf_counter()
            for request_index, chunk in model.stream_live_batch(
                sources,
                chunk_tokens=chunk_tokens,
                max_gen_len=600,
                use_cuda_graph=use_cuda_graph,
                decoder_left_context_tokens=decoder_left_context_tokens,
                on_timing=on_timing,
            ):
                chunks[request_index].append(
                    {
                        "ready": time.perf_counter() - started_at,
                        "duration": chunk.duration_seconds,
                        "final": bool(chunk.is_final),
                    },
                )
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started_at

            requests = []
            for index in range(concurrency):
                stream = chunks[index]
                if not stream:
                    raise RuntimeError(f"request {index} returned no audio")
                audio_seconds = sum(item["duration"] for item in stream)
                completion_seconds = stream[-1]["ready"]
                playback_cursor = stream[0]["ready"]
                underruns = 0
                for item in stream:
                    if item["ready"] > playback_cursor + 0.005:
                        underruns += 1
                        playback_cursor = item["ready"]
                    playback_cursor += item["duration"]
                requests.append(
                    {
                        "requestIndex": index,
                        "firstSpeechTokenSeconds": round(
                            timings[index]["first_speech_token"],
                            4,
                        ),
                        "firstPcmSeconds": round(timings[index]["first_pcm"], 4),
                        "completionSeconds": round(completion_seconds, 4),
                        "audioSeconds": round(audio_seconds, 4),
                        "rtf": round(completion_seconds / audio_seconds, 4),
                        "playbackUnderruns": underruns,
                        "chunkCount": len(stream),
                        "receivedFinalChunk": bool(stream[-1]["final"]),
                    },
                )

            measurement = {
                "concurrency": concurrency,
                "trial": trial,
                "wallSeconds": round(wall_seconds, 4),
                "firstSpeechTokenSeconds": summarize(
                    [item["firstSpeechTokenSeconds"] for item in requests],
                ),
                "firstPcmSeconds": summarize(
                    [item["firstPcmSeconds"] for item in requests],
                ),
                "rtf": summarize([item["rtf"] for item in requests]),
                "playbackUnderruns": sum(
                    item["playbackUnderruns"] for item in requests
                ),
                "peakTorchAllocatedMiB": round(
                    torch.cuda.max_memory_allocated() / 1024**2,
                    1,
                ),
                "peakTorchReservedMiB": round(
                    torch.cuda.max_memory_reserved() / 1024**2,
                    1,
                ),
                "requests": requests,
            }
            measurements.append(measurement)
            print(json.dumps(measurement, sort_keys=True), flush=True)

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torchVersion": torch.__version__,
        "chunkTokens": chunk_tokens,
        "cudaGraphs": use_cuda_graph,
        "decoderLeftContextTokens": decoder_left_context_tokens,
        "text": TEXT,
        "measurements": measurements,
    }


@app.local_entrypoint()
def main(
    concurrencies: str = "1,2,4,8",
    chunk_tokens: int = 12,
    repetitions: int = 1,
    cuda_graphs: str = "true",
    decoder_left_context_tokens: int = 25,
    output: str = "chatterbox-live-pipeline-a10.json",
) -> None:
    values = [int(value) for value in concurrencies.split(",") if value.strip()]
    use_cuda_graph = cuda_graphs.strip().lower() in {"1", "true", "yes", "on"}
    result = run_benchmark.remote(
        values,
        chunk_tokens,
        repetitions,
        use_cuda_graph,
        decoder_left_context_tokens,
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
