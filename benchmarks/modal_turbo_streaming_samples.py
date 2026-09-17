"""Generate downloadable single-stream and batched Chatterbox Turbo samples."""

from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path
from typing import Any

import modal

APP_NAME = "chatterbox-turbo-batched-streaming-a10-samples"
MODEL_CACHE = "/root/.cache/huggingface"
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
        str(repo_root / "pyproject.toml"),
        "/opt/chatterbox/pyproject.toml",
        copy=True,
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


def _wav_bytes(audio: Any, sample_rate: int) -> bytes:
    import numpy as np

    waveform = np.asarray(audio).reshape(-1)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


@app.function(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: cache},
    timeout=20 * 60,
    startup_timeout=20 * 60,
)
def generate_samples() -> dict[str, Any]:
    import numpy as np
    import torch

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    torch.manual_seed(20260917)
    started_at = time.monotonic()
    single_chunks = []
    single_first_audio = None
    for chunk in model.stream(TEXTS[0], chunk_tokens=24, max_gen_len=600):
        if single_first_audio is None:
            single_first_audio = time.monotonic() - started_at
        single_chunks.append(chunk)
    single_elapsed = time.monotonic() - started_at
    single_audio = np.concatenate(
        [chunk.audio.numpy().reshape(-1) for chunk in single_chunks]
    )

    torch.manual_seed(20260917)
    batch_audio: list[list[Any]] = [[] for _ in range(9)]
    batch_first_audio: list[float | None] = [None] * 9
    batch_started_at = time.monotonic()
    texts = [TEXTS[index % len(TEXTS)] for index in range(9)]
    for request_index, chunk in model.stream_batch(
        texts,
        chunk_tokens=24,
        max_gen_len=600,
    ):
        if batch_first_audio[request_index] is None:
            batch_first_audio[request_index] = time.monotonic() - batch_started_at
        batch_audio[request_index].append(chunk.audio.numpy().reshape(-1))
    batch_elapsed = time.monotonic() - batch_started_at

    selected = [0, 2, 7]
    files = {
        "single-stream.wav": _wav_bytes(single_audio, model.sr),
    }
    manifest = {
        "model": "ResembleAI/chatterbox-turbo",
        "gpu": torch.cuda.get_device_name(0),
        "sampleRate": model.sr,
        "chunkTokens": 24,
        "single": {
            "filename": "single-stream.wav",
            "text": TEXTS[0],
            "firstAudioSeconds": round(single_first_audio, 4),
            "wallSeconds": round(single_elapsed, 4),
            "audioSeconds": round(len(single_audio) / model.sr, 4),
        },
        "batch": {
            "concurrency": 9,
            "wallSeconds": round(batch_elapsed, 4),
            "samples": [],
        },
        "note": "Streaming output is not watermarked; generate() is required for Perth watermarking.",
    }
    for request_index in selected:
        audio = np.concatenate(batch_audio[request_index])
        filename = f"batch-9-stream-{request_index + 1}.wav"
        files[filename] = _wav_bytes(audio, model.sr)
        manifest["batch"]["samples"].append(
            {
                "filename": filename,
                "requestIndex": request_index,
                "text": texts[request_index],
                "firstAudioSeconds": round(batch_first_audio[request_index], 4),
                "audioSeconds": round(len(audio) / model.sr, 4),
            }
        )

    return {"files": files, "manifest": manifest}


@app.local_entrypoint()
def main(output_dir: str = "chatterbox-a10-audio-samples") -> None:
    result = generate_samples.remote()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, contents in result["files"].items():
        (destination / filename).write_bytes(contents)
    (destination / "manifest.json").write_text(
        json.dumps(result["manifest"], indent=2) + "\n"
    )
    print(json.dumps(result["manifest"], indent=2))
