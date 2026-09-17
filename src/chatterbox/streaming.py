from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Iterator

import numpy as np
import torch


@dataclass
class StreamingAudioChunk:
    """A chunk of streamed mono audio.

    Audio is always float32 on CPU with shape ``[1, samples]``. Chatterbox
    streaming chunks are not Perth-watermarked; use ``generate`` when the final
    full waveform must include a watermark.
    """

    audio: torch.Tensor
    sample_rate: int
    index: int
    is_final: bool
    start_sample: int
    end_sample: int
    generated_tokens: int
    watermarked: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


@dataclass(frozen=True)
class LiveTextSnapshot:
    """One atomic view of text supplied to a live TTS request."""

    text: str
    version: int
    input_done: bool
    cancelled: bool


class LiveTextStream:
    """Thread-safe append-only text source for live Turbo generation."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._text = ""
        self._version = 0
        self._input_done = False
        self._cancelled = False

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self._input_done:
                raise RuntimeError("text input is already complete")
            if self._cancelled:
                raise RuntimeError("text input is cancelled")
            self._text += text
            self._version += 1

    def finish(self) -> None:
        with self._lock:
            if self._input_done:
                raise RuntimeError("text input is already complete")
            if self._cancelled:
                raise RuntimeError("text input is cancelled")
            self._input_done = True
            self._version += 1

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._version += 1

    def snapshot(self) -> LiveTextSnapshot:
        with self._lock:
            return LiveTextSnapshot(
                text=self._text,
                version=self._version,
                input_done=self._input_done,
                cancelled=self._cancelled,
            )


def audio_to_pcm_s16le(audio: torch.Tensor) -> bytes:
    """Convert a mono float audio tensor to raw little-endian signed 16-bit PCM."""
    audio_np = audio.detach().cpu().reshape(-1).numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767.0).astype("<i2", copy=False).tobytes()


def chunks_to_pcm_s16le(chunks: Iterable[StreamingAudioChunk]) -> Iterator[bytes]:
    """Yield raw little-endian signed 16-bit PCM bytes for streamed chunks."""
    for chunk in chunks:
        yield audio_to_pcm_s16le(chunk.audio)


def write_chunks_to_wav(
    path: str | Path, chunks: Iterable[StreamingAudioChunk]
) -> Path:
    """Write streamed chunks to a mono 16-bit PCM WAV file."""
    path = Path(path)
    iterator = iter(chunks)

    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot write WAV from an empty chunk stream") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(first.sample_rate)
        wav_file.writeframes(audio_to_pcm_s16le(first.audio))

        for chunk in iterator:
            if chunk.sample_rate != first.sample_rate:
                raise ValueError(
                    f"stream sample rate changed from {first.sample_rate} to {chunk.sample_rate}"
                )
            wav_file.writeframes(audio_to_pcm_s16le(chunk.audio))

    return path
