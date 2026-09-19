from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Iterable, Iterator

import numpy as np
import torch


@dataclass
class StreamingAudioChunk:
    """A chunk of streamed mono audio.

    Legacy streaming paths expose float32 CPU ``audio`` with shape
    ``[1, samples]``. The continuous production path instead exposes an
    ``pcm_transfer`` whose GPU-side PCM16 conversion and pinned-memory copy are
    already in flight. Chatterbox streaming chunks are not Perth-watermarked;
    use ``generate`` when the final full waveform must include a watermark.
    """

    audio: torch.Tensor | None
    sample_rate: int
    index: int
    is_final: bool
    start_sample: int
    end_sample: int
    generated_tokens: int
    watermarked: bool = False
    pcm_transfer: AsyncPCMTransfer | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


class AsyncPCMTransfer:
    """One ownership-safe GPU-to-pinned-host PCM transfer.

    The pinned buffer belongs exclusively to this object until ``to_bytes`` or
    ``discard`` completes. Finalization returns it to the manager's size-bucket
    pool, so callers must not retain a NumPy or tensor view of the buffer.
    """

    def __init__(
        self,
        *,
        manager: AsyncPCMTransferManager,
        buffer: torch.Tensor,
        samples: torch.Tensor,
        sample_count: int,
        source: torch.Tensor,
        producer_ready_event: Any | None,
        ready_event: Any | None,
        copy_start_event: Any | None,
        copy_end_event: Any | None,
    ) -> None:
        self._manager = manager
        self._buffer = buffer
        self._samples = samples
        self._source = source
        self._producer_ready_event = producer_ready_event
        self._ready_event = ready_event
        self._copy_start_event = copy_start_event
        self._copy_end_event = copy_end_event
        self._lock = RLock()
        self._finalized = False
        self.sample_count = int(sample_count)
        self.byte_count = self.sample_count * 2

    def is_ready(self) -> bool:
        event = self._ready_event
        return event is None or bool(event.query())

    def to_bytes(self) -> bytes:
        return self._finalize(make_bytes=True)

    def discard(self) -> None:
        self._finalize(make_bytes=False)

    def _finalize(self, *, make_bytes: bool) -> bytes:
        with self._lock:
            if self._finalized:
                raise RuntimeError("PCM transfer is already finalized")
            self._finalized = True
            wait_started = perf_counter()
            copy_ms = 0.0
            wait_ms = 0.0
            delivered = False
            try:
                if self._ready_event is not None:
                    self._ready_event.synchronize()
                wait_ms = (perf_counter() - wait_started) * 1_000
                if (
                    self._copy_start_event is not None
                    and self._copy_end_event is not None
                ):
                    copy_ms = float(
                        self._copy_start_event.elapsed_time(self._copy_end_event),
                    )
                # ``numpy()`` is a zero-copy view of the pinned tensor. The
                # final bytes object is the only unavoidable host allocation.
                payload = self._samples.numpy().tobytes() if make_bytes else b""
                delivered = make_bytes
                return payload
            finally:
                self._manager._finalize_transfer(
                    self,
                    self._buffer,
                    copy_ms=copy_ms,
                    wait_ms=wait_ms,
                    delivered=delivered,
                )
                self._samples = torch.empty(0, dtype=torch.int16)
                self._source = torch.empty(0, dtype=torch.int16)
                self._buffer = torch.empty(0, dtype=torch.int16)
                self._ready_event = None
                self._producer_ready_event = None
                self._copy_start_event = None
                self._copy_end_event = None


class AsyncPCMTransferManager:
    """Convert waveform chunks to PCM16 and copy them without blocking CUDA."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._copy_stream: Any | None = None
        self._copy_device: torch.device | None = None
        self._pool: dict[int, list[torch.Tensor]] = {}
        self._active: dict[int, AsyncPCMTransfer] = {}
        self._buffer_allocations = 0
        self._buffer_reuses = 0
        self._allocated_capacity_samples = 0
        self._queued_transfers = 0
        self._completed_transfers = 0
        self._discarded_transfers = 0
        self._queued_bytes = 0
        self._delivered_bytes = 0
        self._max_pending_transfers = 0
        self._copy_gpu_time_ms = 0.0
        self._delivery_wait_time_ms = 0.0
        self._max_copy_gpu_time_ms = 0.0
        self._max_delivery_wait_time_ms = 0.0
        self._inference_steps_with_pending_delivery = 0
        self._inference_steps_with_copy_in_flight = 0

    def enqueue(self, audio: torch.Tensor) -> AsyncPCMTransfer:
        waveform = audio.detach().reshape(-1)
        if waveform.dtype != torch.float32:
            waveform = waveform.to(dtype=torch.float32)
        sample_count = int(waveform.numel())
        buffer = self._acquire_buffer(sample_count, pinned=waveform.is_cuda)
        samples = buffer[:sample_count]
        ready_event = None
        producer_ready_event = None
        copy_start_event = None
        copy_end_event = None

        if waveform.is_cuda:
            copy_stream = self._cuda_copy_stream(waveform.device)
            producer_stream = torch.cuda.current_stream(waveform.device)
            pcm = torch.round(
                torch.clamp(waveform, -1.0, 1.0) * 32_767.0,
            ).to(dtype=torch.int16)
            if not pcm.is_contiguous():
                pcm = pcm.contiguous()
            producer_ready_event = torch.cuda.Event()
            producer_ready_event.record(producer_stream)
            copy_start_event = torch.cuda.Event(enable_timing=True)
            copy_end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(producer_ready_event)
                copy_start_event.record(copy_stream)
                samples.copy_(pcm, non_blocking=True)
                copy_end_event.record(copy_stream)
            ready_event = copy_end_event
        else:
            pcm = torch.round(
                torch.clamp(waveform, -1.0, 1.0) * 32_767.0,
            ).to(device="cpu", dtype=torch.int16)
            samples.copy_(pcm)

        transfer = AsyncPCMTransfer(
            manager=self,
            buffer=buffer,
            samples=samples,
            sample_count=sample_count,
            source=pcm,
            producer_ready_event=producer_ready_event,
            ready_event=ready_event,
            copy_start_event=copy_start_event,
            copy_end_event=copy_end_event,
        )
        with self._lock:
            self._active[id(transfer)] = transfer
            self._queued_transfers += 1
            self._queued_bytes += transfer.byte_count
            self._max_pending_transfers = max(
                self._max_pending_transfers,
                len(self._active),
            )
        return transfer

    def mark_inference_step(self) -> None:
        with self._lock:
            transfers = list(self._active.values())
        if not transfers:
            return
        in_flight = sum(not transfer.is_ready() for transfer in transfers)
        with self._lock:
            self._inference_steps_with_pending_delivery += 1
            if in_flight:
                self._inference_steps_with_copy_in_flight += 1

    def metrics(self) -> dict[str, int | float]:
        with self._lock:
            transfers = list(self._active.values())
            completed = self._completed_transfers
            copy_time = self._copy_gpu_time_ms
            wait_time = self._delivery_wait_time_ms
            result = {
                "queuedTransfers": self._queued_transfers,
                "completedTransfers": completed,
                "discardedTransfers": self._discarded_transfers,
                "pendingTransfers": len(transfers),
                "maxPendingTransfers": self._max_pending_transfers,
                "queuedBytes": self._queued_bytes,
                "deliveredBytes": self._delivered_bytes,
                "bufferAllocations": self._buffer_allocations,
                "bufferReuses": self._buffer_reuses,
                "allocatedPinnedCapacityBytes": (self._allocated_capacity_samples * 2),
                "copyGpuTimeMs": round(copy_time, 4),
                "meanCopyGpuTimeMs": round(copy_time / completed, 4)
                if completed
                else 0.0,
                "maxCopyGpuTimeMs": round(self._max_copy_gpu_time_ms, 4),
                "deliveryWaitTimeMs": round(wait_time, 4),
                "meanDeliveryWaitTimeMs": round(wait_time / completed, 4)
                if completed
                else 0.0,
                "maxDeliveryWaitTimeMs": round(
                    self._max_delivery_wait_time_ms,
                    4,
                ),
                "inferenceStepsWithPendingDelivery": (
                    self._inference_steps_with_pending_delivery
                ),
                "inferenceStepsWithCopyInFlight": (
                    self._inference_steps_with_copy_in_flight
                ),
            }
        result["copiesInFlight"] = sum(
            not transfer.is_ready() for transfer in transfers
        )
        return result

    def _cuda_copy_stream(self, device: torch.device) -> Any:
        with self._lock:
            if self._copy_stream is None:
                self._copy_device = torch.device(device)
                self._copy_stream = torch.cuda.Stream(device=device)
            elif self._copy_device != torch.device(device):
                raise RuntimeError("PCM transfer manager cannot span CUDA devices")
            return self._copy_stream

    def _acquire_buffer(self, sample_count: int, *, pinned: bool) -> torch.Tensor:
        capacity = max(1, 1 << max(0, sample_count - 1).bit_length())
        key = capacity if pinned else -capacity
        with self._lock:
            available = self._pool.get(key)
            if available:
                self._buffer_reuses += 1
                return available.pop()
            self._buffer_allocations += 1
            self._allocated_capacity_samples += capacity
        return torch.empty(
            capacity,
            dtype=torch.int16,
            device="cpu",
            pin_memory=pinned,
        )

    def _finalize_transfer(
        self,
        transfer: AsyncPCMTransfer,
        buffer: torch.Tensor,
        *,
        copy_ms: float,
        wait_ms: float,
        delivered: bool,
    ) -> None:
        capacity = int(buffer.numel())
        key = capacity if buffer.is_pinned() else -capacity
        with self._lock:
            self._active.pop(id(transfer), None)
            self._completed_transfers += 1
            self._discarded_transfers += int(not delivered)
            if delivered:
                self._delivered_bytes += transfer.byte_count
            self._copy_gpu_time_ms += copy_ms
            self._delivery_wait_time_ms += wait_ms
            self._max_copy_gpu_time_ms = max(self._max_copy_gpu_time_ms, copy_ms)
            self._max_delivery_wait_time_ms = max(
                self._max_delivery_wait_time_ms,
                wait_ms,
            )
            self._pool.setdefault(key, []).append(buffer)


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
    return np.rint(audio_np * 32767.0).astype("<i2", copy=False).tobytes()


def chunks_to_pcm_s16le(chunks: Iterable[StreamingAudioChunk]) -> Iterator[bytes]:
    """Yield raw little-endian signed 16-bit PCM bytes for streamed chunks."""
    for chunk in chunks:
        if chunk.pcm_transfer is not None:
            yield chunk.pcm_transfer.to_bytes()
        elif chunk.audio is not None:
            yield audio_to_pcm_s16le(chunk.audio)
        else:
            raise RuntimeError("streaming chunk contains neither audio nor PCM")


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
        wav_file.writeframes(next(chunks_to_pcm_s16le([first])))

        for chunk in iterator:
            if chunk.sample_rate != first.sample_rate:
                raise ValueError(
                    f"stream sample rate changed from {first.sample_rate} to {chunk.sample_rate}"
                )
            wav_file.writeframes(next(chunks_to_pcm_s16le([chunk])))

    return path
