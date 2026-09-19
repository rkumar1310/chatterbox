import tempfile
import unittest
import wave
from pathlib import Path

import torch

from chatterbox.streaming import (
    AsyncPCMTransferManager,
    LiveTextStream,
    StreamingAudioChunk,
    audio_to_pcm_s16le,
    chunks_to_pcm_s16le,
    write_chunks_to_wav,
)


class StreamingUtilsTest(unittest.TestCase):
    def test_live_text_stream_is_append_only_and_versioned(self):
        source = LiveTextStream()
        source.append("Hello")
        source.append(" world")

        snapshot = source.snapshot()
        self.assertEqual(snapshot.text, "Hello world")
        self.assertEqual(snapshot.version, 2)
        self.assertFalse(snapshot.input_done)
        self.assertFalse(snapshot.cancelled)

        source.finish()
        self.assertTrue(source.snapshot().input_done)
        with self.assertRaisesRegex(RuntimeError, "already complete"):
            source.append("!")

    def test_live_text_stream_can_cancel_generation(self):
        source = LiveTextStream()
        source.append("Stop this request")
        source.cancel()

        snapshot = source.snapshot()
        self.assertTrue(snapshot.cancelled)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            source.finish()

    def test_audio_to_pcm_s16le_clamps_and_converts(self):
        audio = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], dtype=torch.float32)

        pcm = audio_to_pcm_s16le(audio)

        self.assertEqual(len(pcm), 10)
        self.assertEqual(pcm.hex(), "0000ff7f0180ff7f0180")

    def test_async_pcm_transfer_matches_reference_and_reuses_buffer(self):
        manager = AsyncPCMTransferManager()
        audio = torch.tensor(
            [[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]],
            dtype=torch.float32,
        )

        first = manager.enqueue(audio)
        self.assertEqual(first.to_bytes(), audio_to_pcm_s16le(audio))
        second = manager.enqueue(audio)
        self.assertEqual(second.to_bytes(), audio_to_pcm_s16le(audio))

        metrics = manager.metrics()
        self.assertEqual(metrics["queuedTransfers"], 2)
        self.assertEqual(metrics["completedTransfers"], 2)
        self.assertEqual(metrics["bufferAllocations"], 1)
        self.assertEqual(metrics["bufferReuses"], 1)
        self.assertEqual(metrics["pendingTransfers"], 0)

    def test_discarded_pcm_transfer_releases_its_buffer(self):
        manager = AsyncPCMTransferManager()
        transfer = manager.enqueue(torch.ones(1, 4))

        transfer.discard()

        metrics = manager.metrics()
        self.assertEqual(metrics["discardedTransfers"], 1)
        self.assertEqual(metrics["deliveredBytes"], 0)
        self.assertEqual(metrics["pendingTransfers"], 0)

    def test_pending_transfers_own_distinct_buffers_until_finalized(self):
        manager = AsyncPCMTransferManager()
        silent = torch.zeros(1, 8)
        loud = torch.ones(1, 8)

        first = manager.enqueue(silent)
        second = manager.enqueue(loud)
        self.assertEqual(manager.metrics()["bufferAllocations"], 2)
        self.assertEqual(first.to_bytes(), audio_to_pcm_s16le(silent))
        self.assertEqual(second.to_bytes(), audio_to_pcm_s16le(loud))

        third = manager.enqueue(silent)
        self.assertEqual(third.to_bytes(), audio_to_pcm_s16le(silent))
        self.assertEqual(manager.metrics()["bufferReuses"], 1)

    def test_chunks_to_pcm_s16le(self):
        chunks = [
            StreamingAudioChunk(
                audio=torch.tensor([[0.0, 0.5]], dtype=torch.float32),
                sample_rate=24000,
                index=0,
                is_final=False,
                start_sample=0,
                end_sample=2,
                generated_tokens=24,
            ),
            StreamingAudioChunk(
                audio=torch.tensor([[-0.5]], dtype=torch.float32),
                sample_rate=24000,
                index=1,
                is_final=True,
                start_sample=2,
                end_sample=3,
                generated_tokens=30,
            ),
        ]

        pcm = b"".join(chunks_to_pcm_s16le(chunks))

        self.assertEqual(len(pcm), 6)

    def test_write_chunks_to_wav(self):
        chunks = [
            StreamingAudioChunk(
                audio=torch.zeros(1, 8),
                sample_rate=24000,
                index=0,
                is_final=False,
                start_sample=0,
                end_sample=8,
                generated_tokens=24,
            ),
            StreamingAudioChunk(
                audio=torch.ones(1, 4) * 0.25,
                sample_rate=24000,
                index=1,
                is_final=True,
                start_sample=8,
                end_sample=12,
                generated_tokens=30,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_chunks_to_wav(Path(tmpdir) / "stream.wav", chunks)
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 24000)
                self.assertEqual(wav_file.getnframes(), 12)


if __name__ == "__main__":
    unittest.main()
