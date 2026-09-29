"""Write streamed chunks, concatenated audio and a matched native comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .runtime import file_sha256, load_nano
from .streaming import StreamConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Yes, I can hear you clearly now. Please go ahead.")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/chatterbox-nano-streaming"))
    parser.add_argument("--assets", type=Path, help="Optional explicit native Nano asset directory")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=5)
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--left-context-tokens", type=int, default=25,
                         help="Generated acoustic history to retain; default is 25 tokens")
    context.add_argument("--full-context", dest="left_context_tokens", action="store_const", const=None,
                         help="Retain all generated history for a diagnostic control")
    parser.add_argument("--overlap-ms", type=float, default=40)
    parser.add_argument("--audio-mode", choices=("conditioned_history", "committed_mels", "legacy"),
                        default="conditioned_history")
    parser.add_argument("--vocoder-lookahead-ms", type=float, default=0)
    parser.add_argument("--max-gen-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.threads < 1 or not args.text.strip():
        parser.error("Threads must be positive and text must be nonempty")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory to preserve earlier results")
    import soundfile as sf

    torch.set_num_threads(args.threads)
    config = StreamConfig(chunk_tokens=args.chunk_tokens, overlap_ms=args.overlap_ms,
                          max_gen_len=args.max_gen_len, decoder_seed=args.seed + 1,
                          left_context_tokens=args.left_context_tokens)
    config = StreamConfig(**{**asdict(config), "audio_mode": args.audio_mode,
                             "vocoder_lookahead_ms": args.vocoder_lookahead_ms})
    loaded = time.perf_counter()
    streamer, provenance = load_nano(device=args.device, config=config, local_assets=args.assets)
    load_seconds = time.perf_counter() - loaded
    if args.reference:
        streamer.model.prepare_conditionals(str(args.reference))
        provenance["reference_sha256"] = file_sha256(args.reference)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Warm up actual token generation plus one partial audio decode; abandon its
    # generator afterwards so warm-up is finite and separate from measurement.
    torch.manual_seed(args.seed)
    warm_started = time.perf_counter()
    warm = streamer.stream(args.text)
    next(warm)
    warm.close()
    warm_seconds = time.perf_counter() - warm_started
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    chunks = []
    rows = []
    # An open SoundFile receives each chunk before the generator requests the next
    # token, making streamed.wav playable as data accumulates in a real consumer.
    with sf.SoundFile(args.output / "streamed.wav", mode="w", samplerate=streamer.model.sr,
                      channels=1, subtype="PCM_16") as sink:
        for i, chunk in enumerate(streamer.stream(args.text)):
            if not bool(torch.isfinite(chunk.samples).all()):
                raise RuntimeError("Nonfinite audio samples")
            if chunks and chunk.start_sample != sum(len(x) for x in chunks):
                raise RuntimeError("Noncontiguous emitted audio")
            samples = chunk.samples.numpy()
            if len(samples):
                name = f"chunk-{i:03d}.wav"
                sf.write(args.output / name, samples, chunk.sample_rate, subtype="PCM_16")
                sink.write(samples)
                sink.flush()
                chunks.append(chunk.samples)
            else:
                name = None
            rows.append({"file": name, "samples": len(samples), "start_sample": chunk.start_sample,
                         "speech_tokens": chunk.speech_tokens, "final": chunk.final,
                         "elapsed_seconds": chunk.elapsed_seconds,
                         "boundary_revision_rms": chunk.boundary_revision_rms})
            print(json.dumps(rows[-1]), flush=True)
    metrics = dict(streamer.metrics)
    if not chunks:
        raise RuntimeError("Model produced no playable audio")
    # Run native T3 separately with the same seed and full input, without any
    # interleaved waveform calls; this tests real sampling/cache parity.
    torch.manual_seed(args.seed)
    text = streamer.normalize_text(args.text)
    ids = streamer.model.tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    native_started = streamer._clock()
    with torch.inference_mode():
        native = streamer.model.t3.inference_turbo(
            streamer.model.conds.t3, ids, temperature=config.temperature, top_k=config.top_k,
            top_p=config.top_p, repetition_penalty=config.repetition_penalty,
            max_gen_len=config.max_gen_len)
    token_seconds = streamer._clock() - native_started
    emitted = metrics["speech_token_ids"]
    if emitted and emitted[-1] == streamer.model.t3.hp.stop_speech_token:
        emitted = emitted[:-1]
    parity = native[0].tolist() == emitted
    if not parity:
        raise RuntimeError("Streamed speech tokens differ from native T3")
    valid = native[native < streamer.model.s3gen.flow.vocab_size]
    valid = torch.cat((valid, torch.tensor([4299] * 3, device=valid.device)))
    # Stock offline S3Gen (its own noise convention), not a claimed exact match.
    torch.manual_seed(config.decoder_seed)
    baseline_started = streamer._clock()
    with torch.inference_mode():
        baseline, _ = streamer.model.s3gen.inference(
            valid, ref_dict=dict(streamer.model.conds.gen), n_cfm_timesteps=config.decoder_steps)
    decode_seconds = streamer._clock() - baseline_started
    sf.write(args.output / "native.wav", baseline.reshape(-1).detach().cpu().numpy(),
             streamer.model.sr, subtype="PCM_16")
    metrics.update(native_speech_token_parity=parity, native_token_seconds=token_seconds,
                   native_decode_seconds=decode_seconds,
                   speech_tokens_per_second=len(emitted) / metrics["total_seconds"],
                   real_time_factor=metrics["total_seconds"] / metrics["audio_seconds"],
                   peak_gpu_allocated_bytes=(torch.cuda.max_memory_allocated()
                                             if args.device == "cuda" else None))
    report = {"config": asdict(config), "text": args.text, "normalized_text": text,
              "provenance": provenance, "load_seconds": load_seconds,
              "warmup_seconds": warm_seconds, "metrics": metrics, "chunks": rows,
              "quality": "Listening review pending; no exact waveform parity claim",
              "files": {p.name: file_sha256(p) for p in args.output.glob("*.wav")}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
