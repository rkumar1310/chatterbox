"""Run real Nano audio cases sequentially and export matched listening examples."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def verify(folder, case, device):
    report = json.loads((folder / "report.json").read_text())
    metrics, config = report["metrics"], report["config"]
    assert report["text"] == case["text"]
    assert config["max_gen_len"] == case["max_gen_len"]
    assert config["decoder_seed"] == case["seed"] + 1
    assert config["audio_mode"] == "conditioned_history"
    assert config["left_context_tokens"] == 25 and config["vocoder_lookahead_ms"] == 0
    assert report["provenance"]["device"] == device
    assert report["provenance"]["trainable_parameters"] == 0
    assert metrics["native_speech_token_parity"]
    assert metrics["stop_reason"] == "eos", "Generation cap reached before EOS"
    assert metrics["max_conditioning_history_tokens"] <= 25
    rows = report["chunks"]
    assert rows[-1]["final"] and sum(row["final"] for row in rows) == 1
    offset = 0
    for row in rows:
        assert row["start_sample"] == offset, "Chunk gap or duplication"
        offset += row["samples"]
    waveform, sr = sf.read(folder / "streamed.wav", dtype="int16")
    chunk_data = [sf.read(folder / row["file"], dtype="int16")[0] for row in rows if row["file"]]
    assert len(waveform) == offset == metrics["audio_samples"] and sr == 24000
    assert np.array_equal(waveform, np.concatenate(chunk_data))
    assert np.any(waveform != 0), "Empty/silent output"
    early_audio = any(row["samples"] and not row["final"] for row in rows)
    if early_audio:
        assert metrics["first_audio_chunk_seconds"] < metrics["total_seconds"]
    else:
        valid_tokens = rows[-1]["speech_tokens"]
        assert valid_tokens < config["chunk_tokens"], "No audio before final chunk"
    for name, expected in report["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    native, native_sr = sf.read(folder / "native.wav", dtype="int16")
    assert native_sr == sr and len(native) == len(waveform), "Native/stream duration mismatch"
    if device == "mps":
        assert set(report["provenance"]["component_devices"].values()) == {"mps:0"}
        assert not report["provenance"]["mps_operator_cpu_fallback_enabled"]
    signal = waveform.astype(np.float64) / 32768
    return {"id": case["id"], "group": case["group"], "text": case["text"],
            "seed": case["seed"], "device": device, "audio_seconds": metrics["audio_seconds"],
            "audio_chunks": metrics["chunk_count"],
            "first_audio_seconds": metrics["first_audio_chunk_seconds"],
            "total_seconds": metrics["total_seconds"], "input_tokens": metrics["input_tokens"],
            "max_conditioning_history_tokens": metrics["max_conditioning_history_tokens"],
            "max_decoder_input_tokens": metrics["max_decoder_input_tokens"],
            "audio_before_eos": early_audio,
            "sample_rate": sr, "dtype": report["provenance"]["dtype"],
            "peak_observed_mps_driver_bytes": metrics.get("max_observed_mps_driver_bytes"),
            "streamed_wav_sha256": report["files"]["streamed.wav"],
            "native_wav_sha256": report["files"]["native.wav"],
            "rms": float(np.sqrt(np.mean(signal * signal))),
            "clipped_sample_fraction": float(np.mean(np.abs(signal) >= 32767 / 32768)),
            "automatic_checks": "passed", "listening_review": "pending"}


def export_examples(rows, output, destination):
    destination.mkdir(parents=True, exist_ok=True)
    audio = destination / "audio"
    audio.mkdir(exist_ok=True)
    for row in rows:
        for kind in ("streamed", "native"):
            shutil.copy2(output / row["id"] / f"{kind}.wav", audio / f"{row['id']}-{kind}.wav")
    (destination / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>Nano streaming listening examples</title>',
            '<style>body{font:16px system-ui;max-width:960px;margin:32px auto;padding:16px}section{border-top:1px solid #ddd;padding:20px 0}audio{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>',
            '<h1>Nano streaming listening examples</h1><p>Each pair uses the same generated speech tokens and voice reference; acoustic predictions differ between streaming and native decoding.</p>',
            '<p>Listen for pipe sounds, metallic tones, clicks, missing words, repeated words and changes in voice around chunk boundaries; automatic checks do not certify audio quality.</p>']
    md = ['# Nano streaming listening examples', '',
          'New recordings generated on the device shown below, with unchanged Nano weights, five-token chunks, 25-token history and no added vocoder delay.', '',
          '| Case | Device | Text | Duration | Streamed audio | Native audio |',
          '| --- | --- | --- | ---: | --- | --- |']
    for row in rows:
        stem = row['id']
        md.append(f"| {stem} | {row['device'].upper()} | {row['text']} | {row['audio_seconds']:.2f} s | [Streamed](audio/{stem}-streamed.wav) | [Native](audio/{stem}-native.wav) |")
        page += [f"<section><h2>{html.escape(stem)}</h2><p>{html.escape(row['text'])}</p>",
                 f"<p>{row['device'].upper()} · {row['audio_seconds']:.2f} s · {row['audio_chunks']} chunks · listening review pending</p><div class=pair>",
                 f'<div><p>Streamed</p><audio controls preload="none" src="audio/{stem}-streamed.wav"></audio></div>',
                 f'<div><p>Native</p><audio controls preload="none" src="audio/{stem}-native.wav"></audio></div></div></section>']
    page.append('</html>')
    (destination / "listen.html").write_text('\n'.join(page) + '\n')
    md += ['', 'Open `listen.html` locally for paired audio players; recordings and `results.json` are versioned so listening feedback can name an exact case and timestamp.', '',
           'All recorded cases passed native speech-token parity, EOS completion, history bounds, finite samples, sample continuity, chunk-to-WAV PCM equality, native duration equality and file hash checks; audio precedes EOS when the utterance reaches the five-token chunk threshold.', '',
           'Listening review remains pending; these checks do not measure phonetic accuracy, intelligibility or freedom from acoustic artifacts.', '',
           'Timings follow a per-case warm-up and exclude loading, preprocessing and speaker preparation; first audio means the first nonempty waveform chunk returned to the consumer.', '',
           'See `results.json` for device, FP32 dtype, seed, input length, chunk count, timing and sampled Metal driver memory; CPU uses two threads and batch size one.', '',
           'Run `python scripts/validate_nano_streaming.py --device cpu --output artifacts/new-validation --export-listening examples/nano_streaming` to regenerate all nine cases sequentially on CPU; use `--device mps` to test Metal explicitly.', '',
           'Use `--only 01-simple-weather` for one case or `--resume` to reuse completed recordings whose inputs, settings, hashes and source still match.']
    (destination / "README.md").write_text('\n'.join(md) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--cases', type=Path, default=ROOT / 'examples/nano_streaming/cases.json')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-listening', type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if args.only:
        unknown = set(args.only) - {case['id'] for case in cases}
        if unknown:
            parser.error(f'Unknown case IDs: {sorted(unknown)}')
        cases = [case for case in cases if case['id'] in args.only]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1'}
    for case in cases:
        folder = args.output / case['id']
        print(json.dumps({'started': case['id'], 'device': args.device, 'text': case['text']}), flush=True)
        if args.resume and (folder / 'report.json').is_file():
            report = json.loads((folder / 'report.json').read_text())
            for name, expected in report['provenance']['experiment_code_sha256'].items():
                assert hashlib.sha256((ROOT / 'src/chatterbox/nano_streaming' / name).read_bytes()).hexdigest() == expected
        else:
            cmd = [sys.executable, '-m', 'chatterbox.nano_streaming', '--device', args.device,
                   '--text', case['text'], '--seed', str(case['seed']), '--max-gen-len', str(case['max_gen_len']),
                   '--output', str(folder)]
            if args.assets:
                cmd += ['--assets', str(args.assets)]
            with (args.output / f"{case['id']}.log").open('w') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = verify(folder, case, args.device)
        rows.append(row)
        (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        if args.export_listening:
            export_examples(rows, args.output, args.export_listening)
        print(json.dumps({'finished': case['id'], **row}), flush=True)
    print(json.dumps({'completed': len(rows), 'device': args.device}), flush=True)


if __name__ == '__main__':
    main()
