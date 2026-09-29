"""Pinned native Nano loading; keep its dependencies out of the core tests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from .streaming import NanoStreamer, StreamConfig

SOURCE_REVISION = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
MODEL_REPO = "ResembleAI/chatterbox-nano"
MODEL_REVISION = "71ccd1d0081b430592cea481f4307e764e07bc64"
ASSET_FILES = ("t3_nano_v1.safetensors", "s3gen_meanflow.safetensors",
               "ve.safetensors", "conds.pt", "vocab.json", "merges.txt",
               "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_upstream_source(package_root, manifest_path=None):
    """Verify native code for Git, editable and wheel installs alike."""
    manifest_path = Path(manifest_path or Path(__file__).with_name("upstream-source.json"))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != SOURCE_REVISION or not manifest.get("files"):
        raise RuntimeError("Invalid pinned Chatterbox source manifest")
    package_root = Path(package_root).resolve()
    for name, expected in manifest["files"].items():
        path = package_root / name
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Chatterbox source differs from the tested revision: {name}")
    return {"source_revision": SOURCE_REVISION,
            "native_package_path": str(package_root),
            "source_validation": "upstream_file_hashes",
            "upstream_source_manifest_sha256": file_sha256(manifest_path)}


def load_nano(*, device="cpu", config=StreamConfig(), local_assets=None):
    import chatterbox
    import torch
    from huggingface_hub import snapshot_download
    from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
    from chatterbox.models.s3gen.const import S3GEN_SIL, S3GEN_SR

    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Streaming supports CPU, CUDA or MPS")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine")
    source = validate_upstream_source(Path(chatterbox.__file__).parent)
    dist = importlib.metadata.distribution("chatterbox-tts")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    installed_revision = direct.get("vcs_info", {}).get("commit_id")
    assets = Path(local_assets) if local_assets else Path(snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, allow_patterns=list(ASSET_FILES), max_workers=2))
    missing = [name for name in ASSET_FILES if not (assets / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Nano assets: {missing}")
    hashes = {name: file_sha256(assets / name) for name in ASSET_FILES}
    model = ChatterboxTurboTTS.from_local(assets, device=device, nano=True)
    if model.sr != S3GEN_SR or S3GEN_SIL != 4299 or not model.s3gen.meanflow:
        raise ValueError("Unexpected Nano speech decoder contract")
    streamer = NanoStreamer(model, config, normalize_text=punc_norm)
    provenance = {
        **source, "source_install_revision": installed_revision, "model_repo": MODEL_REPO,
        "experiment_code_sha256": {name: file_sha256(Path(__file__).parent / name)
                                   for name in ("streaming.py", "runtime.py", "__main__.py")},
        "model_revision": MODEL_REVISION if local_assets is None else None,
        "local_assets": str(assets.resolve()), "asset_sha256": hashes,
        "device": device, "dtype": str(model.t3.text_emb.weight.dtype),
        "component_devices": {name: str(next(module.parameters()).device)
                              for name, module in (("t3", model.t3), ("s3gen", model.s3gen),
                                                   ("vocoder", model.s3gen.mel2wav), ("voice", model.ve))},
        "mps_operator_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "embedding_width": model.t3.dim, "batch_size": 1,
        "total_parameters": sum(p.numel() for module in (model.t3, model.s3gen, model.ve)
                                for p in module.parameters()),
        "trainable_parameters": sum(p.numel() for module in (model.t3, model.s3gen, model.ve)
                                    for p in module.parameters() if p.requires_grad),
        "asset_bytes": sum((assets / name).stat().st_size for name in ASSET_FILES),
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("torch", "transformers", "chatterbox-tts", "torchaudio")},
        "threads": torch.get_num_threads(),
    }
    return streamer, provenance
