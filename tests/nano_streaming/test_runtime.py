import hashlib
import json

import pytest

from chatterbox.nano_streaming.runtime import SOURCE_REVISION, validate_upstream_source


def test_local_source_validation_accepts_without_git_metadata(tmp_path):
    package = tmp_path / "installed_package"
    package.mkdir()
    native = package / "tts_turbo.py"
    native.write_text("native code\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "revision": SOURCE_REVISION,
        "files": {native.name: hashlib.sha256(native.read_bytes()).hexdigest()},
    }))
    provenance = validate_upstream_source(package, manifest)
    assert provenance["source_revision"] == SOURCE_REVISION
    assert provenance["native_package_path"] == str(package)
    native.write_text("modified code\n")
    with pytest.raises(RuntimeError, match="tts_turbo.py"):
        validate_upstream_source(package, manifest)
    native.unlink()
    with pytest.raises(RuntimeError, match="tts_turbo.py"):
        validate_upstream_source(package, manifest)


def test_source_validation_rejects_unpinned_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"revision": "different", "files": {"x.py": "unused"}}))
    with pytest.raises(RuntimeError, match="Invalid pinned"):
        validate_upstream_source(tmp_path, manifest)
