from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


TOOLS = Path(__file__).parents[1] / "tools"
import sys

sys.path.insert(0, str(TOOLS))
from verify_contract_release import ReleaseVerificationError, verify_release  # noqa: E402


SCHEMAS = {
    "contracts/schemas/module.v1.schema.json": ("kb.module", "1.0"),
    "contracts/schemas/knowledge_artifact_manifest.v1.schema.json": (
        "kb.knowledge_artifact_manifest",
        "1.0",
    ),
    "contracts/schemas/knowledge_profile_claim.v1.schema.json": (
        "kb.knowledge_profile_claim",
        "1.0",
    ),
}


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    hashes: dict[str, str] = {}
    schema_entries = []
    for relative, (schema_id, schema_version) in SCHEMAS.items():
        value = (json.dumps({"schema_id": schema_id, "schema_version": schema_version}) + "\n").encode()
        hashes[relative] = _write(root / relative, value)
        schema_entries.append(
            {"schema_id": schema_id, "schema_version": schema_version, "schema_ref": relative}
        )
    for relative, value in {
        "contracts/test_vectors/stable_references.v1.json": b"{}\n",
        "contracts/migrations/v1-rc1.md": b"# Migration\n",
    }.items():
        hashes[relative] = _write(root / relative, value)
    manifest = {
        "schema_id": "kb.interop_release",
        "schema_version": "1.0",
        "release_id": "kb-interop.v1-rc1",
        "status": "release_candidate",
        "source_commit": "10fcfa001f93fd222f96ee9d37f5428104791156",
        "schemas": schema_entries,
        "test_vectors": ["contracts/test_vectors/stable_references.v1.json"],
        "migration_ref": "contracts/migrations/v1-rc1.md",
        "validation": {"command": "npm run contract:validate", "offline": True, "deterministic": True},
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(hashes.items())],
    }
    manifest_path = root / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, root


def test_verifies_complete_immutable_release_bundle(tmp_path: Path) -> None:
    manifest, root = _bundle(tmp_path)
    result = verify_release(manifest, root)
    assert result["release_id"] == "kb-interop.v1-rc1"
    assert result["source_commit"] == "10fcfa001f93fd222f96ee9d37f5428104791156"
    assert result["verified_file_count"] == 5
    assert result["offline"] is result["deterministic"] is True


def test_rejects_missing_or_hash_mismatched_release_file(tmp_path: Path) -> None:
    manifest, root = _bundle(tmp_path)
    target = root / "contracts/migrations/v1-rc1.md"
    target.write_text("altered", encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match="hash mismatch"):
        verify_release(manifest, root)
    target.unlink()
    with pytest.raises(ReleaseVerificationError, match="missing or unsafe"):
        verify_release(manifest, root)


def test_rejects_schema_identity_mismatch_and_unsafe_paths(tmp_path: Path) -> None:
    manifest, root = _bundle(tmp_path)
    schema_path = root / "contracts/schemas/module.v1.schema.json"
    schema_path.write_text('{"schema_id":"wrong","schema_version":"1.0"}\n', encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in payload["files"]:
        if entry["path"] == "contracts/schemas/module.v1.schema.json":
            entry["sha256"] = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match="schema identity mismatch"):
        verify_release(manifest, root)

    manifest, root = _bundle(tmp_path / "unsafe")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../outside.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match="not portable"):
        verify_release(manifest, root)
