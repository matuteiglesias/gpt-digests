#!/usr/bin/env python3
"""Verify an immutable KB Contracts release bundle without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_RELEASE_ID = "kb-interop.v1-rc1"
EXPECTED_SOURCE_COMMIT = "10fcfa001f93fd222f96ee9d37f5428104791156"
REQUIRED_SCHEMAS = {
    ("kb.module", "1.0"): "contracts/schemas/module.v1.schema.json",
    ("kb.knowledge_artifact_manifest", "1.0"): (
        "contracts/schemas/knowledge_artifact_manifest.v1.schema.json"
    ),
    ("kb.knowledge_profile_claim", "1.0"): (
        "contracts/schemas/knowledge_profile_claim.v1.schema.json"
    ),
}


class ReleaseVerificationError(ValueError):
    """Raised when supplied release evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"could not read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"expected a JSON object: {path}")
    return value


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReleaseVerificationError("release file path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ReleaseVerificationError(f"release file path is not portable: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(manifest_path: Path, bundle_root: Path) -> dict[str, Any]:
    """Verify identity, inventory hashes, schema identities, and offline claims."""
    manifest = _load_json(manifest_path)
    expected_roots = {
        "schema_id": "kb.interop_release",
        "schema_version": "1.0",
        "release_id": EXPECTED_RELEASE_ID,
        "status": "release_candidate",
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }
    for field, expected in expected_roots.items():
        if manifest.get(field) != expected:
            raise ReleaseVerificationError(
                f"unexpected {field}: expected {expected!r}, got {manifest.get(field)!r}"
            )

    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ReleaseVerificationError("release validation declaration is missing")
    if validation.get("offline") is not True or validation.get("deterministic") is not True:
        raise ReleaseVerificationError("release validation must be offline and deterministic")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ReleaseVerificationError("release file inventory is missing")

    declared: dict[str, str] = {}
    verified: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReleaseVerificationError("release file entry must be an object")
        relative = _safe_relative_path(entry.get("path"))
        relative_text = relative.as_posix()
        wanted_hash = entry.get("sha256")
        if (
            not isinstance(wanted_hash, str)
            or len(wanted_hash) != 64
            or any(character not in "0123456789abcdef" for character in wanted_hash)
        ):
            raise ReleaseVerificationError(f"invalid SHA-256 declaration: {relative_text}")
        if relative_text in declared:
            raise ReleaseVerificationError(f"duplicate release file: {relative_text}")
        declared[relative_text] = wanted_hash
        physical = bundle_root.joinpath(*relative.parts)
        if not physical.is_file() or physical.is_symlink():
            raise ReleaseVerificationError(f"release file is missing or unsafe: {relative_text}")
        actual_hash = _sha256(physical)
        if actual_hash != wanted_hash:
            raise ReleaseVerificationError(
                f"release hash mismatch: {relative_text}: expected {wanted_hash}, got {actual_hash}"
            )
        verified.append({"path": relative_text, "sha256": actual_hash})

    schemas = manifest.get("schemas")
    if not isinstance(schemas, list):
        raise ReleaseVerificationError("release schema inventory is missing")
    observed_schemas: dict[tuple[str, str], str] = {}
    for entry in schemas:
        if not isinstance(entry, dict):
            raise ReleaseVerificationError("release schema entry must be an object")
        key = (entry.get("schema_id"), entry.get("schema_version"))
        schema_ref = _safe_relative_path(entry.get("schema_ref")).as_posix()
        if schema_ref not in declared:
            raise ReleaseVerificationError(f"schema is absent from release files: {schema_ref}")
        schema = _load_json(bundle_root / schema_ref)
        if schema.get("schema_id") != key[0] or schema.get("schema_version") != key[1]:
            raise ReleaseVerificationError(f"schema identity mismatch: {schema_ref}")
        observed_schemas[key] = schema_ref
    if observed_schemas != REQUIRED_SCHEMAS:
        raise ReleaseVerificationError(
            f"unexpected release schema set: expected {REQUIRED_SCHEMAS!r}, got {observed_schemas!r}"
        )

    for reference in manifest.get("test_vectors", []):
        relative = _safe_relative_path(reference).as_posix()
        if relative not in declared:
            raise ReleaseVerificationError(f"test vector is absent from release files: {relative}")
    migration_ref = _safe_relative_path(manifest.get("migration_ref")).as_posix()
    if migration_ref not in declared:
        raise ReleaseVerificationError("migration document is absent from release files")

    return {
        "release_id": manifest["release_id"],
        "source_commit": manifest["source_commit"],
        "status": manifest["status"],
        "verified_file_count": len(verified),
        "verified_files": verified,
        "schemas": [
            {"schema_id": key[0], "schema_version": key[1], "schema_ref": value}
            for key, value in sorted(observed_schemas.items())
        ],
        "offline": True,
        "deterministic": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_release(arguments.manifest, arguments.bundle_root)
    except ReleaseVerificationError as error:
        parser.exit(2, f"CONTRACT RELEASE INVALID: {error}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
