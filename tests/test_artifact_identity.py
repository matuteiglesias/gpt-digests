from __future__ import annotations

import hashlib

from kb_artifacts.artifact_identity import (
    SELECTED_EVIDENCE_ID_ALGORITHM,
    selected_evidence_artifact_id,
)


PAYLOAD = b'{"record_id":"fixture:one"}\n'


def test_selected_evidence_artifact_id_stable_vector() -> None:
    expected_digest = hashlib.sha256(PAYLOAD).hexdigest()
    assert SELECTED_EVIDENCE_ID_ALGORITHM == "selected-evidence-sha256.v1"
    assert selected_evidence_artifact_id(PAYLOAD) == f"selected-evidence.sha256.{expected_digest}"
    assert expected_digest == "ff2aefc4720bf0defe0fd3fcdf7b61a27c895823958ee77f06502fb9377314db"


def test_one_byte_change_changes_artifact_id() -> None:
    assert selected_evidence_artifact_id(PAYLOAD) != selected_evidence_artifact_id(PAYLOAD + b"\n")


def test_noncanonical_context_does_not_affect_artifact_id() -> None:
    baseline = selected_evidence_artifact_id(PAYLOAD)

    contexts = (
        {"output": "/tmp/first", "generated_at": "2026-01-01", "csv": b"first", "markdown": b"# First"},
        {"output": "/another/path", "generated_at": "2027-02-02", "csv": b"second", "markdown": b"# Second"},
    )

    assert {selected_evidence_artifact_id(PAYLOAD) for _context in contexts} == {baseline}


def test_artifact_id_is_usable_as_an_opaque_reference() -> None:
    published = selected_evidence_artifact_id(PAYLOAD)
    consumer_reference = published
    assert consumer_reference.startswith("selected-evidence.sha256.")
    assert consumer_reference == published
