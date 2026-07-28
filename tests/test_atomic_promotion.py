from __future__ import annotations

from pathlib import Path

import pytest

from kb_artifacts.selection import SelectionRequest, select
from kb_artifacts.sources.jsonl_bus import SourceInputError


FIXTURE = Path(__file__).parent / "fixtures" / "profile1" / "chunks.jsonl"
LEGACY_OUTPUTS = {"selected.jsonl", "selected.csv", "artifact.md", "manifest.json"}


def _request() -> SelectionRequest:
    return SelectionRequest(chunk_globs=(str(FIXTURE),), tags=("run book",))


def _staging_siblings(output: Path) -> list[Path]:
    return list(output.parent.glob(f".{output.name}.staging-*"))


@pytest.mark.parametrize(
    "failed_filename",
    ["selected.jsonl", "selected.csv", "artifact.md", "manifest.json"],
)
def test_write_failure_leaves_no_partial_output_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_filename: str,
) -> None:
    output = tmp_path / "run"
    source_before = FIXTURE.read_bytes()

    from kb_artifacts import selection

    real_write = selection._write_candidate_file

    def fail_selected_write(path: Path, content: bytes) -> None:
        if path.name == failed_filename:
            raise OSError("injected write failure")
        real_write(path, content)

    monkeypatch.setattr(selection, "_write_candidate_file", fail_selected_write)
    with pytest.raises(SourceInputError, match="Could not publish"):
        select(_request(), output=output)

    assert not output.exists()
    assert _staging_siblings(output) == []
    assert FIXTURE.read_bytes() == source_before

    monkeypatch.setattr(selection, "_write_candidate_file", real_write)
    select(_request(), output=output)
    assert {path.name for path in output.iterdir()} == LEGACY_OUTPUTS


@pytest.mark.parametrize("failure_point", ["validation", "promotion"])
def test_late_failure_cleans_staging_and_preserves_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    output = tmp_path / "run"
    output.mkdir()

    from kb_artifacts import selection

    symbol = "_validate_candidate" if failure_point == "validation" else "_promote_candidate"
    original = getattr(selection, symbol)

    def fail(*_args, **_kwargs):
        if failure_point == "validation":
            raise SourceInputError("injected validation failure")
        raise OSError("injected promotion failure")

    monkeypatch.setattr(selection, symbol, fail)
    with pytest.raises(SourceInputError):
        select(_request(), output=output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert _staging_siblings(output) == []

    monkeypatch.setattr(selection, symbol, original)
    select(_request(), output=output)
    assert {path.name for path in output.iterdir()} == LEGACY_OUTPUTS


def test_symlinked_output_is_rejected_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("untouched", encoding="utf-8")
    output = tmp_path / "run"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(SourceInputError, match="must not contain a symlink"):
        select(_request(), output=output)

    assert marker.read_text(encoding="utf-8") == "untouched"
    assert _staging_siblings(output) == []


def test_symlinked_output_parent_is_rejected(tmp_path: Path) -> None:
    physical_parent = tmp_path / "physical"
    physical_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(physical_parent, target_is_directory=True)
    output = linked_parent / "run"

    with pytest.raises(SourceInputError, match="must not contain a symlink"):
        select(_request(), output=output)

    assert not (physical_parent / "run").exists()
    assert list(physical_parent.iterdir()) == []
