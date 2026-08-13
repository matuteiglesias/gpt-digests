from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "testpypi.yml"


def test_rehearsal_builds_and_publishes_only_release_candidates() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for expected in (
        '"v*rc*"',
        r"v\d+\.\d+\.\d+rc\d+",
        "python -m pytest -q",
        "python -m build",
        "python -m twine check --strict dist/*",
        "name: testpypi",
        "id-token: write",
        "pypa/gh-action-pypi-publish@release/v1",
        "repository-url: https://test.pypi.org/legacy/",
        "needs: publish",
    ):
        assert expected in workflow
    assert "PYPI_TOKEN" not in workflow
    assert "password:" not in workflow


def test_external_consumer_uses_testpypi_and_runs_tutorial() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for expected in (
        "--index-url https://test.pypi.org/simple/",
        "--extra-index-url https://pypi.org/simple/",
        '"kb-artifacts==${version}"',
        "from kb_artifacts import SelectionRequest, inspect_source, select",
        'kb-artifact" --help',
        "cat > evidence.jsonl",
        "--tag runbook",
        "selected/selected.jsonl",
    ):
        assert expected in workflow
    assert "actions/checkout" not in workflow.split("  verify:", 1)[1]


def test_production_workflow_rejects_prerelease_tags() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert r"v\d+\.\d+\.\d+" in workflow
    assert "production release tag must be stable" in workflow
