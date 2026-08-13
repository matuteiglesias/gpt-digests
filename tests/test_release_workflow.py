from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def test_release_workflow_validates_before_publishing() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for expected in (
        '"v[0-9]*.[0-9]*.[0-9]*"',
        "release tag {actual!r} does not match package version {expected!r}",
        "python -m pytest -q",
        "python -m build",
        "python -m twine check --strict dist/*",
        "pip install dist/*.whl",
        "from kb_artifacts import",
        'kb-artifact" --help',
        "sha256sum * | tee SHA256SUMS",
        "actions/upload-artifact@v4",
        "needs: build",
    ):
        assert expected in workflow


def test_publish_job_uses_trusted_publishing() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "id-token: write" in workflow
    assert "name: pypi" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "attestations: true" in workflow
    assert "PYPI_TOKEN" not in workflow
    assert "password:" not in workflow


def test_production_package_is_verified_before_github_release() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for expected in (
        "needs: publish",
        "--index-url https://pypi.org/simple/",
        '"kb-artifacts==${version}"',
        "from kb_artifacts import SelectionRequest, select",
        "print('import OK')",
        'kb-artifact" --help',
        "cat > evidence.jsonl",
        "needs: verify",
        "contents: write",
        "gh release create",
        "--verify-tag",
        "--generate-notes",
    ):
        assert expected in workflow
    verify_job = workflow.split("  verify:", 1)[1].split("  github-release:", 1)[0]
    assert "actions/checkout" not in verify_job


def test_ordinary_ci_does_not_publish() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "gh-action-pypi-publish" not in workflow
    assert "id-token: write" not in workflow
