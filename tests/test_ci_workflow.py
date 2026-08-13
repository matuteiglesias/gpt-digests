from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def test_ci_covers_supported_distribution_boundaries() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for version in ('"3.10"', '"3.12"', '"3.14"'):
        assert version in workflow
    for command in (
        "python -m pytest -q",
        "python -m build",
        "python -m twine check --strict dist/*",
        "python -m venv",
        "pip install dist/*.whl",
        "from kb_artifacts import",
        'kb-artifact" --help',
    ):
        assert command in workflow


def test_ci_installed_package_checks_are_isolated_from_checkout() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "working-directory: ${{ runner.temp }}" in workflow
    assert "PYTHONPATH" not in workflow
    assert 'contents: read' in workflow
