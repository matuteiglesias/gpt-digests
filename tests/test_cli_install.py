from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_distribution_declares_the_public_cli_entry_point() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"kb-artifact": "kb_artifacts.cli:main"}


def test_distribution_verifier_does_not_use_the_source_tree_for_imports() -> None:
    verifier = (ROOT / "tools" / "verify_distribution.py").read_text(encoding="utf-8")
    assert '"-I"' in verifier
    assert "PYTHONPATH" not in verifier
