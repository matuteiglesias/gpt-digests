from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
PAGES = (
    "index.md",
    "getting-started.md",
    "jsonl-format.md",
    "cli.md",
    "python-api.md",
    "outputs.md",
    "provenance.md",
    "interoperability.md",
    "examples.md",
    "contributing.md",
    "releasing.md",
)


def test_canonical_documentation_pages_exist_and_are_navigable() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "site_url: https://matuteiglesias.github.io/kb-artifacts/" in config
    for page in PAGES:
        assert (ROOT / "docs" / page).is_file()
        assert page in config


def test_pages_workflow_builds_strictly_before_deployment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(
        encoding="utf-8"
    )
    for expected in (
        "python -m mkdocs build --strict",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v3",
        "actions/deploy-pages@v4",
        "pages: write",
        "id-token: write",
        "name: github-pages",
    ):
        assert expected in workflow
    assert "if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'" in workflow
