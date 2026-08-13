from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_critical_open_source_files_exist_and_are_linked() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for filename in ("LICENSE", "CONTRIBUTING.md", "CHANGELOG.md", "SECURITY.md"):
        assert (ROOT / filename).is_file()
        assert filename in readme


def test_project_metadata_links_to_the_changelog_file() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["urls"]["Changelog"].endswith("/blob/main/CHANGELOG.md")


def test_github_contribution_templates_exist() -> None:
    expected = (
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/pull_request_template.md",
    )
    for relative in expected:
        assert (ROOT / relative).is_file()
    config = (ROOT / ".github/ISSUE_TEMPLATE/config.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: true" in config
    assert "/security/advisories/new" in config


def test_changelog_does_not_claim_an_unperformed_release() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## Unreleased" in changelog
    assert "No production version has been recorded" in changelog
    assert "## [0.1.0]" not in changelog
