# Changelog

All notable user-visible changes to this project will be documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Typed JSON-compatible query expressions with deterministic evaluation and optional
  integration into `SelectionRequest`.
- Read-only corpus describe, facet, count, and bounded sample APIs and JSON CLI
  commands.
- Local TOML corpus profiles, path-private discovery, and profile-backed exploration
  and selection.
- Offline progressive agent workflow example and agent-boundary documentation.

- Public package-root API for evidence contracts, inspection, and selection.
- Self-contained CLI and Python examples for JSONL evidence collections.
- PEP 561 inline typing marker and publication-grade package metadata.
- Installed-wheel distribution verification across the supported Python range.
- Canonical MkDocs documentation with automated GitHub Pages deployment.
- TestPyPI rehearsal and PyPI Trusted Publishing workflows with external-consumer
  verification and package attestations.
- Open-source contribution, security, conduct, issue, and pull request guidance.

### Changed

- `select()` accepts string and path-like output locations in addition to `Path`.
- Package version prepared as `0.2.0` for the backward-compatible query and corpus
  exploration release; publication still requires maintainer acceptance.

## Release history

No production version has been recorded in this changelog yet. After an approved
release succeeds and its external-consumer checks pass, move the relevant entries from
**Unreleased** into a dated version section without rewriting prior release entries.
