# Contributing to kb-artifacts

Thank you for helping improve `kb-artifacts`. The project is intentionally narrow: it
inspects, filters, selects, and reproducibly exports records from JSONL evidence
collections.

## Before opening a change

- Search [existing issues](https://github.com/matuteiglesias/kb-artifacts/issues).
- Use an issue template for a reproducible defect or a focused feature proposal.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Obtain owner approval before changing ranking, eligibility, exclusion,
  deduplication, scoring, promotion, manifest, or interoperability semantics.

Please do not use an unrelated change to add databases, vector stores, LLM
integrations, servers, plugin systems, web interfaces, non-JSONL formats, or a new
packaging or CLI framework.

## Development setup

```bash
git clone https://github.com/matuteiglesias/kb-artifacts.git
cd kb-artifacts
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,docs]'
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Make a focused change

- Keep selection logic separate from source interpretation.
- Keep randomness absent or explicit and seeded.
- Add minimal, sanitized fixtures; never commit private evidence or large corpora.
- Do not hand-edit generated outputs or manifests. Fix their producer and rerun it.
- Preserve failed-run evidence and never describe partial output as promoted success.
- Update public documentation and `CHANGELOG.md` when user-visible behavior changes.

## Run checks

```bash
make test
make smoke
make distribution-test
python -m mkdocs build --strict
```

`make distribution-test` builds the wheel and sdist, installs the wheel in a clean
environment outside the checkout, imports the public API, and exercises the installed
CLI. Contract-release verification requires an exact approved bundle supplied locally;
do not infer or download one silently.

## Open a pull request

Complete the pull request template, including the selection-policy declaration,
commands run, generated artifacts, and publication/promotion declaration. Link the
smallest relevant issue. A green build is evidence for review, not approval to publish
a package or promote evidence.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
