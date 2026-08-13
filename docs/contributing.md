# Contributing

Keep changes within the package's narrow purpose: deterministic inspection,
selection, manifests, promotion, and publication mechanics for JSONL evidence.

## Set up a checkout

```bash
git clone https://github.com/matuteiglesias/kb-artifacts.git
cd kb-artifacts
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,docs]'
```

## Run checks

```bash
make test
make smoke
make distribution-test
mkdocs build --strict
```

`distribution-test` builds the wheel and sdist, installs the wheel in a fresh
environment outside the checkout, imports the public API, and runs the installed CLI.

Contract-release verification additionally requires the exact approved release
bundle supplied locally; it is not inferred or downloaded automatically.

## Change discipline

- Do not change ranking, eligibility, exclusion, deduplication, scoring, or promotion
  semantics without explicit owner approval.
- Never hand-edit generated selections or manifests.
- Use minimal sanitized fixtures, not copied evidence collections.
- Add tests for public behavior and run distribution verification for packaging
  changes.
- Do not publish or promote merely because generation or CI succeeded.
