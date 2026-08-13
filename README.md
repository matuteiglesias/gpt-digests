# kb-artifacts

[![CI](https://github.com/matuteiglesias/kb-artifacts/actions/workflows/ci.yml/badge.svg)](https://github.com/matuteiglesias/kb-artifacts/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kb-artifacts.svg)](https://pypi.org/project/kb-artifacts/)
[![Python: >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`kb-artifacts` is a small Python library and CLI for inspecting, filtering,
selecting, and reproducibly exporting records from JSONL evidence collections.

## What it does

- Reads newline-delimited JSON without modifying the source files.
- Accepts common fields including `title`, `text`, `content`, `summary`, `tags`,
  timestamps, and nested `meta` values.
- Filters records by tags, fields, dates, text, family, and maturity.
- Supports explicit, nested JSON query expressions and deterministic corpus
  describe/facet/count/sample operations.
- Resolves optional local corpus profiles without exposing their filesystem globs in
  normal agent-facing output.
- Writes the same selection as JSONL, CSV, readable Markdown, and a provenance
  manifest.

## Install

```bash
python -m pip install kb-artifacts
kb-artifact --help
```

Python 3.10 or newer is required.

## 60-second example

Create a two-record JSONL collection:

```bash
cat > evidence.jsonl <<'EOF'
{"title":"Deploy app","text":"Build and deploy the service","tags":["runbook","ops"]}
{"title":"Buy groceries","text":"Milk and bread","tags":["personal"]}
EOF
```

Select the runbook record:

```bash
kb-artifact select \
  --chunk-glob evidence.jsonl \
  --tag runbook \
  --output selected
```

The command reports `selected=1` and creates:

```text
selected/
├── selected.jsonl
├── selected.csv
├── artifact.md
└── manifest.json
```

`selected.jsonl` is the machine-readable selection, `selected.csv` is convenient
for spreadsheets, `artifact.md` is readable output, and `manifest.json` records the
request, input checksum, counts, and generated files.

## Python API

```python
from kb_artifacts import SelectionRequest, select

request = SelectionRequest(
    chunk_globs=("evidence.jsonl",),
    tags=("runbook",),
)

result = select(request, output="selected-python")
print(result["counts"]["selected"])  # 1
```

The supported public API for 0.2 preserves the original contracts and adds the typed
query, corpus-exploration, and profile-loading boundary documented on the
[Python API documentation](https://matuteiglesias.github.io/kb-artifacts/python-api/)
page. See the runnable offline [agent example](examples/agent/progressive_query.py)
for the progressive discovery-to-selection workflow.

## CLI

Inspect bounded source metadata without including record bodies:

```bash
kb-artifact inspect source \
  --chunk-glob evidence.jsonl \
  --output source-report.json
```

Use `kb-artifact --help`, `kb-artifact select --help`, or
`kb-artifact corpus --help` for all options. Repeated tags use OR semantics;
different filter types must all match.

## Documentation / development

The [documentation](https://matuteiglesias.github.io/kb-artifacts/) covers selection,
provenance, operational guarantees, and interoperability. Runnable examples live in
[`examples/`](examples/).

For a development checkout:

```bash
python -m pip install -e '.[dev,docs]'
make test
make smoke
make distribution-test
```

`make distribution-test` builds both distribution formats, installs the wheel into
an isolated temporary environment, verifies the public imports, runs the installed
CLI help, and selects the minimal example outside the source checkout.

Source code and issues are hosted on
[GitHub](https://github.com/matuteiglesias/kb-artifacts).
See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change,
[CHANGELOG.md](CHANGELOG.md) for user-visible changes, and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.
