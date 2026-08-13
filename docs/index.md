# kb-artifacts

**Inspect, filter, select, and reproducibly export records from JSONL evidence
collections.**

`kb-artifacts` is a small, read-only Python library and CLI. It accepts ordinary
newline-delimited JSON, applies explicit filters, and writes machine-readable,
spreadsheet-friendly, human-readable, and provenance outputs together.

## Start in one minute

```bash
python -m pip install kb-artifacts

cat > evidence.jsonl <<'EOF'
{"title":"Deploy app","text":"Build and deploy the service","tags":["runbook","ops"]}
{"title":"Buy groceries","text":"Milk and bread","tags":["personal"]}
EOF

kb-artifact select --chunk-glob evidence.jsonl --tag runbook --output selected
```

The command selects one record and creates `selected.jsonl`, `selected.csv`,
`artifact.md`, and `manifest.json` in `selected/`.

## Where to go next

- Follow [Getting started](getting-started.md) for CLI and Python walkthroughs.
- Learn which input fields are recognized in [JSONL format](jsonl-format.md).
- Browse the [CLI](cli.md) and [Python API](python-api.md) references.
- Understand [outputs](outputs.md) and [provenance](provenance.md).
- Copy a recipe from [Examples](examples.md).

The canonical documentation site is
[`https://matuteiglesias.github.io/kb-artifacts/`](https://matuteiglesias.github.io/kb-artifacts/).
