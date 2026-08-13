# Getting started

## Install

`kb-artifacts` requires Python 3.10 or newer.

```bash
python -m pip install kb-artifacts
kb-artifact --help
```

## Create an input

JSONL stores one JSON object per line. Create a small collection:

```bash
cat > evidence.jsonl <<'EOF'
{"title":"Deploy app","text":"Build and deploy the service","tags":["runbook","ops"]}
{"title":"Buy groceries","text":"Milk and bread","tags":["personal"]}
EOF
```

## Inspect without exposing bodies

```bash
kb-artifact inspect source \
  --chunk-glob evidence.jsonl \
  --output source-report.json
```

Inspection reports bounded schema and tag statistics. Text excerpts are excluded
unless `--include-excerpts` is explicitly supplied.

## Select and export

```bash
kb-artifact select \
  --chunk-glob evidence.jsonl \
  --tag runbook \
  --output selected
```

The output directory must be absent or empty. Remove it before repeating the example;
the tool does not overwrite an existing result silently.

## Use Python instead

```python
from kb_artifacts import SelectionRequest, select

request = SelectionRequest(
    chunk_globs=("evidence.jsonl",),
    tags=("runbook",),
)
manifest = select(request, output="selected-python")
print(manifest["counts"]["selected"])
```

Continue with [JSONL format](jsonl-format.md) or the [CLI reference](cli.md).
