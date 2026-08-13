# Examples

## Select one tag

```bash
kb-artifact select --chunk-glob 'data/*.jsonl' --tag runbook --output selected
```

## Match one of several tags

```bash
kb-artifact select \
  --chunk-glob 'data/*.jsonl' \
  --tag procedure \
  --tag checklist \
  --output procedures
```

## Combine field and text filters

```bash
kb-artifact select \
  --summary-glob 'summaries/*.jsonl' \
  --field actionable=true \
  --text 'checklist|steps' \
  --output actionable
```

## Bound a private inspection

```bash
kb-artifact inspect source \
  --chunk-glob 'data/*.jsonl' \
  --max-files-per-kind 10 \
  --max-records 100 \
  --output source-report.json
```

## Python

```python
from kb_artifacts import SelectionRequest, select

result = select(
    SelectionRequest(chunk_globs=("data/*.jsonl",), tags=("runbook",)),
    output="selected-python",
)
print(result["counts"])
```

The repository also ships a sanitized, runnable collection in `examples/basic` and
the script `examples/python_api.py`.
