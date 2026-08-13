# Basic CLI example

From the repository root, select the runbook record from the sample collection:

```bash
kb-artifact select \
  --chunk-glob examples/basic/evidence.jsonl \
  --tag runbook \
  --output selected
```

The command selects one of the two records and creates `selected.jsonl`,
`selected.csv`, `artifact.md`, and `manifest.json` inside `selected/`.

Remove `selected/` before running the command again. Selection output directories
must be empty so an existing result cannot be overwritten accidentally.
