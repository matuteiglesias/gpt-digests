# kb-artifacts

A read-only governed JSONL evidence selector. `kb-artifact select` reads chunk and
summary buses through the same adapter, applies explicit filters once, and writes
`selected.jsonl`, `selected.csv`, `artifact.md`, and `manifest.json` to an output run
directory.

```bash
python -m pip install -e . --no-build-isolation
kb-artifact inspect source --chunk-glob 'data/chunks/*.jsonl' --output artifacts/source-report.json
kb-artifact select --chunk-glob 'data/chunks/*.jsonl' --tag runbook --field actionable=true --output artifacts/runs/runbook
```

See [the operator guide](docs/index.md) for selection examples, read-only guarantees,
and provenance details.
