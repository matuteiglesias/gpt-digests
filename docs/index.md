# kb-artifacts

`kb-artifact` is a read-only selector for governed JSONL evidence buses. It scans
metadata without changing sources, then exports a deterministic selected record set.

## Install

```bash
python -m pip install -e . --no-build-isolation
```

## Inspect a source

```bash
kb-artifact inspect source --chunk-glob "$CHUNK_GLOB" --output artifacts/source-report.json
```

The source report contains bounded schema, field, and normalized tag statistics. It
omits source bodies unless explicitly requested.

## Select evidence

```bash
kb-artifact select --chunk-glob "$CHUNK_GLOB" --tag runbook --tag procedure --tag checklist --family operations --output artifacts/runs/operations
kb-artifact select --chunk-glob "$CHUNK_GLOB" --field category=cooking --output artifacts/runs/cooking
kb-artifact select --summary-glob "$SUMMARY_GLOB" --field actionable=true --text 'checklist|steps|pasos' --group-by domain --output artifacts/runs/actionable
```

Filters combine predictably: repeated tags use OR semantics, while distinct filter
kinds must all match. `--allow-empty` is required to write an intentional empty run.

## Output and provenance

Each successful run contains `selected.jsonl`, `selected.csv`, `artifact.md`, and
`manifest.json`. The JSONL preserves record identity, selected annotations, source
provenance, and selection reasons. CSV uses bounded text excerpts. The manifest records
the request, input partition hashes, counts, and generated output names.

The existing `manifest.json` is producer-local operational evidence; it is not a
shared knowledge-artifact manifest. The canonical knowledge product is
`selected.jsonl`. Its producer-owned, content-addressed identity is documented in
[the interoperability boundary](interoperability.md). Shared module and artifact
manifests remain gated on an exact, locally verified KB Contracts release bundle;
the producer has no runtime dependency on a sibling repository.
