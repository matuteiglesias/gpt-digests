# Outputs

Every successful selection creates one directory with four files.

```text
selected/
├── selected.jsonl
├── selected.csv
├── artifact.md
└── manifest.json
```

## `selected.jsonl`

The canonical selected-evidence product. Each line contains normalized identity,
common projections, annotations, tags, timestamp, provenance, selection reasons, and
optional classification values.

## `selected.csv`

A compatibility view for tabular tools. Text is bounded to an excerpt and repeated
values are rendered for readability. Use JSONL when exact structure matters.

## `artifact.md`

A human-readable view grouped by the requested `group_by` value. It includes content,
selection reasons, and source reference for each selected record.

## `manifest.json`

Producer-local operational evidence containing the selection request, input file
checksums, counts, generation time, and output names. It is not a shared artifact
contract. See [Provenance](provenance.md) and
[Interoperability](interoperability.md).

## Atomic output

Files are rendered in a private sibling staging directory, validated, hashed, and
then promoted at the directory boundary. Existing non-empty output directories and
symlinked output paths are rejected.
