# Provenance

`kb-artifacts` preserves evidence about what was read and why each record was
selected. It does not infer missing upstream provenance.

## Record provenance

Exported records carry a source reference with the input partition, one-based line
number, optional upstream reference, and optional text checksum. Selection reasons
state which explicit filters matched.

## Run provenance

`manifest.json` records:

- the complete selection request;
- every matched input partition and its SHA-256;
- scanned, invalid, deduplicated, and selected counts;
- generation time and output filenames.

Paths in a local manifest describe the inputs used for that run. Review manifests
before sharing them if local path names are sensitive.

## Privacy defaults

Source inspection excludes record bodies by default and reports bounded metadata.
`--include-excerpts` is an explicit opt-in. Selection exports do contain selected
content, so treat their destination according to the source material's access rules.

## Determinism boundary

For identical inputs, request, code, and environment assumptions, selected-record
ordering and rendered evidence views are deterministic. The operational manifest's
generation timestamp changes between runs and is not part of selected-evidence
content identity.
