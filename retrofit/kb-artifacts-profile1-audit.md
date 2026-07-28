# KB Artifacts Profile 1 audit

Status: frozen current-state evidence packet

Repository: `matuteiglesias/gpt-digests`

Package: `kb-artifacts`

Audit baseline: `228b2287d6c4171b8c2298591df1f2bd3bc49da1`

Audit date: 2026-07-28

## Decision summary

KB Artifacts is ready to be the first producer used to prove a shared-contract
release candidate, subject to that release being decided and published outside
this repository. Its selection pipeline is small, deterministic, read-only at
the source boundary, and covered by a passing test and CLI smoke baseline.

The retrofit must preserve selection semantics. The missing capability is not
evidence selection; it is the contractual run lifecycle around selection:

- stable run identity;
- initial and final run records;
- an artifact manifest and a run-bundle manifest;
- structured failures;
- staging and atomic promotion;
- binding to a shared-contract release; and
- output checksums.

This packet records the current state and identifies implementation seams. It
does not define shared schema fields and makes no production-code change.

## Gate 0B: local write safety

The audit environment reported the following before editing:

| Check | Observed value |
|---|---|
| `pwd` | `/workspace/gpt-digests` |
| repository | `gpt-digests` |
| branch | `work` |
| `HEAD` | `228b2287d6c4171b8c2298591df1f2bd3bc49da1` |
| remote | none configured in this checkout |
| working tree | clean (`## work`) |
| `AGENTS.md` in scope | none |
| `retrofit/kb-artifacts-profile1` exists | no |
| `retrofit/kb-artifacts-profile1-audit` exists | no |

`LOCAL WRITE GATE: PASS`

- Working tree was clean.
- The task environment's assigned branch is `work`; the requested retrofit
  branch names do not exist locally or as visible remote-tracking refs.
- `HEAD` is the locally available and tested baseline.
- Write authority is restricted to this repository.
- The absence of the other retrofit repositories from this task environment is
  a scope boundary, not a local-write blocker.

The lack of a configured remote prevents this task from independently proving a
fresh remote default-branch baseline. That is a remote-baseline limitation, not
a failure of the local write gate.

## Repository shape and public surfaces

The project is a Python 3.10+ package using a `src/` layout. Its canonical
console entry point is:

```text
kb-artifact = kb_artifacts.cli:main
```

The documented commands are:

```text
kb-artifact inspect source
kb-artifact select
```

The public or compatibility-sensitive surfaces are:

- `pyproject.toml`: distribution metadata and console entry point;
- `src/kb_artifacts/__init__.py`: exported Python symbols;
- `src/kb_artifacts/cli.py`: command names, arguments, exit behavior, and user
  messages;
- `src/kb_artifacts/contracts.py`: local Python data contracts;
- `src/kb_artifacts/sources/jsonl_bus.py`: accepted source representations,
  normalization, and stable-ID precedence;
- `src/kb_artifacts/selection.py`: selection semantics and output shapes;
- `src/kb_artifacts/inspection.py`: bounded inspection report shape;
- `README.md` and `docs/index.md`: operator-facing promises; and
- the generated `selected.jsonl`, `selected.csv`, `artifact.md`, and
  `manifest.json` files.

A filename-regex inventory alone is insufficient: it can find files whose names
contain `contract`, `manifest`, or `artifact`, but can miss `pyproject.toml` even
though it defines the public CLI.

## Current lifecycle

### Inspection path

```text
CLI inspect source
  -> expand_globs
  -> bound files independently for chunk and summary sources
  -> scan_jsonl
  -> normalize_record
  -> aggregate schema, missingness, value, tag, identity and diagnostic data
  -> fingerprint input partitions
  -> write one JSON source report
```

`inspect_source()` applies `max_files_per_kind` independently to chunk and
summary inputs and applies `max_records` across the resulting scan. Source text
and summary excerpts are absent unless `include_excerpts` is explicitly enabled.

### Selection path

```text
CLI select
  -> construct SelectionRequest
  -> expand explicit chunk and summary globs
  -> scan_jsonl
  -> normalize_record
  -> deterministic sort
  -> optional deduplication
  -> filter and optional classification
  -> enforce explicit empty-result policy
  -> create output directory
  -> write selected.jsonl
  -> write selected.csv
  -> write artifact.md
  -> build and write manifest.json
```

The source adapters only read input partitions. All writes are directed to the
requested output directory. A pre-existing non-empty output directory is
rejected, preventing an implicit overwrite of a previous run.

## Exact current symbols

### Local data contracts

`src/kb_artifacts/contracts.py` defines:

- `SourceReference`
  - `source_ref`
  - `partition`
  - `line_number`
  - `text_sha256`
- `EvidenceRecord`
  - `record_id`
  - `source_kind`
  - `text`
  - `summary`
  - `title`
  - `timestamp`
  - `conversation_id`
  - `message_id`
  - `tags`
  - `annotations`
  - `provenance`
  - `raw_record`
- `SelectionDecision`
  - `record_id`
  - `disposition`
  - `score`
  - `reasons`
  - `matched_values`
  - `canonical_record_id`
  - `artifact_family`
  - `artifact_maturity`
  - `classification_reasons`

These are useful producer-internal contracts. They are not evidence that a
versioned, cross-repository shared contract has been released.

### Selection request

`SelectionRequest` currently carries:

- `chunk_globs` and `summary_globs`;
- inclusive `start` and `end` dates;
- `tags`, annotation `fields`, and a `text_pattern`;
- optional artifact `families` and `maturities`;
- `limit`;
- `deduplicate`; and
- `group_by`.

### Stable-ID precedence

`_stable_id()` resolves record identity in this order:

1. upstream `provenance.source_ref`;
2. `conversation_id:message_id`;
3. `sha256:<text_sha256>`; or
4. `<source_kind>:<sha256-of-canonicalized-raw-record>`.

The final fallback serializes the raw mapping with sorted keys and compact JSON
separators before hashing. This is deterministic for an identical raw record,
but adding an otherwise irrelevant upstream field changes the fallback ID.

Contract decisions still required include namespace ownership, escaping,
algorithm versioning, the stability impact of added raw fields, and whether the
precedence itself is part of the public compatibility promise.

### Input normalization

`normalize_record()` accepts several compatible representations:

- content from `text`, `content`, or `outputs.summary_text`;
- summary from top-level `summary`, `meta.summary`, or `outputs.summary`;
- title from top-level `title` or `meta.title`;
- conversation and message identity from top-level fields or `meta`;
- tags from top-level `tags`, `meta.tags`, or `outputs.tags`; and
- timestamps from several top-level and metadata names, including epoch values
  and ISO-8601 strings.

Known metadata is projected into `EvidenceRecord`, while the entire source
mapping remains available as `raw_record`. `meta.provenance` is removed from the
annotation projection and represented by `SourceReference` instead.

Tag matching is case-, accent-, whitespace-, underscore-, and hyphen-tolerant.
Canonical display tags use `namespace:value`, defaulting to the `free`
namespace.

## Selection semantics that must not change

The retrofit must not alter these behaviors incidentally:

1. Inputs are discovered only through caller-supplied globs.
2. Paths are sorted and deduplicated.
3. Date bounds reject records with absent timestamps when a bound is active.
4. Repeated requested tags have OR semantics.
5. Distinct filter types combine with AND semantics.
6. Each requested annotation field must match after value normalization.
7. Classification runs only when family or maturity filters require it.
8. Text matching uses a case-insensitive regular expression over title, summary,
   and text.
9. Records are sorted by timestamp, record ID, partition, and line number.
10. Deduplication uses `text_sha256` when present, otherwise `record_id`.
11. The first record in deterministic order is retained.
12. `limit` is applied after filtering and deduplication.
13. Empty selection fails unless `--allow-empty` is explicit.
14. A non-empty output directory fails rather than being overwritten.

Classification precedence is also deliberate:

```text
recipe -> plan -> playbook -> operations -> template -> strategy -> reference
```

This prevents cooking procedures from becoming operations and migration plans
from becoming generic SOP output.

## Output inventory

### `selected.jsonl`

Each selected payload currently contains:

- `record_id`;
- `source_kind`;
- `title`;
- `summary`;
- `annotations`;
- `tags`;
- normalized `timestamp`;
- serialized `provenance`;
- `selection_reasons`;
- `artifact_family`; and
- `artifact_maturity`.

The full source text and `raw_record` are deliberately not included in this
JSONL payload.

### `selected.csv`

The fixed columns are:

```text
record_id
source_kind
timestamp
title
summary
tags
artifact_family
artifact_maturity
selection_reasons
text_excerpt
source_ref
```

`text_excerpt` is bounded to 320 characters.

### `artifact.md`

The Markdown artifact groups selected records by `domain` by default, with
special handling for `conversation` and `date`. Each entry includes its body or
summary, selection reasons, and source identity.

### Current `manifest.json`

The current manifest contains:

- `selection_request`
  - input globs;
  - date bounds;
  - tags;
  - fields;
  - text pattern;
  - families;
  - maturities;
  - limit;
  - deduplication flag; and
  - grouping field;
- `generated_at`;
- `matched_partitions`, with source kind, path, and SHA-256;
- `counts`
  - `scanned`;
  - `invalid`;
  - `deduplicated`; and
  - `selected`; and
- `outputs`, currently the four fixed output filenames.

The manifest is a real public producer surface but is not self-versioning. It
does not currently contain a manifest schema version, shared-contract release,
producer identity/version, producer Git revision, run ID, run state, output
hashes, structured failure inventory, compatibility declaration, staging state,
or promotion evidence.

## Failure map

### CLI validation failures

The CLI maps `SourceInputError` and relevant `ValueError` instances to a message
on stderr and exit code 2. Examples include:

- malformed `--field` values;
- an invalid date;
- `--from` later than `--to`;
- no matching input files;
- an invalid text regular expression;
- a non-empty output directory; and
- no selected records without `--allow-empty`.

These failures are human-readable but are not persisted as structured run
failures because no run lifecycle exists yet.

### Per-record source diagnostics

`scan_jsonl()` records and skips:

- `invalid_json`;
- `not_object`;
- `invalid_meta`; and
- `missing_usable_content`.

Selection includes their total only as `counts.invalid`. Inspection exposes a
breakdown by reason. Neither path currently produces a versioned structured
failure artifact tied to a stable run identity.

### Write and promotion failures

Output files are written directly into the final output directory. There is no
staging directory, atomic directory promotion, rollback marker, finalization
record, or recovery protocol. An interruption can therefore leave a partially
materialized output directory that future runs reject as non-empty.

## Retrofit seams

The following seams can be introduced without changing selection meaning:

1. Extract manifest construction from `select()` into a dedicated internal
   builder while preserving its current serialized shape initially.
2. Introduce an internal run-context provider for clock, producer identity,
   producer revision, and future shared-contract binding.
3. Isolate output rendering so bytes can be hashed after generation.
4. Render all outputs into a same-filesystem staging directory.
5. Add an atomic promotion boundary after validation and hashing.
6. Capture source diagnostics in a structured internal failure collection.
7. Add initial/final run-record lifecycle hooks around the unchanged selection
   engine.
8. Add golden fixtures for existing output shapes before evolving them.
9. Add contract conformance at the serialization boundary after the external
   release is frozen.
10. Keep shared contracts as released data/schema dependencies; do not import
    runtime code from sibling repository checkouts.

## Compatibility constraints

- Do not change CLI command names, option semantics, or exit code 2 behavior
  without an explicit compatibility decision.
- Do not change stable-ID precedence during lifecycle work.
- Do not change sorting, deduplication, filtering, classification, grouping, or
  empty-result behavior.
- Preserve the existing four outputs during the first retrofit increment.
- Treat additions to `selected.jsonl`, CSV columns, and `manifest.json` as public
  shape changes requiring fixtures and review.
- Preserve read-only source handling.
- Preserve bounded excerpts and privacy defaults.
- Do not require a runtime import from `kb-contracts`, `context`, or
  `knowledge-inspect` checkouts.
- Do not claim a shared-contract version until the shared release exists.
- Do not finalize schema field names in producer scaffolding.
- Make failed or interrupted runs distinguishable from complete runs before
  enabling atomic promotion.

## Baseline verification

The repository declares these golden-path checks:

```text
make test
make smoke
```

They resolve to:

```text
python3 -m pytest -q
PYTHONPATH=src python3 -m kb_artifacts.cli --help
```

Observed at the audit baseline:

```text
15 passed
smoke exit 0
```

The tests cover classification precedence and maturity, bounded/private source
inspection, deterministic file bounds, tag normalization, the canonical console
entry point, absence of legacy runtime imports, selection across both bus kinds,
consistent IDs across output formats, deduplication, explicit empty handling,
and optional classification filters.

## Proposed implementation sequence

This is a reviewable sequence, not authorization to implement it in this audit
commit.

1. **Audit packet** — freeze current behavior, output inventory, failures,
   compatibility constraints, and seams. No production change.
2. **Golden compatibility fixtures** — capture representative existing bytes and
   semantics for all four outputs and relevant CLI failures.
3. **Internal lifecycle seams** — extract manifest construction, rendering, and
   run context without changing public output bytes except nondeterministic time.
4. **Staging and atomic promotion** — write a complete candidate bundle, validate
   it, hash it, and atomically promote it on the same filesystem.
5. **Shared release binding** — consume the human-approved `kb-contracts`
   release, add conformance fixtures, and record the exact release identity.
6. **Versioned run lifecycle** — emit initial/final run records, artifact and
   bundle manifests, output hashes, and structured failures under the released
   schemas.
7. **Compatibility proof** — demonstrate unchanged selection results and an
   explicit migration story for any evolved public output shape.

Each step should be independently reviewable. Shared-field decisions belong to
the shared-contract release and Human Gate 1, not to opportunistic producer
implementation.

## Cross-repository session consensus

One checkout per Codex task is an expected topology. A task containing only
KB Artifacts does not block the wider retrofit; it limits that task's evidence
and write authority to KB Artifacts.

The corrected task topology is:

```text
Wave 1: Agent A (kb-contracts)
        Agent B (gpt-digests)
        Agent C1 (context)

Wave 2: Agent C2 (knowledge-inspect)
        Phase 1 consolidator
```

Every task must independently report its local write gate. Remote-baseline
resolution and local-write safety are separate gates. A missing sibling checkout
must not be misreported as local repository corruption.

For the session-level freeze tooling, blocked or missing repositories must still
appear in `baseline_records` with a null baseline, `all_blockers` must be printed
before the verdict, baseline resolution must record whether it used
`fresh_origin`, `cached_origin`, `local_fallback`, or `github_api`, and worktree
inventory must run per repository. Ownership and protection booleans must be
derived from structured state rather than hard-coded values or matching words in
human-readable errors.

## Human Gate 1 recommendation

`PASS WITH BOUNDED CORRECTIONS`

Agent B has enough evidence to prepare schema-independent lifecycle scaffolding,
but production implementation must wait for explicit scope approval. Binding to
shared fields, versions, or release identifiers must wait for the approved
shared-contract release. Selection semantics remain out of scope for change.
