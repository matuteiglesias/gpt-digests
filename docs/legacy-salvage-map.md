# Legacy Salvage Map

`kb_artifacts` is the canonical, read-only artifact compiler. This map records
the evidence needed to prune `digests_project` only after the stated gates are
met. It is not an endorsement of the legacy orchestration.

Unless otherwise stated, abbreviated `compute/...` paths in the table are
relative to `src/digests_project/bags_pipeline/`.

| Capability | Exact legacy path/symbol | What it actually does | Known caller | Evidence it worked | Useful for new mission? | Decision | New target or deletion gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tag canonicalization | `src/digests_project/bags_pipeline/compute/normalize.py:297-310,369-452` | Parses list/string tags, accent-folds through `slug_value`, namespaces, and deduplicates. | EDA bridge at `eda_bridge.py:18-38`. | Imported by active EDA code. | Yes: bus inventory and recipe matching need it. | `PORT_NOW` | Ported as `kb_artifacts.normalization`; delete legacy only after a second recipe uses it. |
| JSONL reading | `compute/io.py:35-65,104-126` | Permissively reads JSONL and skips malformed fragments. | Log/session ingestion. | Used by `ingest_logs.py:247-258` and `ingest_sessions.py:122-127`. | Yes, but silence is unsafe. | `REIMPLEMENT_NOW` | New adapter records structured diagnostics at `kb_artifacts/sources/jsonl_bus.py`. |
| JSON/CSV/build-tree writing | `compute/io.py:67-99,150-155,230-251` | Atomic JSONL, JSON, CSV, and artifact-tree helpers. | Compute CLI and L2. | Direct calls in `kbctl_compute.py:198-215,361-412`. | JSON only now; CSV later only if review needs it. | `REFERENCE_ONLY` | Revisit after manual SOP review; no port now. |
| Event normalization | `compute/ingest_logs.py:57-187` | Guesses bus shape and emits lossy `Event`. | `bags-logs`. | CLI imports/calls it at `kbctl_compute.py:15-19,93`. | Stable projections are needed, but lossy model is not. | `REIMPLEMENT_NOW` | `EvidenceRecord` adapter; legacy `Event` stays isolated. |
| Event index | `compute/index.py:44-102` | Builds lookup by event ID with first-wins aliases. | Historical hydration/L2. | Called by `l2.py:321-345`. | Potentially, after real evidence proves context is needed. | `KEEP_TEMPORARILY` | Extract only a bounded conversation resolver after SOP review. |
| Session index | `compute/index.py:105-160` | Attempts a session lookup index. | Historical hydration/L2. | Call signature disagrees with `normalize_session_line`; no fixture evidence. | Not now. | `DELETE_AFTER_GATE` | Delete after legacy external-reference audit and no session recipe need. |
| Source hydration | `compute/hydrate.py:125-167,208-257` | Resolves source tuples, suppresses duplicate sources, silently skips misses. | L2 build. | Called by `l2.py:337-346`. | Ideas useful; implementation is not safe enough. | `KEEP_TEMPORARILY` | Future bounded context expansion must expose unresolved counts. |
| Markdown snippet formatting | `compute/hydrate.py:263-403` | Escapes/snippets events and wraps sources in Markdown details blocks. | `bag-md`, L2. | `kbctl_compute.py:234-237` calls incompatible signature. | Plain Markdown is needed, legacy API is broken. | `REIMPLEMENT_NOW` | `kb_artifacts/renderers/markdown.py`; retain source formatting only as reference. |
| Tag tally / EDA long form | `compute/eda_bridge.py:18-89` | Builds normalized document-tag table and pair analyses. | EDA CLI. | Called by `kbctl_compute.py:331-413`. | Field/tag frequencies useful; pandas graph analysis is not required now. | `REIMPLEMENT_NOW` | `kb_artifacts.inspection` provides bounded inventories and collision counts. |
| Tag-pair / graph EDA | `compute/pairs.py:23-175,182-575` | Computes NPMI/lift communities and gated edge sets. | EDA bridge and pair bags. | Legacy CLI exposes EDA command. | Maybe later for topic review, not SOP v1. | `REFERENCE_ONLY` | Re-evaluate after two recipes and a measured grouping requirement. |
| Time/session unitization | `compute/unitize.py:31-145,167-229,306-462` | Aggregates events/sessions into day/week/month/session `Unit`s. | bags commands. | `kbctl_compute.py:102,121-129`. | Post-selection grouping may need pieces; not central input model. | `KEEP_TEMPORARILY` | Extract a selected-evidence grouping function only if real use requires it. |
| Topic / tag-pair grouping | `compute/tag_select.py:379-409` | Groups `Unit` records by day/topic/tag pair. | No direct current CLI caller found. | Pure helpers but tied to `Unit`. | Potentially later. | `REFERENCE_ONLY` | Reimplement over selected evidence if required. |
| Unit selection | `compute/tag_select.py:110-180` | Exact unit type/tag/window predicate. | `units-select`. | `kbctl_compute.py:180-200`. | Wrong abstraction for record evidence mining. | `DELETE_AFTER_GATE` | Remove after legacy CLI has no external callers. |
| JSONL deduplication | `compute/dedupe.py:8-34` | Drops duplicate JSONL rows by dotted key/content fallback. | No direct call found; CLI duplicates logic. | Exists as a standalone pure function. | Concept useful; provenance-aware policy is required. | `REIMPLEMENT_NOW` | New engine dedupes before evaluation with winner linkage. |
| L2 model/render/write | `compute/l2.py:54-479` | Historical digest IDs, manifests, MDX layouts, and hydration. | Systemd/docs reference old L2 commands. | Conflicting `L2Digest` definitions and absent fields prevent trust. | No. | `DELETE_AFTER_GATE` | Delete only after external automation inventory and replacement decision. |
| MDX publication | `publish/publish.py:18-141` | Walks manifests or falls back to all MDX, copies/symlinks. | `kbctl_publish.py:15-68`. | Broad exception swallowing and old import paths. | Not for first artifact compiler. | `DELETE_AFTER_GATE` | Rebuild as a renderer only if publication has a confirmed owner. |
| Manifest handling | `compute/l2.py:132-215`; `publish/publish.py:40-80` | Reads varied historical manifest shapes. | L2 index/publish. | Tree/flat/onefile fallback code exists. | New manifest is intentionally smaller and stricter. | `REFERENCE_ONLY` | Keep only as migration reference. |
| Mutable caches | `compute/index.py:166-183` | Reuses `cache/event_index.json` or session equivalent regardless of inputs. | No direct caller found. | Code exists, no tests. | No. | `DELETE_AFTER_GATE` | Never import into canonical spine. |
| Legacy CLI orchestration | `cli/kbctl.py:6-17`; `cli/kbctl_compute.py:64-413`; `cli/kbctl_publish.py:15-68` | Mounts compute/publish commands around Units/L2. | `pyproject.toml`, docs, Make. | Current command tree is callable but docs are stale. | No. | `UNKNOWN_EXTERNAL_USAGE` | Inventory user services before deletion. |
| Textflow integration | `scripts/bridge_textflow_to_units.py:5-24` | Reads a local SQLite database at a hardcoded home path. | `Makefile:40-42`. | Script is present; no current fixture or data. | No. | `UNKNOWN_EXTERNAL_USAGE` | Confirm local users, then delete or relocate as separate tool. |
| Make automation | `Makefile:7-54` | Invokes old `cli.kbctl`, Textflow, EDA, and an intentionally failing smoke target. | Manual make users. | References command names absent from current root CLI. | No. | `UNKNOWN_EXTERNAL_USAGE` | Replace only after external reference inventory. |
| Systemd automation | `ops/systemd/systemd/kb-l2.service:1-10`; `kb-l3.service:1-10` | Runs old L2/index commands with hardcoded paths. | User-level systemd, unverified. | Runbook calls timers inactive at this path (`ops/runbooks/brief.md:15-23`). | No. | `UNKNOWN_EXTERNAL_USAGE` | Audit user services before deletion. |
| Static MDX examples | `examples/tagbags/*.mdx` | Illustrate prior tagbag output formatting. | Documentation/examples only. | Checked-in static files. | Useful visual reference, not runtime behavior. | `REFERENCE_ONLY` | Retain until a new renderer has reviewed examples. |
| Historical notebooks/archive | `bags_pipeline/_archive/*` | Old EDA, L2/L3 registry, and notebooks. | Unknown. | Explicit archive directory. | No runtime value. | `DELETE_AFTER_GATE` | Tag baseline and delete after reference inventory. |

## Canonical accounting convention

For a build, **scanned** means every nonblank input row attempted. It equals
`selected + rejected + deduplicated + invalid_or_unusable`; ordinary nonmatches
are represented as `rejected` decisions rather than copied source records.
`decisions.jsonl` includes every selected, rejected, and deduplicated record;
`errors.jsonl` includes invalid/unusable rows only when any exist.

## Installation

The canonical package uses standard setuptools package discovery and the only
supported installation command is:

```bash
python -m pip install -e . --no-build-isolation
```

This needs `setuptools` to already be installed in the active environment. The
current audit container has neither `setuptools` nor any other build backend,
so it cannot perform an offline editable install; this is an environment
limitation rather than a reason to keep Hatchling.
