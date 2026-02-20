## Runbook v1 (how you’re supposed to use this repo)

Create `ops/runbooks/runbook.md` with this structure:

### 1) Purpose

This repo turns Event Bus artifacts (merged logs + sessionized JSONL) into “digest artifacts” (units, bags, published outputs).

### 2) Inputs (explicit artifact contracts)

* **Event log bus**: JSONL events (ex: `.../03_merged_logs/YYYY-MM-DD.jsonl`)
* **Session bus**: JSONL sessions (ex: `.../15_sessions_parsed/YYYY-MM-DD.jsonl`)

### 3) Outputs (single artifact root)

Everything is written under:

* `artifacts/runs/` (units JSONL outputs, intermediate indices)
* `artifacts/index/` (window indices, registries like `l2_by_window.json`)
* `artifacts/published/` (rendered bag markdown/mdx, if used)

### 4) Typical flows (CLI)

* Build units from logs
* Build units from sessions
* Build L2 bag(s) from units
* Publish/render bag outputs

### 5) Ops

* Timers/services live in `ops/systemd/`
* Troubleshooting: “empty corpus”, “schema drift”, “tag normalization surprises”, “idempotency”

### 6) Evidence checklist

A run is “good” if it produced:

* a run JSONL in `artifacts/runs/`
* an updated index in `artifacts/index/`
* a short run summary (counts, dates, hashes)

