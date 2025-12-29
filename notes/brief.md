# brief_gpt_digests.md
---
id: 01
title: GPT Digests – Logs → MDX pipeline
heartbeat: weekly
status: in_build
last_touch: 2025-10-17
next_review: 2025-10-24
tags: [etl, summarization, mdx, chromadb]
impact: high
confidence: medium
dependencies: [SUC44, SUC40]
---

## 1. Current State
- ETL scripts (`ingest_logs`, `unitize`, `pairs`) working locally.
- Latest run produced May–Aug digests under `_published/`.
- Systemd timers exist but are not active at this path.

## 2. Friction
- Not under git → weak provenance of artifacts.
- Manual env vars; `LOGS_GLOB` can yield empty runs.
- Dependencies not pinned (`networkx`, `pyyaml`, `pandas`).

## 3. Next Unlock
→ Create `.env.sh` + `make run-window` for a 14-day test window.

## 4. Evidence / Links
- script: `scripts/run_eda_may_aug.sh`
- output: `runs/units_cohorts_may-aug_daily_combo2.jsonl`
- published: `digests/_published/2025-may-aug-all/_views/`
- docs: `docs/guide/pipelines.md`

## 5. Notes / Debris (optional)
- Future: re-enable L2/L3 timers once repo migrated to Git.



