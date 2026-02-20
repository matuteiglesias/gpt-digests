---
id: 01
title: GPT Digests – Windowed EDA Run & Publish
last_verified: 2025-10-17
next_check: 2025-11-17
maintainer: matias
env: /home/matias/Documents/GPT_digests
dependencies: [python3.10, pandas, numpy, networkx, pyyaml]
risk_level: medium
---

## 1. Purpose
Generate publishable MDX digests from chat/session logs through a local ETL (ingest → unitize → pair → publish).

## 2. Core Operations
### A. Start or Rebuild
1. Ensure required packages installed: `pip install -r requirements.txt` (or manually install deps).
2. Source environment: `. ./.env.sh`
3. Run pipeline: `bash scripts/run_eda_may_aug.sh`
4. Build views: `bash rebuild_views.sh`
5. Verify outputs exist:
   ```bash
   ls runs/units_cohorts_*.jsonl | tail
   ls digests/_published/2025-may-aug-all/_views/ | head
````

### B. Troubleshoot

| Symptom               | Probable cause                       | Fix                                    |
| --------------------- | ------------------------------------ | -------------------------------------- |
| No rows in units_*    | Empty LOGS_GLOB                      | Check path in `.env.sh`                |
| Missing co-tag pairs  | Missing `networkx` or pandas error   | Reinstall deps                         |
| Views empty           | Regex mismatch in `rebuild_views.sh` | Edit script pattern                    |
| Systemd timer ignored | Wrong root path                      | Update `WorkingDirectory` in unit file |

### C. Deploy or Publish

1. Confirm `_views/` contains fresh MDX files.
2. Copy or sync to documentation site (manual).
3. Optionally enable systemd timers when repo migrated.

## 3. Parameters / Config

* `.env.sh` defines:

  * `ROOT`
  * `LOGS_GLOB`
  * `SINCE`, `UNTIL`
* YAML configs: `selectors.yaml`, `l2_registry.yaml`
* Data dirs: `digests_project/data/logs/`, `digests_project/data/sessions/`

## 4. Recovery Protocol

1. Delete partial artifacts in `runs/` and `_published/`.
2. Rerun steps A1–A4.
3. If hash mismatches persist, rebuild canonical tag registry.

## 5. Verification Checklist

* [ ] JSONL outputs > 0 KB
* [ ] ≥ 1 co_tag_pairs.csv produced
* [ ] `_views/` non-empty
* [ ] `.env.sh` committed
* [ ] Deps listed in README or pyproject

## 6. Notes

* Next evolution: integrate into Control Tower scheduler.
* Consider pushing outputs to Git-tracked repo for versioned diffs.

