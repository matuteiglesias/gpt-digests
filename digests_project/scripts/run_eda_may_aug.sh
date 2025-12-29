#!/usr/bin/env bash
set -euo pipefail

# --- Config knobs (edit as needed) ---
ROOT="${ROOT:-$HOME/repos/GPT_digests}"          # repo root
PKG="${PKG:-$ROOT/digests_project}"              # python package folder
RUNS="${RUNS:-$ROOT/runs}"                       # runs folder
LOGS_GLOB="${LOGS_GLOB:-$PKG/data/logs/03_merged_logs/*.jsonl}"

SINCE="${SINCE:-2025-05-01T00:00:00Z}"
UNTIL="${UNTIL:-2025-09-01T00:00:00Z}"

# cohort params
GROUP_BY="${GROUP_BY:-day}"
COMBO_SIZE="${COMBO_SIZE:-2}"                    # daily bundles of 2 days
MIN_EVENTS="${MIN_EVENTS:-4}"
TOP_K_TAGS="${TOP_K_TAGS:-30}"

# pairs params
TOP_K="${TOP_K:-800}"
MIN_DOCS="${MIN_DOCS:-2}"
MIN_NPMI="${MIN_NPMI:-0.02}"

# gates policy (soft)
GATES_JSON="${GATES_JSON:-$PKG/outputs/eda_units/gates_soft.json}"

# outputs
OUT_BASE="${OUT_BASE:-$PKG/outputs/eda_units}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/may-aug_daily_combo2}"
UNITS_JSONL="${UNITS_JSONL:-$RUNS/units_cohorts_may-aug_daily_combo2.jsonl}"

# --- Ensure env is ready ---
echo ">>> Installing package (editable)…"
pip install -e "$ROOT" >/dev/null

# Optional: install kernel once
# python -m ipykernel install --user --name gpt-digests --display-name "GPT Digests"

# --- Soft gates (write once if missing) ---
mkdir -p "$OUT_BASE"
if [[ ! -f "$GATES_JSON" ]]; then
  cat > "$GATES_JSON" <<'JSON'
{
  "CO_DEFAULT": 25,
  "CO_BACKBONE": 60,
  "CO_NICHE_LO": 10,
  "CO_NICHE_HI": 40,
  "NPMI_KEEP": 0.12,
  "LIFT_KEEP": 2.30,
  "NPMI_STRONG": 0.12,
  "LIFT_STRONG": 2.28,
  "NPMI_VSTRONG": 0.167,
  "LIFT_VSTRONG": 3.037,
  "NPMI_BRIDGE": 0.18
}
JSON
  echo ">>> Wrote soft gates → $GATES_JSON"
fi

# --- Clean noisy output folders (only the target!) ---
echo ">>> Cleaning output dir → $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# --- Build cohorts (JSONL of Units) ---
echo ">>> Building cohorts → $UNITS_JSONL"
kbctl bags-logs \
  --logs-glob "$LOGS_GLOB" \
  --since "$SINCE" \
  --until "$UNTIL" \
  --group-by "$GROUP_BY" \
  --combo-size "$COMBO_SIZE" \
  --min-events "$MIN_EVENTS" \
  --top-k-tags "$TOP_K_TAGS" \
  --out "$UNITS_JSONL"

# --- Run EDA from units (pairs + communities + subsets) ---
echo ">>> EDA: pairs/communities/subsets → $OUT_DIR"
kbctl eda-tagpairs-from-units \
  --units-jsonl "$UNITS_JSONL" \
  --out-dir "$OUT_DIR" \
  --top-k "$TOP_K" \
  --min-docs "$MIN_DOCS" \
  --min-npmi "$MIN_NPMI" \
  --gates-json "$GATES_JSON"

# --- Quick sanity checks ---
echo ">>> Sanity checks:"
if command -v jq >/dev/null 2>&1; then
  echo "  index.json:" && jq . "$OUT_DIR/index.json" || true
  echo "  counts.json:" && jq . "$OUT_DIR/counts.json" || true
  echo "  gates.json:" && jq . "$OUT_DIR/gates.json" || true
  echo "  quantiles.json:" && jq . "$OUT_DIR/quantiles.json" || true
fi

echo ">>> CSV row counts:"
for f in co_tag_pairs.csv tag_communities.csv edges_default.csv edges_niche.csv edges_bridges.csv edges_audit.csv; do
  [[ -f "$OUT_DIR/$f" ]] && echo "  $(wc -l < "$OUT_DIR/$f") lines  $f" || echo "  (missing) $f"
done

echo "✅ Done → $OUT_DIR"
