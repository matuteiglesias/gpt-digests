#!/bin/sh
# POSIX-compatible setup script — Jan–Aug 2025, flat buckets with month-prefixed filenames
set -eu

# Base paths (override BASE if needed)
BASE="${BASE:-$HOME/repos/GPT_digests}"
PUB="$BASE/digests/_published"

date_tag=$(date +%Y%m%d_%H%M)
OUT="$HOME/Downloads/digests_tmp_$date_tag"
echo "Creating output directory: $OUT"

# Months Jan–Aug 2025
target_months="2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08"
months_regex='2025-0[1-8]'

# 0) Buckets: explicit mkdir -p
echo "[0] Creating bucket directories..."
mkdir -p "$OUT/01_arcs/common"
mkdir -p "$OUT/01_arcs/free_promptflow"
mkdir -p "$OUT/01_arcs/msg_reflect"
mkdir -p "$OUT/01_arcs/auto_exec"
for fam in domain_python category_programming free_automation; do
  mkdir -p "$OUT/02_instr_exec/$fam/instruction"
  mkdir -p "$OUT/02_instr_exec/$fam/execute"
  mkdir -p "$OUT/02_instr_exec/$fam/plan"
done
mkdir -p "$OUT/03_promptflow"
mkdir -p "$OUT/04_reflection/strict"
mkdir -p "$OUT/04_reflection/cotags"
mkdir -p "$OUT/05_cookbook/tagbag"
mkdir -p "$OUT/05_cookbook/all"
echo "Buckets created."

# Helper: link files into a bucket, prefixing month in filename
do_link() {
  dest="$1"; shift
  for src in "$@"; do
    [ -f "$src" ] || continue
    month=$(printf "%s" "$src" \
      | sed -n 's#.*/_published/\(2025-0[1-8]\)/.*#\1#p')
    base=$(basename "$src")
    ln -sf "$src" "$dest/${month}__${base}"
  done
}

# 1) ARC HUNTING: filenames common across all existing months
echo "[1] Arc hunting: common files"
# pick first existing month list
first_list=""
for m in $target_months; do
  [ -d "$PUB/$m/L2" ] || continue
  find "$PUB/$m/L2" -type f -name '*.mdx' -printf '%f\n' | sort -u > /tmp/mdx_first
  first_list=/tmp/mdx_first
  break
done
# intersect with each month
for m in $target_months; do
  [ -d "$PUB/$m/L2" ] || continue
  find "$PUB/$m/L2" -type f -name '*.mdx' -printf '%f\n' | sort -u > /tmp/mdx_cur
  comm -12 "$first_list" /tmp/mdx_cur > /tmp/mdx_new
  mv /tmp/mdx_new "$first_list"
done
# link common across months
while IFS= read -r file; do
  for m in $target_months; do
    src="$PUB/$m/L2/$file"
    [ -f "$src" ] && do_link "$OUT/01_arcs/common" "$src"
  done
done < "$first_list"
rm -f "$first_list"

# 2) Targeted arcs
echo "[2] Targeted arcs"
do_link "$OUT/01_arcs/free_promptflow" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/tagbag/memo/tagbag__free_promptflow\.mdx" 2>/dev/null)
do_link "$OUT/01_arcs/msg_reflect" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/pairbag/memo/pairbag__msg_type_reflection\+stage_reflect\.mdx" 2>/dev/null)
do_link "$OUT/01_arcs/auto_exec" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/pairbag/memo/pairbag__free_automation\+stage_execute\.mdx" 2>/dev/null)

# 3) Instruction → Execution audit
echo "[3] Instruction → Execution audit"
for fam in domain_python category_programming free_automation; do
  set +u  # allow empty
  matches=$(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/pairbag/memo/pairbag__.*${fam}.*(msg_type_instruction|stage_execute|stage_plan).*\.mdx" 2>/dev/null)
  set -u
  for src in $matches; do
    case "$src" in
      *msg_type_instruction*) sub=instruction;;
      *stage_execute*)        sub=execute;;
      *stage_plan*)           sub=plan;;
      *) continue;;
    esac
    do_link "$OUT/02_instr_exec/$fam/$sub" "$src"
  done
done

# 4) PromptFlow backbone
echo "[4] PromptFlow backbone"
do_link "$OUT/03_promptflow" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/tagbag/memo/tagbag__free_promptflow\.mdx" 2>/dev/null)

# 5) Reflection distillation
echo "[5] Reflection distillation"
do_link "$OUT/04_reflection/strict" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/pairbag/memo/pairbag__msg_type_reflection\+stage_reflect\.mdx" 2>/dev/null)
do_link "$OUT/04_reflection/cotags" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/pairbag/memo/pairbag__.*(category_business|format_type_report|note_type_insight).*msg_type_reflection.*\.mdx" 2>/dev/null)

# 6) Cookbook extraction
echo "[6] Cookbook extraction"
do_link "$OUT/05_cookbook/tagbag" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/tagbag/memo/tagbag__category_cooking\.mdx" 2>/dev/null)
do_link "$OUT/05_cookbook/all" \
  $(find "$PUB" -regextype posix-extended \
    -regex ".*/$months_regex/L2/(pairbag|tagbag)/memo/.*category_cooking.*\.mdx" 2>/dev/null)

echo "Done. Browse: $OUT"
