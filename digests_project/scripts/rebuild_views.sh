#!/usr/bin/env bash
# Rebuild view folders as COPIES (or hard links) from pairbag/*.mdx
# Usage:
#   ./rebuild_views.sh            # copies (default)
#   ./rebuild_views.sh --hardlink # use hard links instead of copies

set -euo pipefail

MODE="copy"  # or "hardlink"
if [[ "${1:-}" == "--hardlink" ]]; then MODE="hardlink"; fi

SOURCE="pairbag"
VIEW="_views"
mkdir -p "$VIEW"/{stages,formats,spine,data,automation,career,ai,health}

copy_into() {
  local bucket="$1"; shift
  # Clean bucket to avoid stale files, then repopulate
  rm -f "$VIEW/$bucket/"*.mdx 2>/dev/null || true
  # Read filenames from stdin and copy/link them into the bucket
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if [[ "$MODE" == "hardlink" ]]; then
      ln -f "$SOURCE/$f" "$VIEW/$bucket/$f"
    else
      cp -f "$SOURCE/$f" "$VIEW/$bucket/$f"
    fi
  done
}

# Helper to list files matching a regex against filenames
match() { ls "$SOURCE" | grep -E "$1" || true; }

# 1) Stages
match 'stage_(plan|execute|reflect)\.mdx$'                  | copy_into stages

# 2) Formats
match 'format_type_(template|checklist|code|script|guide|report)' | copy_into formats

# 3) Programming spine
match 'category_programming|category_software_dev|domain_python|free_web_development' | copy_into spine

# 4) Data work
match 'free_(data|data_processing|debugging|error_handling)\.mdx$' | copy_into data

# 5) Automation cluster
match 'category_automation|free_(automation|screening|knowledge_management|knowledge_organization)' | copy_into automation

# 6) Career
match 'category_business\+|(free_(career_development|job_application(s)?|job_search))|format_type_template|stage_(plan|reflect)' | copy_into career

# 7) AI / Agents
match 'category_ai(_agents)?|free_ai'                       | copy_into ai

# 8) Health island
match 'category_health\+domain_healthcare'                  | copy_into health

echo "Views rebuilt in $VIEW/ using MODE=$MODE"
