cat > rebuild_views.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
BASE="."
VIEW="$BASE/_views"
mkdir -p "$VIEW"/{stages,formats,spine,data,automation,career,ai,health}
cd "$BASE"

linkall() { while read -r f; do ln -sf "../pairbag/$f" "$VIEW/$1/$f"; done; }

ls | grep -E 'stage_(plan|execute|reflect)\.mdx$'                                      | linkall stages
ls | grep -E 'format_type_(template|checklist|code|script|guide|report)'               | linkall formats
ls | grep -E 'category_programming|category_software_dev|domain_python|free_web_development' | linkall spine
ls | grep -E 'free_(data|data_processing|debugging|error_handling)\.mdx$'              | linkall data
ls | grep -E 'category_automation|free_(automation|screening|knowledge_management|knowledge_organization)' | linkall automation
ls | grep -E 'category_business\+|(free_(career_development|job_application(s)?|job_search))|format_type_template|stage_(plan|reflect)' | linkall career
ls | grep -E 'category_ai(_agents)?|free_ai'                                           | linkall ai
ls | grep -E 'category_health\+domain_healthcare'                                      | linkall health
SH
chmod +x rebuild_views.sh
