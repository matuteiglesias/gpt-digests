# clean_empty_events.py
import json, sys

MIN_TOKENS = 12  # tune
def keep(ev):
    c = ev.get("content") or ""
    c = c.strip()
    if not c: return False
    if len(c.split()) < MIN_TOKENS: return False
    return True

with open(sys.argv[1], "r", encoding="utf-8") as fin:
    for line in fin:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if keep(ev):
            sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
