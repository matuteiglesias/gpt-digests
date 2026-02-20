# bridge_textflow_to_units.py
import sqlite3, pandas as pd, json
from pathlib import Path

def export_textflow_nodes(db_path="~/repos/textflow/embeds.sqlite",
                          out_path="runs/units_textflow.jsonl"):
    con = sqlite3.connect(Path(db_path).expanduser())
    df = pd.read_sql("SELECT id, text, ts_ms, header_path, fname FROM nodes", con)
    units = []
    for _, r in df.iterrows():
        units.append({
            "unit_id": r["id"],
            "unit_type": "textflow_node",
            "start_ts": r["ts_ms"],
            "end_ts": r["ts_ms"],
            "content": {"text": r["text"]},
            "tags": [r["header_path"]],
            "topic_ids": [],
        })
    Path(out_path).write_text("\n".join(json.dumps(u) for u in units))
    print(f"[OK] Exported {len(units)} units → {out_path}")

if __name__ == "__main__":
    export_textflow_nodes()

