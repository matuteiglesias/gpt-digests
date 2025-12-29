# bags_pipeline/validate.py
from __future__ import annotations
from typing import Iterator, Tuple, List
from pathlib import Path
import json

try:
    from jsonschema import Draft202012Validator
except Exception as e:
    raise RuntimeError("jsonschema is required: pip install jsonschema") from e

def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def iter_validate(docs: Iterator[dict], schema: dict) -> Iterator[Tuple[dict, List[str]]]:
    """
    Yields (doc, errors). errors is [] when valid; otherwise list of "path: message".
    """
    validator = Draft202012Validator(schema)
    for d in docs:
        errs = [f"{e.json_path}: {e.message}" for e in validator.iter_errors(d)]
        yield d, errs


# - return json.loads(path.read_text())
# + from bags_pipeline.io import read_json
# + return read_json(path)
