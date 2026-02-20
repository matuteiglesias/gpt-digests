# bags_pipeline/publish.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bags_pipeline.io import (
    read_json,
    read_mdx_front_matter,
    ensure_dir,
    copy_file,
    symlink_file,
)
from bags_pipeline.config import TZ_LOCAL, _in_window_range, parse_utc_any

Pathish = Union[str, Path]

def _locate_mdx(manifest_path: Path, digest_dict: Dict[str, Any]) -> Optional[Path]:
    # 1) tree layout: digest.mdx next to manifest
    tree = manifest_path.parent / "digest.mdx"
    if tree.exists():
        return tree

    # 2) flat: {id}.mdx sibling
    did = digest_dict.get("id")
    if did:
        flat = manifest_path.parent / f"{did}.mdx"
        if flat.exists():
            return flat

    # 3) any first .mdx sibling
    for candidate in manifest_path.parent.glob("*.mdx"):
        return candidate

    return None




def _load_items_from_manifests(root: Path) -> List[Dict[str, Any]]:
    """
    Walk <root> for manifest.json, read each, locate its .mdx, and collect:
      { mdx:Path, manifest:Path, validated:bool, start_ts:str, end_ts:str }
    If no manifests found, fall back to every standalone .mdx (always valid).
    """
    items: List[Dict[str, Any]] = []

    for mf in root.rglob("manifest.json"):
        try:
            j = read_json(mf)
            d = j.get("digest", j)
            valid = bool(d.get("validated")) \
                    or (d.get("validation", {}).get("status") in {"ok", "valid", "passed"})
            mdx = _locate_mdx(mf, d)
            if not mdx or not mdx.exists():
                continue

            items.append({
                "mdx":       mdx,
                "manifest":  mf,
                "validated": valid,
                "start_ts":  d.get("start_ts"),
                "end_ts":    d.get("end_ts"),
            })
        except Exception:
            continue

    if not items:
        # fallback: any .mdx under root
        for mdx in root.rglob("*.mdx"):
            fm = read_mdx_front_matter(mdx) or {}
            items.append({
                "mdx":       mdx,
                "manifest":  None,
                "validated": True,
                "start_ts":  fm.get("start_ts"),
                "end_ts":    fm.get("end_ts"),
            })

    return items


def publish_l2(
        root: Pathish,
        *,
        only_validated: bool   = False,
        out_dir: Optional[Pathish] = None,
        since:   Optional[str] = None,
        until:   Optional[str] = None,
        link:    bool          = False,
    ) -> List[Path]:
    """
    Publish L2 digests by copying (or symlinking) their .mdx into <out_dir>,
    preserving the directory structure under <root>.  Optionally filter by:
      • only_validated
      • [since, until) time window.

    Returns the list of published destination Paths.
    """
    root = Path(root)
    target = Path(out_dir) if out_dir else (root / "_published")
    ensure_dir(target)

    # 1) gather all candidate items
    items = _load_items_from_manifests(root)

    # 2) apply validation filter
    if only_validated:
        items = [it for it in items if it["validated"]]

    # 3) apply time-window filter
    if since or until:
        items = [
            it for it in items
            if _in_window_range(
                it.get("start_ts"), it.get("end_ts"),
                slice_start=since, slice_end=until,
                tz=TZ_LOCAL, parse_fn=parse_utc_any
            )
        ]

    # 4) copy or link into place
    published: List[Path] = []
    for it in items:
        src = it["mdx"]
        try:
            rel = src.relative_to(root)
        except ValueError:
            rel = Path(src.name)

        dst = target / rel
        ensure_dir(dst.parent)

        if link:
            symlink_file(src, dst)
        else:
            copy_file(src, dst)

        published.append(dst)

    return published

# def _is_relative_to(p: Path, base: Path) -> bool:
#     try:
#         p.relative_to(base)
#         return True
#     except ValueError:
#         return False

# def _normalize_items(
#     digests_root: Path, 
#     items: Optional[Iterable[Union[Pathish, Dict[str, Any]]]]
# ) -> List[Dict[str, Any]]:
#     """
#     Normalize 'items' to a list of dicts with keys:
#       - mdx: Path (required)
#       - manifest: Optional[Path]
#       - validated: Optional[bool]
#     """
#     out: List[Dict[str, Any]] = []
#     if not items:
#         # Back-compat: walk manifests
#         for mf in digests_root.rglob("manifest.json"):
#             m = _read_manifest(mf)
#             mdx = _manifest_to_mdx_path(mf, m)
#             if mdx:
#                 out.append({"mdx": mdx, "manifest": mf, "validated": _manifest_validated(m)})
#         # If nothing found (e.g., onefile layout), fall back to mdx scan
#         if not out:
#             for mdx in digests_root.rglob("*.mdx"):
#                 out.append({"mdx": mdx})
#         return out

#     # Explicit items provided
#     for it in items:
#         if isinstance(it, (str, Path)):
#             out.append({"mdx": Path(it)})
#         elif isinstance(it, dict):
#             mdx = it.get("mdx") or it.get("path") or it.get("file")
#             if not mdx:
#                 continue
#             row = {"mdx": Path(mdx)}
#             if "manifest" in it and it["manifest"]:
#                 row["manifest"] = Path(it["manifest"])
#             if "validated" in it:
#                 row["validated"] = bool(it["validated"])
#             out.append(row)
#     return out

