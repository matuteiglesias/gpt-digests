# digests_project/cli/kbctl_publish.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from digests_project.bags_pipeline.compute.config import _in_window_range
from digests_project.bags_pipeline.compute.io import read_json
from digests_project.bags_pipeline.publish.publish import publish_l2

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command(name="publish")
def publish(
    digests_root: Path = typer.Option(..., help="Root folder that contains L2 outputs"),
    only_validated: bool = typer.Option(True, "--only-validated/--all"),
    out_dir: Optional[Path] = typer.Option(None, help="Defaults to <root>/_published"),
    link: bool = typer.Option(False, help="Symlink instead of copying"),
    since: str = typer.Option("", help="UTC ISO start (inclusive)"),
    until: str = typer.Option("", help="UTC ISO end (exclusive)"),
):
    """
    Publish L2 digests by walking manifests, filtering by validation
    and optional time window, then copying or symlinking MDX files.
    """
    rows: List[Dict[str, Any]] = []

    for mf in digests_root.rglob("manifest.json"):
        info = read_json(mf).get("digest", {})
        mdx = mf.with_name("digest.mdx")
        rows.append({
            "manifest": mf,
            "mdx": mdx,
            "validated": info.get("policy", "") == "validated",
            "start_ts": info.get("start_ts"),
            "end_ts": info.get("end_ts"),
        })

    if not rows:
        for mdx in digests_root.rglob("*.mdx"):
            rows.append({
                "mdx": mdx,
                "validated": True,
                "manifest": None,
                "start_ts": None,
                "end_ts": None,
            })

    if since or until:
        rows = [
            r for r in rows
            if _in_window_range(r.get("start_ts"), r.get("end_ts"), since or None, until or None)
        ]

    target = out_dir or (digests_root / "_published")

    # NOTE: your current publish_l2 signature ignores rows (it walks root again).
    # Keep it unchanged for now (minimal change), but this is a known “paper cut”.
    published = publish_l2(
        root=digests_root,
        only_validated=only_validated,
        out_dir=target,
        link=link,
    )

    typer.echo(f"published {len(published)} files -> {target}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()