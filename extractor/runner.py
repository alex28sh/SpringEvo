"""Command-line interface to run the whole pipeline end-to-end."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from .api_extractor import take_snapshot
from .diff import write_diff
from .git_utils import list_tags

app = typer.Typer(add_help_option=True, rich_markup_mode="rich")
console = Console()


@app.command()
def main(
    repo_cache: Path = typer.Option(Path("./spring_repo"), help="Path to bare mirror cache."),
    out_dir: Path = typer.Option(Path("./data"), help="Where to write snapshots & diffs."),
    include_tags: List[str] = typer.Option(
        [],
        "--include-tag",
        "-i",
        help="Glob pattern. Repeat to specify multiple tags to include.",
    ),
    exclude_tags: List[str] = typer.Option(
        [],
        "--exclude-tag",
        "-e",
        help="Glob pattern. Repeat to specify multiple tags to exclude.",
    ),
    max_tags: Optional[int] = typer.Option(None, help="Process only the first N tags."),
    jobs: Optional[int] = typer.Option(None, "-j", help="Parallel parser processes (default: CPU count)."),
):
    print(f"include_tags: {include_tags}")
    """Run snapshot & diff generation for selected tags."""
    # ---------------------------------------------------------------------
    # determine tag list
    # ---------------------------------------------------------------------
    tags = list_tags(repo_cache)
    if include_tags:
        import fnmatch

        tags = [t for t in tags if any(fnmatch.fnmatch(t, pat) for pat in include_tags)]
    if exclude_tags:
        import fnmatch

        tags = [t for t in tags if not any(fnmatch.fnmatch(t, pat) for pat in exclude_tags)]

    if max_tags is not None:
        tags = tags[:max_tags]

    console.print(f"Processing tags: {', '.join(tags)}")

    # ---------------------------------------------------------------------
    # snapshots
    # ---------------------------------------------------------------------
    snap_dir = out_dir / "snapshots"
    diff_dir = out_dir / "diffs"
    prev_tag = None
    prev_snap = None
    for tag in tags:
        snap_path = snap_dir / f"{tag}.json"
        snap_path = take_snapshot(tag, repo_cache, snap_path, processes=jobs)

        # diff with previous
        if prev_tag is not None and prev_snap is not None:
            write_diff(prev_tag, tag, prev_snap, snap_path, diff_dir)
        prev_tag, prev_snap = tag, snap_path


if __name__ == "__main__":
    app() 