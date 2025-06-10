"""Snapshotting the API of a given git *tag* into a JSON file."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .git_utils import checkout_tag
from .java_parser_utils import parse_java_file

console = Console()

__all__ = ["take_snapshot"]


def _collect_java_files(src_root: Path) -> list[Path]:
    return [p for p in src_root.rglob("*.java") if "src/test" not in str(p)]


def _parse_one(path: Path) -> Dict[str, Dict]:  # pragma: no cover
    try:
        return parse_java_file(path)
    except Exception as exc:  # noqa: BLE001
        console.log(f"[red]Failed[/] to parse {path}: {exc}")
        return {}


def take_snapshot(
    tag: str,
    cache_dir: Path | str,
    out_path: Path | str,
    processes: int | None = None,
) -> Path:
    """Extract the API for *tag* and write it to *out_path* (JSON).

    The function is idempotent and will skip work if the file already exists.
    Returns the path to the written snapshot.
    """
    out_path = Path(out_path)
    if out_path.exists():
        console.print(f"[green]✓[/] Snapshot for {tag} already exists -> {out_path}")
        return out_path

    with checkout_tag(cache_dir, tag) as worktree:
        java_files = _collect_java_files(worktree)
        console.print(f"Extracting API for [b]{tag}[/] ({len(java_files)} .java files)…")
        with Progress(
            SpinnerColumn(),
            "{task.description}",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing", total=len(java_files))
            result: Dict[str, Dict] = {}
            with mp.Pool(processes=processes or mp.cpu_count()) as pool:
                for api_items in pool.imap_unordered(_parse_one, java_files, chunksize=20):
                    result.update(api_items)
                    progress.advance(task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    console.print(f"[green]✔[/] Snapshot for [b]{tag}[/] written to {out_path} (items: {len(result)})")
    return out_path 