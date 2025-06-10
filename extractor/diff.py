"""Diff two API snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from rich.console import Console

console = Console()

def _load_snapshot(path: Path | str) -> Dict[str, Dict]:
    return json.loads(Path(path).read_text())


def diff_snapshots(old_path: Path | str, new_path: Path | str) -> Dict[str, Dict]:
    """Return a structured diff between *old* and *new* snapshots."""
    old = _load_snapshot(old_path)
    new = _load_snapshot(new_path)

    diff: Dict[str, Dict] = {}

    # additions & modifications
    for key, new_item in new.items():
        if key not in old:
            diff[key] = {"change": "added", **new_item}
        else:
            old_item = old[key]
            changes = {}
            # deprecated/undeprecated
            if old_item.get("deprecated") and not new_item.get("deprecated"):
                changes["change"] = "undeprecated"
            elif (not old_item.get("deprecated")) and new_item.get("deprecated"):
                changes["change"] = "deprecated"

            # signature changed? (method/constructor)
            sig_fields = {"return", "params", "type"}
            for field in sig_fields:
                if field in old_item or field in new_item:
                    if old_item.get(field) != new_item.get(field):
                        changes["change"] = "signature_changed"
                        break

            # modifiers change
            if old_item.get("modifiers") != new_item.get("modifiers"):
                changes["change"] = changes.get("change", "modifier_changed")

            if changes:
                diff[key] = {**changes, **new_item}

    # removals
    for key, old_item in old.items():
        if key not in new:
            diff[key] = {"change": "removed", **old_item}

    return diff


def write_diff(old_tag: str, new_tag: str, old_snap: Path | str, new_snap: Path | str, out_dir: Path | str) -> Path:
    """Compute and store diff JSON.  Returns path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{old_tag}__{new_tag}.json"
    if out_path.exists():
        console.print(f"[green]✓[/] Diff {out_path.name} already exists")
        return out_path
    result = diff_snapshots(old_snap, new_snap)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    console.print(f"[blue]↺[/] Diff written to {out_path} (changes: {len(result)})")
    return out_path 