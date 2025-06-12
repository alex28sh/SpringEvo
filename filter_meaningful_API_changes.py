from __future__ import annotations

"""Filter API changes extracted by the *extractor* package with the help of an LLM.

This script walks over all JSON diff files (as produced by
`SpringEvo/extractor/runner.py`) and decides — *per individual change* —
whether the change is *meaningful* from the perspective of an external Spring
library consumer.

The decision is delegated to an LLM (OpenAI ChatCompletion).  Results are
persisted so that subsequent runs do *not* burn additional tokens for already
seen changes.

Example usage from the repository root:

    $ python -m SpringEvo.filter_meaningful_API_changes \\
        --diff-dir SpringEvo/data_filtered/diffs \\
        --out-dir  SpringEvo/data_filtered/diffs_meaningful

Set the environment variable ``OPENAI_API_KEY`` (or use ``--api-key``) before
execution.
"""

import json
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from openai import OpenAI

# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

app = typer.Typer(add_help_option=True, rich_markup_mode="rich")
console = Console()
max_entries = 100 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_change_hash(change_id: str, change: Dict[str, Any]) -> str:
    """Return a short but stable *hash* representing this *change*.

    This is used as cache key so that we don't call the LLM twice for an
    identical change.
    """

    blob = json.dumps({"id": change_id, "change": change}, sort_keys=True)
    return sha256(blob.encode()).hexdigest()[:32]


def _build_prompt(change_id: str, change: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return a ChatCompletion *messages* list instructing the LLM."""

    system = (
        "You are a senior Java developer familiar with the Spring Framework. "
        "You will be shown the details of a single API change that happened "
        "between two framework versions.  Your task is to decide whether this "
        "change is *meaningful* for downstream users of the public API. A "
        "meaningful change is one that could break user code, requires an "
        "adjustment, or is otherwise significant enough to be mentioned in the "
        "release notes.  Minor changes such as adding or removing the "
        "'deprecated' annotation, switching access modifiers without user-"
        "impact, or internal refactorings should be marked as *not meaningful*. "
        "Answer with a single word – either 'yes' or 'no'."
    )

    user_parts = [
        f"Change-type: {change.get('change', 'unknown')}",
        f"Identifier : {change_id}",
    ]

    # include a few interesting fields if present – keeps the prompt concise
    interesting_fields = [
        "kind",
        "signature",
        "return",
        "params",
        "modifiers",
        "deprecated",
    ]
    for field in interesting_fields:
        if field in change:
            user_parts.append(f"{field.capitalize():<10}: {change[field]}")

    user = "\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _ask_llm_cached(
    change_id: str,
    change: Dict[str, Any],
    *,
    model: str,
    api_key: str,
    # cache: Dict[str, bool],
    dry_run: bool = False,
    max_retries: int = 3,
) -> bool:
    """Return *True* if the LLM judges *change* to be meaningful.

    Results are looked-up/recorded in *cache* using a stable hash key.
    """

    key = _stable_change_hash(change_id, change)
    # if key in cache:
    #     return cache[key]

    if dry_run:
        # Simple heuristic fallback: treat removals and signature changes as meaningful
        meaningful = change.get("change") in {"removed", "signature_changed", "added"}
        # cache[key] = meaningful
        return meaningful

    messages = _build_prompt(change_id, change)
    backoff = 1.5
    # for attempt in range(1, max_retries + 1):
    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0 # + (attempt * 0.1)  # Slightly increase temperature for diversity in retries
        )

        response = completion.choices[0].message.content.strip().lower()
        console.print(response)
        # resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.0)
        # answer = resp.choices[0].message.content.strip().lower()
        meaningful = response.startswith("y")  # 'yes' → meaningful
        # cache[key] = meaningful
        return meaningful
    # except openai.error.RateLimitError:
    #     console.print(f"[yellow]Rate-limited. Retry {attempt}/{max_retries} in {backoff:.1f}s…[/]")
    #     time.sleep(backoff)
    #     backoff *= 2
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]LLM error:[/] {exc}. Marking change as not meaningful.")
        # cache[key] = False
        return False

    # console.print("[red]Exceeded retry count; giving up on this change.[/]")
    # cache[key] = False
    # return False


# ---------------------------------------------------------------------------
# CLI implementation
# ---------------------------------------------------------------------------

def process_path(path: Path, out_dir: Path, dry_run: bool, model: str, api_key: str):
    diff_data: Dict[str, Dict[str, Any]] = json.loads(path.read_text())
    filtered: Dict[str, Dict[str, Any]] = {}

    import random
    
    # print(diff_data.items()[0])
    data_shrinked = random.sample(diff_data.items(), min(max_entries, len(diff_data.items())))
    # print(data_shrinked[0])

    for change_id, change in data_shrinked:
        if _ask_llm_cached(change_id, change, model=model, api_key=api_key, dry_run=dry_run):
            filtered[change_id] = change
    out_path = out_dir / path.name
    out_path.write_text(json.dumps(filtered, indent=2, sort_keys=True))
    return filtered

@app.command()
def main(
    diff_dir: Path = typer.Option(
        Path("data_filtered/diffs"), help="Directory containing diff *.json files."
    ),
    out_dir: Path = typer.Option(
        Path("data_filtered/diffs_meaningful"), help="Where to write the filtered diffs."
    ),
    # cache_path: Path = typer.Option(
    #     Path(".llm_judgement_cache.json"), help="Persistence file for LLM answers."
    # ),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key (defaults to OPENAI_API_KEY env var)."),
    model: str = typer.Option("o1-mini"),
    max_files: Optional[int] = typer.Option(None, help="Process only the first N diff files."),
    dry_run: bool = typer.Option(False, help="Skip the LLM and use a simple heuristic (for testing)."),
):
    """Filter *diff* JSON files leaving only *meaningful* API changes."""

    # ---------------------------------------------------------------------
    # Environment setup & cache loading
    # ---------------------------------------------------------------------
    # if cache_path.exists():
    #     cache: Dict[str, bool] = json.loads(cache_path.read_text())
    # else:
    #     cache = {}

    diff_paths = sorted(diff_dir.glob("*.json"))
    if max_files is not None:
        diff_paths = diff_paths[:max_files]

    total_changes = 0
    kept_changes = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    import concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_path, path, out_dir, dry_run, model, api_key): path for path in diff_paths}
        for fut in concurrent.futures.as_completed(futures):
            path = futures[fut]
            console.print(f"[green]✓[/] {path.name}: kept {len(fut.result())} changes → {out_dir / path.name}")

    # for diff_path in diff_paths:
    #     diff_data: Dict[str, Dict[str, Any]] = json.loads(diff_path.read_text())
    #     filtered: Dict[str, Dict[str, Any]] = {}

    #     for change_id, change in diff_data.items():
    #         total_changes += 1
    #         if _ask_llm_cached(change_id, change, model=model, api_key=api_key, dry_run=dry_run):
    #             filtered[change_id] = change
    #             kept_changes += 1

    #     out_path = out_dir / diff_path.name
    #     if filtered:
    #         out_path.write_text(json.dumps(filtered, indent=2, sort_keys=True))
    #         console.print(
    #             f"[green]✓[/] {diff_path.name}: kept {len(filtered)}/{len(diff_data)} changes → {out_path}"
    #         )
    #     else:
    #         console.print(f"[blue]•[/] {diff_path.name}: no meaningful changes found – skipping output file")

    #     # Persist the cache *after every file* so that progress is not lost
    #     # cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    # console.print(
    #     f"\n[bold]Finished.[/] Meaningful changes: {kept_changes}/{total_changes} ({kept_changes/total_changes:0.1%})."
    # )


if __name__ == "__main__":
    app()
