"""Utilities for interacting with the spring-framework Git repository.

We keep a *bare* clone as a cache so that switching between tags is
instantaneous and bandwidth-friendly.
"""
from __future__ import annotations

import contextlib
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, List

import git  # type: ignore[import]
from rich.console import Console
from rich.progress import Progress

__all__ = [
    "ensure_local_mirror",
    "list_tags",
    "checkout_tag",
]

REPO_URL = "https://github.com/spring-projects/spring-framework.git"
console = Console()

def ensure_local_mirror(cache_dir: Path | str) -> Path:
    """Clone the remote repo if necessary and return the path.

    We use `--mirror` so that *all* refs are available locally and no working
    tree is checked out (saves space).  A temporary *worktree* will be created
    on demand for each tag checkout via :pyfunc:`checkout_tag`.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    if cache_dir.exists():
        return cache_dir

    console.print(f"Cloning spring-framework into cache [path]{cache_dir}[/]…")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    git.Repo.clone_from(REPO_URL, cache_dir, mirror=True)
    return cache_dir


def list_tags(cache_dir: Path | str, pattern: str | None = None) -> List[str]:
    """Return a list of git tags (sorted chronologically)."""
    cache_path = ensure_local_mirror(cache_dir)
    repo = git.Repo(cache_path)
    tags = [tag.name for tag in repo.tags]
    if pattern is not None:
        import fnmatch

        tags = [t for t in tags if fnmatch.fnmatch(t, pattern)]
    # sort by commit date
    tags.sort(key=lambda t: repo.commit(t).committed_datetime)
    return tags


@contextlib.contextmanager
def checkout_tag(cache_dir: Path | str, tag: str) -> Generator[Path, None, None]:
    """Context manager that checks out *tag* into a temporary worktree.

    Example:
        >>> with checkout_tag("./spring_repo", "v6.1.5") as path:
        ...     # parse Java files under *path*
    """
    cache_path = ensure_local_mirror(cache_dir)
    repo = git.Repo(cache_path)
    if tag not in repo.tags:
        raise ValueError(f"Tag {tag} not found in local mirror {cache_path}")

    # Using git-worktree is much faster than plain checkout because the mirror
    # is bare.
    with tempfile.TemporaryDirectory(prefix=f"spring-{tag}-") as tmpdir:
        tmp_path = Path(tmpdir)
        cmd = [
            "git",
            "--git-dir",
            str(cache_path),
            "worktree",
            "add",
            "--detach",
            str(tmp_path),
            tag,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            yield tmp_path
        finally:
            # Remove worktree (suppress output)
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(cache_path),
                    "worktree",
                    "remove",
                    "--force",
                    str(tmp_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) 