"""Single source of truth for the ROCKETSHIP Python version and build identity."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Dict, Optional

__version__ = "2.0.rc"


@lru_cache(maxsize=1)
def git_revision() -> Optional[str]:
    """Short commit hash of the checkout this file lives in, `-dirty` if edited.

    Returns None when the source is not a git checkout. That is the normal case for a
    release download, not an error: `__version__` above is the released identity, and this
    only adds *which* commit a development run came from -- which is the thing you cannot
    recover afterwards when a release candidate is being tested against real data.
    """
    repo = Path(__file__).resolve().parent.parent
    head = _git(repo, "rev-parse", "--short", "HEAD")
    if not head:
        return None
    # Untracked files count as dirty: an uncommitted new module changes what ran just as
    # much as an edit does. The repository's own outputs (out/, RUNNER_DATA/) are ignored,
    # so a run does not dirty the tree merely by writing its results.
    dirty = _git(repo, "status", "--porcelain")
    return f"{head}-dirty" if dirty else head


def _git(repo: Path, *args: str) -> Optional[str]:
    """Run one git command in `repo`, or None if git is absent, fails or hangs."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def build_identity() -> Dict[str, Optional[str]]:
    """Version and revision, as recorded in run summaries and the event stream.

    One shape for every caller, so a summary JSON and a rendered header cannot disagree
    about what produced a run.
    """
    return {"version": __version__, "git_revision": git_revision()}
