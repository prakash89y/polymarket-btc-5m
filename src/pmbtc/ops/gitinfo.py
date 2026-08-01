"""Git provenance for experiment tracking.

Every trained model records the commit, branch, and tag it was produced from,
so an experiment can be traced back to the exact code that produced it. Without
this a model card says *what* was configured but not *what ran*, and the two
diverge the moment anyone edits a feature.

The working tree's cleanliness is recorded too. A model trained from a dirty
tree is not reproducible from git history — the commit does not contain the code
that ran — so it is marked, and the promotion gate can refuse it.

Everything here degrades gracefully: outside a checkout, or with git absent, the
fields read ``"unavailable"`` rather than raising. Collection must never stop
because a VCS is missing.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.ops.gitinfo")

UNAVAILABLE = "unavailable"


def _git(args: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


@dataclass(frozen=True)
class GitProvenance:
    """Where the running code came from."""

    commit: str = UNAVAILABLE
    short_commit: str = UNAVAILABLE
    branch: str = UNAVAILABLE
    tag: str = ""
    dirty: bool = False
    #: Files modified but not committed, capped for readability.
    dirty_files: tuple[str, ...] = field(default_factory=tuple)
    remote_url: str = ""

    @property
    def available(self) -> bool:
        return self.commit != UNAVAILABLE

    @property
    def reproducible(self) -> bool:
        """True only when the exact running code exists in git history."""
        return self.available and not self.dirty

    @property
    def describe(self) -> str:
        if not self.available:
            return UNAVAILABLE
        parts = [self.short_commit]
        if self.branch and self.branch != UNAVAILABLE:
            parts.append(f"@{self.branch}")
        if self.tag:
            parts.append(f"({self.tag})")
        if self.dirty:
            parts.append("[dirty]")
        return " ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "short_commit": self.short_commit,
            "branch": self.branch,
            "tag": self.tag,
            "dirty": self.dirty,
            "dirty_files": list(self.dirty_files),
            "remote_url": self.remote_url,
            "reproducible": self.reproducible,
            "describe": self.describe,
        }


def collect(repo: Path | None = None) -> GitProvenance:
    """Read provenance from the working tree. Never raises."""
    root = repo or Path(__file__).resolve().parents[3]
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return GitProvenance()

    status = _git(["status", "--porcelain"], root)
    dirty_files = tuple(
        line[3:].strip() for line in status.splitlines()[:20] if line.strip()
    )
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root) or UNAVAILABLE
    # --exact-match: only report a tag when HEAD *is* the tag, not the nearest
    # ancestor. "close to v1.2" is not the same claim as "is v1.2".
    tag = _git(["describe", "--tags", "--exact-match"], root)

    return GitProvenance(
        commit=commit,
        short_commit=commit[:12],
        branch=branch,
        tag=tag,
        dirty=bool(status.strip()),
        dirty_files=dirty_files,
        remote_url=_git(["config", "--get", "remote.origin.url"], root),
    )


def require_clean_tree(repo: Path | None = None) -> GitProvenance:
    """Provenance, with a warning when the tree is dirty.

    Not fatal by itself — the promotion gate decides whether an unreproducible
    model may ship — but always logged, because a dirty-tree model that reaches
    production is a model nobody can rebuild.
    """
    provenance = collect(repo)
    if provenance.available and provenance.dirty:
        log.warning(
            "git.dirty_tree",
            commit=provenance.short_commit,
            files=len(provenance.dirty_files),
            detail="model will be marked unreproducible from git history",
        )
    return provenance
