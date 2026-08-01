"""Archive source resolution: committed fixture versus live collection.

The distinction this module encodes matters, so it is worth stating plainly.

**Determinism is a property of the code, not of any particular dataset.** To
verify it you need a *fixed* input replayed twice. A committed fixture is a
better fixed input than a live archive, because it is byte-identical on every
machine forever, whereas a live archive differs by host and by hour. So CI
verifies determinism against the fixture — and verifies it more strongly than a
live archive could, since the expected values can be pinned.

**A live archive proves something different and equally real:** that the
production collector is actually running and emitting output the replay path
can consume. That is an *operational* property. It is only meaningful where
collection happens, and it is meaningless on a runner that has never collected
anything.

Conflating the two is what broke the release workflow: a code invariant was
being tested with an operational precondition attached. Splitting them means CI
gets a stronger determinism check, local validation keeps the operational one,
and neither is weakened.

Nothing here skips a check. The determinism assertions run everywhere; only the
*source* of the frames changes, and which source was used is always reported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.live.fixtures")

#: Committed slice of a real CLOB stream. Small, but covers every event type the
#: parser handles, for both outcome tokens.
FIXTURE_TICKS = Path("tests/fixtures/ticks")


class ArchiveSource(StrEnum):
    LIVE = "live"
    FIXTURE = "fixture"
    NONE = "none"


def in_continuous_integration() -> bool:
    """Whether this process is running on a CI runner.

    ``GITHUB_ACTIONS`` is set by GitHub; ``CI`` is the near-universal
    convention. Either is enough — the point is to know that no live collector
    has ever run here.
    """
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"


@dataclass(frozen=True, slots=True)
class ResolvedArchive:
    """Where replay frames are coming from, and why."""

    root: Path
    source: ArchiveSource
    files: tuple[Path, ...]
    reason: str

    @property
    def available(self) -> bool:
        return bool(self.files)

    @property
    def is_fixture(self) -> bool:
        return self.source is ArchiveSource.FIXTURE

    def describe(self) -> str:
        return (
            f"{self.source.value}: {len(self.files)} file(s) under "
            f"{self.root} — {self.reason}"
        )


def _clob_files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(sorted(p for p in root.rglob("*.jsonl.gz") if "clob" in p.name))


def resolve_clob_archive(
    config: Config, *, require_live: bool = False, repo_root: Path | None = None
) -> ResolvedArchive:
    """Pick the archive to replay from.

    Order of preference:

    1. A live archive, when one exists and ``require_live`` or local operation
       makes it the meaningful source.
    2. The committed fixture, which is always present in a checkout.

    ``require_live=True`` refuses the fixture entirely — that is the local
    production check, asserting the collector really is producing replayable
    output.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    live_root = config.resolved_path(config.feeds.archive_dir)
    live = _clob_files(live_root)

    if require_live:
        return ResolvedArchive(
            root=live_root,
            source=ArchiveSource.LIVE if live else ArchiveSource.NONE,
            files=live,
            reason=(
                "live archive required (production validation)"
                if live
                else "live archive required but none found — has the collector run?"
            ),
        )

    # On CI the fixture is preferred even if some archive somehow exists: the
    # committed bytes are the ones whose interpretation is pinned.
    if in_continuous_integration():
        fixture_root = root / FIXTURE_TICKS
        return ResolvedArchive(
            root=fixture_root,
            source=ArchiveSource.FIXTURE,
            files=_clob_files(fixture_root),
            reason="running in CI; using the committed fixture for reproducibility",
        )

    if live:
        return ResolvedArchive(
            root=live_root,
            source=ArchiveSource.LIVE,
            files=live,
            reason="live archive present on this machine",
        )

    fixture_root = root / FIXTURE_TICKS
    return ResolvedArchive(
        root=fixture_root,
        source=ArchiveSource.FIXTURE,
        files=_clob_files(fixture_root),
        reason="no live archive on this machine; using the committed fixture",
    )


def fixture_archive(repo_root: Path | None = None) -> ResolvedArchive:
    """The committed fixture, unconditionally. Used by the test suite."""
    root = (repo_root or Path(__file__).resolve().parents[3]) / FIXTURE_TICKS
    return ResolvedArchive(
        root=root,
        source=ArchiveSource.FIXTURE,
        files=_clob_files(root),
        reason="committed fixture requested explicitly",
    )
