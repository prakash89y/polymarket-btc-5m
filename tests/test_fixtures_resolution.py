"""Archive source resolution.

The property under test: determinism validation works on a clean checkout with
no collected data, without weakening what it asserts. A CI runner and a
developer machine must reach the same verdict about the code, differing only in
which bytes they replay — and the local production check must still be able to
demand a real live archive.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pmbtc.config import load_config
from pmbtc.features.base import FeatureContext
from pmbtc.features.engine import FeatureEngine
from pmbtc.live.archive import TickArchive
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.live.fixtures import (
    FIXTURE_TICKS,
    ArchiveSource,
    fixture_archive,
    in_continuous_integration,
    resolve_clob_archive,
)

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path: Path):  # type: ignore[no-untyped-def]
    # base_dir in a temp directory means no live archive exists, which is
    # exactly the state of a clean CI checkout.
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})


class TestFixturePresence:
    def test_committed_fixture_exists(self) -> None:
        # Without this, determinism cannot be verified on a clean checkout.
        assert (REPO / FIXTURE_TICKS).is_dir()
        assert fixture_archive(REPO).available

    def test_fixture_is_small_enough_to_commit(self) -> None:
        total = sum(p.stat().st_size for p in fixture_archive(REPO).files)
        assert total < 512 * 1024, f"{total / 1024:.0f} KB"

    def test_fixture_covers_every_event_type(self) -> None:
        # A fixture exercising one code path proves one code path.
        archive = TickArchive(REPO / FIXTURE_TICKS, enabled=False)
        kinds: set[str] = set()
        tokens: set[str] = set()
        for path in fixture_archive(REPO).files:
            for _, payload in archive.read_frames(path):
                for event in payload if isinstance(payload, list) else [payload]:
                    if isinstance(event, dict):
                        kinds.add(str(event.get("event_type", "")))
                        if event.get("event_type") == "book":
                            tokens.add(str(event.get("asset_id", "")))
        assert {"book", "price_change", "last_trade_price"} <= kinds
        assert len(tokens) >= 2


class TestResolution:
    def test_ci_prefers_the_fixture(self, config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        resolved = resolve_clob_archive(config, repo_root=REPO)
        assert resolved.source is ArchiveSource.FIXTURE
        assert resolved.available

    def test_ci_detection(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        assert in_continuous_integration() is False
        monkeypatch.setenv("CI", "true")
        assert in_continuous_integration() is True

    def test_clean_checkout_falls_back_to_the_fixture(self, config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # No CI marker, no live archive — a developer who has never collected.
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        resolved = resolve_clob_archive(config, repo_root=REPO)
        assert resolved.source is ArchiveSource.FIXTURE
        assert resolved.available

    def test_require_live_refuses_the_fixture(self, config) -> None:  # type: ignore[no-untyped-def]
        # The local production check must not be satisfiable by committed data.
        resolved = resolve_clob_archive(config, require_live=True, repo_root=REPO)
        assert resolved.source is ArchiveSource.NONE
        assert not resolved.available
        assert "collector" in resolved.reason

    def test_source_is_always_reported(self, config) -> None:  # type: ignore[no-untyped-def]
        resolved = resolve_clob_archive(config, repo_root=REPO)
        assert resolved.source.value in resolved.describe()
        assert resolved.reason


class TestDeterminismOnTheFixture:
    """The assertion CI relies on, verified here against the committed bytes."""

    def _replay(self) -> tuple[str, int]:
        resolved = fixture_archive(REPO)
        archive = TickArchive(resolved.root, enabled=False)
        frames = archive.read_frames(resolved.files[0])

        tokens: list[str] = []
        for _, payload in frames:
            for event in payload if isinstance(payload, list) else [payload]:
                if isinstance(event, dict) and event.get("event_type") == "book":
                    asset = str(event.get("asset_id", ""))
                    if asset and asset not in tokens:
                        tokens.append(asset)
            if len(tokens) >= 2:
                break

        feed = PolymarketMarketFeed(
            "wss://replay", tokens[0], tokens[1] if len(tokens) > 1 else "", slug="fixture"
        )
        midpoint = max(1, len(frames) // 2)

        async def run() -> None:
            for received_ms, payload in frames[:midpoint]:
                await feed.handle(payload, received_ms)

        asyncio.run(run())
        last_ms = frames[midpoint - 1][0]
        vector = FeatureEngine().compute(
            FeatureContext(
                now_ms=last_ms,
                window_open_ms=last_ms - 150_000,
                settlement_ms=last_ms + 150_000,
                books=feed.books,
                pm_tape=feed.up_tape,
                regime={"tick_size": 0.01},
            )
        )
        return vector.fingerprint(), vector.populated

    def test_replay_is_reproducible(self) -> None:
        first, populated = self._replay()
        second, _ = self._replay()
        assert first == second
        assert populated > 10

    def test_fixture_rebuilds_a_two_sided_book(self) -> None:
        _, populated = self._replay()
        assert populated >= 30, f"only {populated} features populated"
