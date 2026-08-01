"""One canonical feature pipeline, proven end to end.

The guarantee this file exists to establish: **live collection, archive replay,
dataset export, and the training loader all produce the same numbers for the
same market at the same instant.**

If they can diverge, every offline result is a guess about live behaviour. So
the test does not compare summaries or tolerances — it compares the fingerprint
of the full feature vector, which is equality to ten decimal places across all
seventy columns.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pmbtc.config import load_config
from pmbtc.constants import Outcome, SettlementSource
from pmbtc.dataset.export import DatasetExporter, build_rows
from pmbtc.dataset.providers import SnapshotContext
from pmbtc.dataset.quality import QualityFilter, QualityScorer
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord
from pmbtc.dataset.store import DatasetStore
from pmbtc.features.families import all_features
from pmbtc.features.graph import build_graph
from pmbtc.features.matrix import MatrixBuilder, feature_set_version
from pmbtc.features.provider import (
    ENGINEERED_FEATURE_NAMES,
    EngineeredFeatureProvider,
    canonical_feature_schema_version,
    observations_to_values,
    vector_fingerprint,
)
from pmbtc.live.archive import TickArchive
from pmbtc.live.clob import PolymarketMarketFeed

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

UP = "75947675343300382612239152835580192418310954975177482443770591765733661013763"
DOWN = "82813676602798150736076499247275218984847413034838945124705815442226616387084"

SETTLE = 1_785_600_000_000
OPEN = SETTLE - 300_000
INSTANT = SETTLE - 60_000

#: These fixtures have no reference feed, so roughly a third of the engineered
#: columns are legitimately missing and the normal quality gate would exclude
#: the row. What is under test here is *vector identity across paths*, not the
#: quality gate — which has its own tests — so the filter is opened up.
PERMISSIVE = QualityFilter(min_snapshot_quality=0.0, min_coverage=0.0)


@pytest.fixture
def config(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})


def synthetic_frames() -> list[tuple[int, Any]]:
    """A deterministic stream in the real CLOB wire format."""
    frames: list[tuple[int, Any]] = [
        (
            OPEN + 1_000,
            {
                "event_type": "book",
                "asset_id": UP,
                "timestamp": str(OPEN + 900),
                "hash": "h0",
                "bids": [{"price": "0.48", "size": "100"}, {"price": "0.47", "size": "200"}],
                "asks": [{"price": "0.52", "size": "50"}, {"price": "0.53", "size": "150"}],
            },
        ),
        (
            OPEN + 1_100,
            {
                "event_type": "book",
                "asset_id": DOWN,
                "timestamp": str(OPEN + 1_000),
                "hash": "h1",
                "bids": [{"price": "0.47", "size": "60"}],
                "asks": [{"price": "0.53", "size": "90"}],
            },
        ),
    ]
    # A run of trades and incremental updates, all before the snapshot instant.
    for i in range(20):
        stamp = OPEN + 10_000 + i * 2_000
        frames.append(
            (
                stamp + 50,
                {
                    "event_type": "last_trade_price",
                    "asset_id": UP,
                    "price": f"{0.49 + 0.001 * (i % 5):.3f}",
                    "size": str(10 + i),
                    "side": "BUY" if i % 3 else "SELL",
                    "timestamp": str(stamp),
                },
            )
        )
        frames.append(
            (
                stamp + 60,
                {
                    "event_type": "price_change",
                    "timestamp": str(stamp + 10),
                    "hash": f"p{i}",
                    "price_changes": [
                        {
                            "asset_id": UP,
                            "price": "0.48",
                            "size": str(100 + i * 3),
                            "side": "BUY",
                        }
                    ],
                },
            )
        )
    return frames


def feed_from_frames(frames: list[tuple[int, Any]]) -> PolymarketMarketFeed:
    feed = PolymarketMarketFeed("wss://replay", UP, DOWN, slug="btc-updown-5m-x")

    async def run() -> None:
        for received_ms, payload in frames:
            await feed.handle(payload, received_ms)

    asyncio.run(run())
    return feed


def snapshot_context() -> SnapshotContext:
    return SnapshotContext(
        condition_id="0xabc",
        slug="btc-updown-5m-x",
        snapshot_time_ms=INSTANT,
        settlement_time_ms=SETTLE,
        window_open_ms=OPEN,
        horizon_seconds=60,
        now_ms=INSTANT,
        market={},
    )


def collect_values(feed: PolymarketMarketFeed) -> tuple[dict[str, float | None], list]:
    """Run the canonical provider and return (values, observations)."""
    provider = EngineeredFeatureProvider(QualityScorer(), clob=feed, reference=None)
    provider.window_open_price = 63_000.0
    provider.tick_size = 0.01
    observations = asyncio.run(provider.collect(snapshot_context()))
    assert provider.last_vector is not None
    return dict(provider.last_vector.values), observations


# =========================================================================== #
class TestSingleCanonicalPipeline:
    def test_only_one_provider_engineers_features(self) -> None:
        # The service must use the engineered provider and nothing else.
        import inspect

        from pmbtc.live import service

        source = inspect.getsource(service.CollectionService._providers_for)
        assert "EngineeredFeatureProvider" in inspect.getsource(service)
        assert "LivePolymarketProvider" not in source
        assert "TimeFeatureProvider" not in source

    def test_schema_version_is_single_sourced(self) -> None:
        graph = build_graph(all_features())
        assert canonical_feature_schema_version() == feature_set_version(graph)
        assert MatrixBuilder().version == canonical_feature_schema_version()

    def test_engineered_set_is_the_full_family_set(self) -> None:
        assert len(ENGINEERED_FEATURE_NAMES) == len(all_features())
        assert set(ENGINEERED_FEATURE_NAMES) == {f.name for f in all_features()}


class TestLiveVersusReplay:
    def test_replay_reproduces_live_bit_for_bit(self) -> None:
        frames = synthetic_frames()

        live_values, _ = collect_values(feed_from_frames(frames))
        replay_values, _ = collect_values(feed_from_frames(frames))

        assert vector_fingerprint(live_values) == vector_fingerprint(replay_values)
        assert live_values == replay_values

    def test_replay_through_the_archive_matches(self, tmp_path: Path) -> None:
        # The full round trip: write frames to the gzipped archive, read them
        # back, rebuild the book, recompute. This is what a backtest does.
        frames = synthetic_frames()
        live_values, _ = collect_values(feed_from_frames(frames))

        archive = TickArchive(tmp_path / "ticks", buffer_frames=5)
        for received_ms, payload in frames:
            archive.write("clob:test", payload, received_ms)
        archive.close()

        restored: list[tuple[int, Any]] = []
        for path in archive.files():
            restored.extend(archive.read_frames(path))
        assert len(restored) == len(frames)

        archived_values, _ = collect_values(feed_from_frames(restored))
        assert vector_fingerprint(archived_values) == vector_fingerprint(live_values)

    def test_populated_features_are_real(self) -> None:
        values, _ = collect_values(feed_from_frames(synthetic_frames()))
        populated = {k: v for k, v in values.items() if v is not None}
        assert len(populated) > 25
        assert populated["ob_mid"] == pytest.approx(0.50)
        assert populated["ob_spread"] == pytest.approx(0.04)


class TestDatasetRoundTrip:
    """Live -> snapshot -> store -> export -> training loader, unchanged."""

    def _market(self) -> MarketRecord:
        return MarketRecord(
            condition_id="0xabc", slug="btc-updown-5m-x", question="q",
            discovery_time_ms=OPEN - 86_400_000,
            open_time_ms=OPEN, lock_time_ms=SETTLE - 20_000,
            settlement_time_ms=SETTLE,
            settlement_provider=SettlementSource.CHAINLINK,
            settlement_spec_hash="h", official_outcome=Outcome.UP,
            settlement_probability=1.0, yes_final_price=1.0, no_final_price=0.0,
            resolved_at_ms=SETTLE + 30_000, resolution_source="polymarket_official",
        )

    def _store_snapshot(self, store: DatasetStore, observations: list) -> FeatureSnapshot:
        snapshot = FeatureSnapshot(
            condition_id="0xabc", slug="btc-updown-5m-x", horizon_seconds=60,
            snapshot_time_ms=INSTANT, captured_at_ms=INSTANT,
            settlement_time_ms=SETTLE, observations=tuple(observations),
        )
        store.record_snapshot(snapshot)
        return snapshot

    def test_snapshot_preserves_the_live_vector(self, tmp_path: Path) -> None:
        live_values, observations = collect_values(feed_from_frames(synthetic_frames()))
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(self._market())
        self._store_snapshot(store, observations)

        stored = store.snapshot("0xabc", 60)
        assert stored is not None
        recovered = observations_to_values(list(stored.observations))
        assert vector_fingerprint(recovered) == vector_fingerprint(live_values)

    def test_export_and_training_loader_agree(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        live_values, observations = collect_values(feed_from_frames(synthetic_frames()))
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(self._market())
        self._store_snapshot(store, observations)

        # Export path.
        result = DatasetExporter(config, store).export(
            "parquet", quality_filter=PERMISSIVE, name="integration"
        )
        import pyarrow.parquet as pq

        table = pq.read_table(result.path).to_pydict()

        # Training-loader path: the same columns the trainer reads.
        rows, _ = build_rows(
            {m.condition_id: m for m in store.markets()},
            store.snapshots(),
            quality_filter=PERMISSIVE,
        )
        assert len(rows) == 1
        loader_values = {
            name: rows[0].get(f"f_{name}") for name in ENGINEERED_FEATURE_NAMES
        }

        export_values = {
            name: table[f"f_{name}"][0] for name in ENGINEERED_FEATURE_NAMES
            if f"f_{name}" in table
        }

        assert vector_fingerprint(loader_values) == vector_fingerprint(live_values)
        assert vector_fingerprint(export_values) == vector_fingerprint(live_values)

    def test_all_four_paths_have_one_fingerprint(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """The headline guarantee, as a single assertion."""
        frames = synthetic_frames()

        # 1. Live collection.
        live_values, observations = collect_values(feed_from_frames(frames))

        # 2. Archive replay.
        archive = TickArchive(tmp_path / "ticks", buffer_frames=5)
        for received_ms, payload in frames:
            archive.write("clob:test", payload, received_ms)
        archive.close()
        restored: list[tuple[int, Any]] = []
        for path in archive.files():
            restored.extend(archive.read_frames(path))
        replay_values, _ = collect_values(feed_from_frames(restored))

        # 3. Dataset export.
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(self._market())
        self._store_snapshot(store, observations)
        result = DatasetExporter(config, store).export(
            "parquet", quality_filter=PERMISSIVE, name="four_way"
        )
        import pyarrow.parquet as pq

        table = pq.read_table(result.path).to_pydict()
        export_values = {
            name: table[f"f_{name}"][0]
            for name in ENGINEERED_FEATURE_NAMES
            if f"f_{name}" in table
        }

        # 4. Training loader.
        rows, _ = build_rows(
            {m.condition_id: m for m in store.markets()},
            store.snapshots(),
            quality_filter=PERMISSIVE,
        )
        loader_values = {
            name: rows[0].get(f"f_{name}") for name in ENGINEERED_FEATURE_NAMES
        }

        fingerprints = {
            "live": vector_fingerprint(live_values),
            "replay": vector_fingerprint(replay_values),
            "export": vector_fingerprint(export_values),
            "loader": vector_fingerprint(loader_values),
        }
        assert len(set(fingerprints.values())) == 1, fingerprints


class TestReadinessCoverage:
    def test_coverage_uses_the_engineered_set(self, config) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.models.readiness import check_readiness

        report = check_readiness(config, [], [])
        coverage = next(c for c in report.checks if c.name == "feature_coverage")
        assert "engineered features" in coverage.detail
        assert str(len(ENGINEERED_FEATURE_NAMES)) in coverage.detail

    def test_coverage_counts_engineered_features_present(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.models.readiness import check_readiness

        _, observations = collect_values(feed_from_frames(synthetic_frames()))
        snapshot = FeatureSnapshot(
            condition_id="0xabc", slug="s", horizon_seconds=60,
            snapshot_time_ms=INSTANT, captured_at_ms=INSTANT,
            settlement_time_ms=SETTLE, observations=tuple(observations),
        )
        report = check_readiness(config, [], [snapshot])
        coverage = next(c for c in report.checks if c.name == "feature_coverage")
        # Some features need the reference feed, which is absent here — so
        # coverage is partial, and that is the honest number.
        assert 0.0 < coverage.actual < 1.0


class TestNoIndependentFeatureEngineering:
    def test_legacy_providers_are_not_used_by_the_service(self) -> None:
        import inspect

        from pmbtc.live import service

        source = inspect.getsource(service)
        assert "LiveReferenceProvider" not in source
        assert "LivePolymarketProvider" not in source

    def test_engine_is_the_only_producer_of_engineered_names(self) -> None:
        # Every engineered column must come from a registered family feature.
        from pmbtc.features.registry import REGISTRY

        for name in ENGINEERED_FEATURE_NAMES:
            spec = REGISTRY.get(name)
            assert spec.formula and spec.inputs

    def test_schema_version_recorded_in_export_manifest(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        _, observations = collect_values(feed_from_frames(synthetic_frames()))
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(
            MarketRecord(
                condition_id="0xabc", slug="s", discovery_time_ms=OPEN - 1,
                open_time_ms=OPEN, lock_time_ms=SETTLE - 20_000,
                settlement_time_ms=SETTLE,
                settlement_provider=SettlementSource.CHAINLINK,
                settlement_spec_hash="h", official_outcome=Outcome.UP,
                resolved_at_ms=SETTLE + 1, resolution_source="polymarket_official",
            )
        )
        store.record_snapshot(
            FeatureSnapshot(
                condition_id="0xabc", slug="s", horizon_seconds=60,
                snapshot_time_ms=INSTANT, captured_at_ms=INSTANT,
                settlement_time_ms=SETTLE, observations=tuple(observations),
            )
        )
        result = DatasetExporter(config, store).export(
            "parquet", quality_filter=PERMISSIVE, name="manifest_check"
        )
        manifest = json.loads(
            (result.path.parent / "manifest_check.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["feature_schema_version"]
        assert manifest["counts"]["snapshots"] == 1
        assert np.isfinite(manifest["counts"]["markets"])
