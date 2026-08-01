"""Dataset integrity: leakage, immutability, quality, labels, versioning, export.

The leakage tests are the point of this file. They do not merely check that
clean data passes — they poison data in each of the ways leakage actually
happens and assert that the guard catches every one. A detector that has never
been shown a positive case is not a detector.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pmbtc.config import load_config
from pmbtc.constants import Outcome, SettlementSource
from pmbtc.dataset import (
    DatasetExporter,
    DatasetStore,
    FeatureSnapshot,
    ImmutableRecordError,
    LeakageGuard,
    LeakageKind,
    MarketRecord,
    Observation,
    QualityFilter,
    QualityFlag,
    QualityScorer,
    audit_against_reference,
    build_rows,
    compute_stats,
    config_hash,
    resolve_label,
    scan_monotonicity,
    scan_snapshots,
)
from pmbtc.exceptions import DataError, LookaheadError

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gamma"

SETTLE_MS = 1_785_503_400_000  # 2026-07-31T13:10:00Z
OPEN_MS = SETTLE_MS - 300_000


@pytest.fixture
def config(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})


@pytest.fixture
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore(tmp_path / "dataset")


def make_observation(
    name: str = "pm_mid",
    value: float | None = 0.51,
    *,
    event_time_ms: int | None = None,
    observation_time_ms: int | None = None,
    source: str = "polymarket_gamma",
    flags: tuple[QualityFlag, ...] = (QualityFlag.OK,),
) -> Observation:
    instant = SETTLE_MS - 60_000
    return Observation(
        name=name,
        value=value,
        data_source=source,
        event_time_ms=event_time_ms if event_time_ms is not None else instant - 500,
        observation_time_ms=(
            observation_time_ms if observation_time_ms is not None else instant - 100
        ),
        flags=flags,
    )


def make_snapshot(
    horizon: int = 60,
    observations: tuple[Observation, ...] | None = None,
    *,
    condition_id: str = "0xmarket",
    settlement_ms: int = SETTLE_MS,
    snapshot_time_ms: int | None = None,
) -> FeatureSnapshot:
    instant = (
        snapshot_time_ms if snapshot_time_ms is not None else settlement_ms - horizon * 1000
    )
    return FeatureSnapshot(
        condition_id=condition_id,
        slug="btc-updown-5m-1785503100",
        horizon_seconds=horizon,
        snapshot_time_ms=instant,
        captured_at_ms=instant,
        settlement_time_ms=settlement_ms,
        observations=observations
        if observations is not None
        else (
            Observation(
                name="pm_mid",
                value=0.51,
                data_source="polymarket_gamma",
                event_time_ms=instant - 500,
                observation_time_ms=instant - 100,
            ),
        ),
    )


def make_market(
    *,
    condition_id: str = "0xmarket",
    outcome: Outcome | None = Outcome.UP,
    settlement_ms: int = SETTLE_MS,
) -> MarketRecord:
    return MarketRecord(
        condition_id=condition_id,
        market_id="3221494",
        event_id="772405",
        series_id="10684",
        series_slug="btc-up-or-down-5m",
        slug="btc-updown-5m-1785503100",
        question="Bitcoin Up or Down",
        discovery_time_ms=settlement_ms - 86_400_000,
        open_time_ms=settlement_ms - 300_000,
        lock_time_ms=settlement_ms - 20_000,
        settlement_time_ms=settlement_ms,
        settlement_provider=SettlementSource.CHAINLINK,
        settlement_spec_hash="0d0a08acbdcf5781",
        settlement_parser_version="2.0",
        trading_pair="BTC/USD",
        tie_rule="tie_up",
        official_outcome=outcome,
        settlement_probability=1.0 if outcome else None,
        yes_final_price=1.0 if outcome is Outcome.UP else 0.0,
        no_final_price=0.0 if outcome is Outcome.UP else 1.0,
        resolved_at_ms=settlement_ms + 30_000 if outcome else None,
        resolution_source="polymarket_official" if outcome else "",
    )


# =========================================================================== #
# Leakage — the mandatory requirement
# =========================================================================== #
class TestLeakageDetection:
    def test_clean_snapshot_passes(self) -> None:
        assert LeakageGuard().inspect(make_snapshot()) == []

    def test_future_event_is_caught(self) -> None:
        # The canonical leak: a value whose event happens after the instant the
        # snapshot claims to represent.
        instant = SETTLE_MS - 60_000
        snapshot = make_snapshot(
            observations=(make_observation(event_time_ms=instant + 5_000),)
        )
        findings = LeakageGuard().inspect(snapshot)
        assert any(f.kind is LeakageKind.FUTURE_EVENT for f in findings)

    def test_future_observation_is_caught(self) -> None:
        # Subtler: the event predates the instant, but we only received it
        # afterwards, so acting on it in real time was impossible.
        instant = SETTLE_MS - 60_000
        snapshot = make_snapshot(
            observations=(
                make_observation(
                    event_time_ms=instant - 1_000, observation_time_ms=instant + 30_000
                ),
            )
        )
        findings = LeakageGuard().inspect(snapshot)
        assert any(f.kind is LeakageKind.FUTURE_OBSERVATION for f in findings)

    def test_small_jitter_is_tolerated(self) -> None:
        instant = SETTLE_MS - 60_000
        snapshot = make_snapshot(
            observations=(
                make_observation(
                    event_time_ms=instant - 1_000, observation_time_ms=instant + 200
                ),
            )
        )
        assert LeakageGuard(observation_tolerance_ms=500).inspect(snapshot) == []

    def test_post_settlement_event_is_caught(self) -> None:
        snapshot = make_snapshot(
            observations=(make_observation(event_time_ms=SETTLE_MS + 60_000),)
        )
        kinds = {f.kind for f in LeakageGuard().inspect(snapshot)}
        assert LeakageKind.POST_SETTLEMENT in kinds

    def test_label_as_a_feature_is_caught(self) -> None:
        # Even with impeccable timestamps, an outcome-derived column is leakage
        # by definition.
        snapshot = make_snapshot(observations=(make_observation(name="settlement_probability"),))
        findings = LeakageGuard().inspect(snapshot)
        assert any(f.kind is LeakageKind.LABEL_IN_FEATURES for f in findings)

    @pytest.mark.parametrize(
        "name", ["label", "outcome", "official_outcome", "yes_final_price", "payout"]
    )
    def test_every_forbidden_name_is_rejected(self, name: str) -> None:
        snapshot = make_snapshot(observations=(make_observation(name=name),))
        assert LeakageGuard().inspect(snapshot)

    def test_horizon_mismatch_is_caught(self) -> None:
        # A snapshot labelled T-60 whose instant is not settlement-60s would
        # silently shift every feature in it.
        snapshot = make_snapshot(horizon=60, snapshot_time_ms=SETTLE_MS - 90_000)
        findings = LeakageGuard().inspect(snapshot)
        assert any(f.kind is LeakageKind.HORIZON_MISMATCH for f in findings)

    def test_snapshot_after_settlement_is_caught(self) -> None:
        snapshot = make_snapshot(horizon=-30)
        kinds = {f.kind for f in LeakageGuard().inspect(snapshot)}
        assert LeakageKind.SNAPSHOT_AFTER_SETTLEMENT in kinds

    def test_missing_values_cannot_leak(self) -> None:
        # No value, no information, nothing to leak.
        instant = SETTLE_MS - 60_000
        snapshot = make_snapshot(
            observations=(make_observation(value=None, event_time_ms=instant + 99_000),)
        )
        assert [f for f in LeakageGuard().inspect(snapshot)
                if f.kind is LeakageKind.FUTURE_EVENT] == []

    def test_guard_refuses_to_store_leaked_snapshots(self, store: DatasetStore) -> None:
        instant = SETTLE_MS - 60_000
        snapshot = make_snapshot(
            observations=(make_observation(event_time_ms=instant + 10_000),)
        )
        with pytest.raises(LookaheadError) as exc:
            store.record_snapshot(snapshot)
        assert exc.value.halts_trading is True
        assert store.snapshots() == []

    def test_dataset_wide_scan(self) -> None:
        instant = SETTLE_MS - 60_000
        clean = make_snapshot(horizon=120)
        dirty = make_snapshot(
            horizon=60, observations=(make_observation(event_time_ms=instant + 10_000),)
        )
        assert scan_snapshots([clean]) == []
        assert scan_snapshots([clean, dirty])

    def test_monotonicity_across_horizons(self) -> None:
        # A T-300 snapshot must not carry information newer than a later one.
        early = make_snapshot(
            horizon=300,
            observations=(make_observation(event_time_ms=SETTLE_MS - 310_000),),
        )
        late = make_snapshot(
            horizon=60,
            observations=(make_observation(event_time_ms=SETTLE_MS - 320_000),),
        )
        assert scan_monotonicity([early, late])

    def test_monotonicity_accepts_ordered_snapshots(self) -> None:
        early = make_snapshot(
            horizon=300, observations=(make_observation(event_time_ms=SETTLE_MS - 310_000),)
        )
        late = make_snapshot(
            horizon=60, observations=(make_observation(event_time_ms=SETTLE_MS - 61_000),)
        )
        assert scan_monotonicity([early, late]) == []


# =========================================================================== #
# Immutability
# =========================================================================== #
class TestImmutability:
    def test_snapshot_is_written_once(self, store: DatasetStore) -> None:
        store.record_snapshot(make_snapshot())
        with pytest.raises(ImmutableRecordError):
            store.record_snapshot(make_snapshot())

    def test_different_horizons_coexist(self, store: DatasetStore) -> None:
        for horizon in (300, 60, 1):
            store.record_snapshot(make_snapshot(horizon))
        assert len(store.snapshots("0xmarket")) == 3

    def test_timeline_is_ordered_by_countdown(self, store: DatasetStore) -> None:
        for horizon in (60, 300, 1):
            store.record_snapshot(make_snapshot(horizon))
        assert [s.horizon_seconds for s in store.snapshots("0xmarket")] == [300, 60, 1]

    def test_store_survives_a_restart(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        first = DatasetStore(root)
        first.record_snapshot(make_snapshot())
        first.upsert_market(make_market())

        reopened = DatasetStore(root)
        assert reopened.has_snapshot("0xmarket", 60)
        assert reopened.market("0xmarket") is not None
        with pytest.raises(ImmutableRecordError):
            reopened.record_snapshot(make_snapshot())

    def test_label_can_be_filled_in_once(self, store: DatasetStore) -> None:
        store.upsert_market(make_market(outcome=None))
        assert store.market("0xmarket").is_labelled is False  # type: ignore[union-attr]
        assert store.upsert_market(make_market(outcome=Outcome.UP)) is True
        assert store.market("0xmarket").official_outcome is Outcome.UP  # type: ignore[union-attr]

    def test_label_cannot_be_changed(self, store: DatasetStore) -> None:
        # A silently rewritten label invalidates every model trained before it.
        store.upsert_market(make_market(outcome=Outcome.UP))
        with pytest.raises(ImmutableRecordError):
            store.upsert_market(make_market(outcome=Outcome.DOWN))

    def test_corrupt_line_does_not_destroy_the_dataset(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        store = DatasetStore(root)
        store.record_snapshot(make_snapshot())
        (root / "snapshots.jsonl").open("a", encoding="utf-8").write("{ not json\n")
        assert len(DatasetStore(root).snapshots()) == 1


# =========================================================================== #
# Quality and time alignment
# =========================================================================== #
class TestQuality:
    def test_latency_is_event_to_observation(self) -> None:
        obs = make_observation(event_time_ms=1_000, observation_time_ms=1_250)
        assert obs.latency_ms == 250

    def test_delayed_flag(self) -> None:
        scorer = QualityScorer()
        obs = scorer.build(
            name="pm_mid",
            value=0.5,
            data_source="polymarket_gamma",
            event_time_ms=1_000_000,
            observation_time_ms=1_005_000,  # 5s: past the 2s budget
            snapshot_time_ms=1_005_000,
        )
        assert QualityFlag.DELAYED in obs.flags
        assert obs.is_usable is True

    def test_hard_budget_excludes_the_value(self) -> None:
        scorer = QualityScorer()
        obs = scorer.build(
            name="pm_mid",
            value=0.5,
            data_source="polymarket_gamma",
            event_time_ms=1_000_000,
            observation_time_ms=1_060_000,  # 60s: past the hard budget
            snapshot_time_ms=1_060_000,
        )
        assert QualityFlag.OUT_OF_BUDGET in obs.flags
        assert obs.is_usable is False
        assert obs.quality_score == 0.0

    def test_stale_flag_is_about_the_snapshot_not_the_wire(self) -> None:
        scorer = QualityScorer()
        obs = scorer.build(
            name="pm_mid",
            value=0.5,
            data_source="polymarket_gamma",
            event_time_ms=1_000_000,
            observation_time_ms=1_000_500,   # arrived promptly
            snapshot_time_ms=1_120_000,      # but the event is 2 minutes old
        )
        assert QualityFlag.STALE in obs.flags
        assert QualityFlag.DELAYED not in obs.flags

    def test_slow_sources_get_generous_budgets(self) -> None:
        # Fear & Greed updates daily; an hour of latency is not a defect.
        scorer = QualityScorer()
        obs = scorer.build(
            name="fear_greed",
            value=55.0,
            data_source="fear_greed",
            event_time_ms=1_000_000,
            observation_time_ms=1_000_000 + 1_800_000,
            snapshot_time_ms=1_000_000 + 1_800_000,
        )
        assert QualityFlag.OUT_OF_BUDGET not in obs.flags

    def test_unknown_source_is_treated_strictly(self) -> None:
        scorer = QualityScorer()
        assert scorer.budget_for("something_new").confidence == 0.5

    def test_missing_observation_is_recorded_not_omitted(self) -> None:
        obs = QualityScorer().missing("ref_price", "binance_spot", 1_000)
        assert obs.value is None
        assert QualityFlag.MISSING in obs.flags
        assert obs.quality_score == 0.0

    def test_snapshot_quality_and_coverage(self) -> None:
        snapshot = make_snapshot(
            observations=(
                make_observation(name="a", value=1.0),
                make_observation(name="b", value=None, flags=(QualityFlag.MISSING,)),
            )
        )
        assert snapshot.coverage == pytest.approx(0.5)
        assert 0.0 < snapshot.quality_score < 1.0
        assert snapshot.missing_count == 1

    def test_quality_filter_excludes_low_quality_snapshots(self) -> None:
        assert QualityFilter().accepts_snapshot(0.9, 0.9) is True
        assert QualityFilter().accepts_snapshot(0.2, 0.9) is False
        assert QualityFilter().accepts_snapshot(0.9, 0.1) is False

    def test_scheduling_jitter_is_recorded(self) -> None:
        instant = SETTLE_MS - 60_000
        snapshot = FeatureSnapshot(
            condition_id="c",
            horizon_seconds=60,
            snapshot_time_ms=instant,
            captured_at_ms=instant + 320,
            settlement_time_ms=SETTLE_MS,
        )
        assert snapshot.scheduling_jitter_ms == 320


# =========================================================================== #
# Labels
# =========================================================================== #
class TestLabels:
    def _resolved(self, up: str, down: str, closed: bool = True) -> dict[str, Any]:
        return {
            "conditionId": "0xabc",
            "closed": closed,
            "outcomes": json.dumps(["Up", "Down"]),
            "outcomePrices": json.dumps([up, down]),
        }

    def test_settled_up(self) -> None:
        result = resolve_label(self._resolved("1", "0"))
        assert result.outcome is Outcome.UP
        assert result.resolved is True
        assert result.yes_final_price == 1.0
        assert result.no_final_price == 0.0

    def test_settled_down(self) -> None:
        result = resolve_label(self._resolved("0", "1"))
        assert result.outcome is Outcome.DOWN
        assert result.settlement_probability == 1.0

    def test_open_market_has_no_label(self) -> None:
        result = resolve_label(self._resolved("0.5", "0.5", closed=False))
        assert result.resolved is False
        assert result.outcome is None

    def test_unsettled_prices_are_not_guessed(self) -> None:
        # Closed but still quoting 0.6/0.4 is not a payout.
        result = resolve_label(self._resolved("0.6", "0.4"))
        assert result.resolved is False
        assert result.outcome is None

    def test_fifty_fifty_void_is_resolved_but_unlabelled(self) -> None:
        result = resolve_label(self._resolved("0.5", "0.5"))
        assert result.resolved is True
        assert result.outcome is None
        assert result.is_void is True

    def test_outcomes_are_matched_by_name_not_position(self) -> None:
        payload = {
            "conditionId": "0xabc",
            "closed": True,
            "outcomes": json.dumps(["Down", "Up"]),
            "outcomePrices": json.dumps(["0", "1"]),
        }
        assert resolve_label(payload).outcome is Outcome.UP

    def test_real_resolved_fixture(self) -> None:
        payload = json.loads(
            (FIXTURES / "btc_5m_resolved_2026_04.json").read_text(encoding="utf-8")
        )
        result = resolve_label(payload)
        assert result.resolved is True
        assert result.outcome is Outcome.DOWN  # outcomePrices ["0", "1"]

    def test_audit_never_changes_the_label(self) -> None:
        # External feeds are for audit only; disagreement is an alarm, not an
        # override.
        audit = audit_against_reference(
            "0xabc", Outcome.UP, open_price=100.0, close_price=99.0,
            reference_source="binance_spot",
        )
        assert audit.official is Outcome.UP
        assert audit.reference is Outcome.DOWN
        assert audit.agrees is False

    def test_audit_applies_the_tie_rule(self) -> None:
        audit = audit_against_reference(
            "0xabc", Outcome.UP, 100.0, 100.0, "binance_spot", tie_resolves_up=True
        )
        assert audit.reference is Outcome.UP
        assert audit.agrees is True

    def test_market_label_is_binary(self) -> None:
        assert make_market(outcome=Outcome.UP).label == 1
        assert make_market(outcome=Outcome.DOWN).label == 0
        assert make_market(outcome=None).label is None


# =========================================================================== #
# Versioning and reproducibility
# =========================================================================== #
class TestManifest:
    def test_config_hash_is_stable(self, config) -> None:  # type: ignore[no-untyped-def]
        assert config_hash(config) == config_hash(config)

    def test_config_hash_changes_with_collection_semantics(self, tmp_path: Path) -> None:
        base = load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path)})
        changed = load_config(
            SHIPPED_CONFIG,
            app={"base_dir": str(tmp_path)},
            clock={"order_safety_window_seconds": 25},
        )
        assert config_hash(base) != config_hash(changed)

    def test_config_hash_ignores_cosmetic_settings(self, tmp_path: Path) -> None:
        # A fingerprint that changes when someone edits a log level is one
        # people learn to ignore.
        base = load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path)})
        noisy = load_config(
            SHIPPED_CONFIG, app={"base_dir": str(tmp_path)}, logging={"level": "DEBUG"}
        )
        assert config_hash(base) == config_hash(noisy)


# =========================================================================== #
# Export
# =========================================================================== #
class TestExport:
    @pytest.fixture
    def populated(self, config, tmp_path: Path) -> DatasetStore:  # type: ignore[no-untyped-def]
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(make_market())
        for horizon in (300, 120, 60, 5, 1):
            store.record_snapshot(make_snapshot(horizon))
        return store

    def test_rows_carry_metadata_and_label(self, populated: DatasetStore) -> None:
        markets = {m.condition_id: m for m in populated.markets()}
        rows, excluded = build_rows(markets, populated.snapshots())
        assert len(rows) == 5
        assert excluded == 0
        assert rows[0]["label"] == 1
        assert rows[0]["settlement_provider"] == "chainlink"
        assert rows[0]["settlement_spec_hash"] == "0d0a08acbdcf5781"
        assert "f_pm_mid" in rows[0]

    def test_rows_are_chronological_then_countdown(self, populated: DatasetStore) -> None:
        markets = {m.condition_id: m for m in populated.markets()}
        rows, _ = build_rows(markets, populated.snapshots())
        assert [r["horizon_seconds"] for r in rows] == [300, 120, 60, 5, 1]

    def test_unlabelled_markets_are_excluded_by_default(
        self, config, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(make_market(outcome=None))
        store.record_snapshot(make_snapshot())
        markets = {m.condition_id: m for m in store.markets()}
        rows, excluded = build_rows(markets, store.snapshots())
        assert rows == []
        assert excluded == 1

    def test_parquet_is_the_default(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        result = DatasetExporter(config, populated).export()
        assert result.format == "parquet"
        assert result.path.suffix == ".parquet"
        assert result.rows == 5

    def test_parquet_round_trip(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        import pyarrow.parquet as pq

        result = DatasetExporter(config, populated).export("parquet")
        table = pq.read_table(result.path)
        assert table.num_rows == 5
        assert "label" in table.column_names
        assert "f_pm_mid" in table.column_names

    def test_arrow_round_trip(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        import pyarrow as pa

        result = DatasetExporter(config, populated).export("arrow")
        with pa.OSFile(str(result.path), "rb") as source:
            assert pa.ipc.open_file(source).read_all().num_rows == 5

    def test_csv_round_trip(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        import csv

        result = DatasetExporter(config, populated).export("csv")
        with result.path.open(encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == 5

    def test_sqlite_round_trip(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        result = DatasetExporter(config, populated).export("sqlite")
        with sqlite3.connect(result.path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            horizons = connection.execute(
                "SELECT horizon_seconds FROM samples ORDER BY horizon_seconds DESC"
            ).fetchall()
        assert count == 5
        assert horizons[0][0] == 300

    def test_manifest_pins_reproducibility(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        manifest = DatasetExporter(config, populated).export().manifest
        assert manifest.dataset_version
        assert manifest.feature_schema_version
        assert manifest.settlement_schema_version == "2.0"
        assert manifest.config_hash
        assert manifest.git_commit  # "unavailable" outside a checkout, never blank
        assert manifest.period_start_ms and manifest.period_end_ms
        assert manifest.settlement_providers == {"chainlink": 1}
        assert "chainlink" not in manifest.fingerprint  # fingerprint is versions only

    def test_manifest_file_is_written(self, config, populated: DatasetStore) -> None:  # type: ignore[no-untyped-def]
        result = DatasetExporter(config, populated).export(name="training")
        manifest_path = result.path.parent / "training.manifest.json"
        assert manifest_path.is_file()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["counts"]["snapshots"] == 5
        assert payload["horizons"][0] == 300

    def test_export_refuses_to_write_leaked_data(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        # A snapshot written by an older, buggier build must not survive into a
        # training set just because it is already on disk.
        store = DatasetStore(tmp_path / "dataset")
        store.upsert_market(make_market())
        store.record_snapshot(make_snapshot())
        poisoned = make_snapshot(
            horizon=120,
            observations=(make_observation(event_time_ms=SETTLE_MS + 1_000),),
        )
        store._snapshots[poisoned.key] = poisoned  # bypass the write-time guard

        with pytest.raises(LookaheadError):
            DatasetExporter(config, store).export()

    def test_empty_export_fails_loudly(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(DataError):
            DatasetExporter(config, DatasetStore(tmp_path / "empty")).export()


# =========================================================================== #
# Statistics
# =========================================================================== #
class TestStats:
    def test_counts_and_balance(self) -> None:
        markets = [
            make_market(condition_id="a", outcome=Outcome.UP),
            make_market(condition_id="b", outcome=Outcome.DOWN),
            make_market(condition_id="c", outcome=None),
        ]
        snapshots = [make_snapshot(h, condition_id="a") for h in (300, 60, 1)]
        stats = compute_stats(markets, snapshots, horizons=(300, 60, 1))
        assert stats.markets == 3
        assert stats.labelled == 2
        assert stats.unlabelled == 1
        assert stats.class_balance == {"up": 1, "down": 1}
        assert stats.up_rate == pytest.approx(0.5)
        assert stats.settlement_providers == {"chainlink": 3}

    def test_timeline_completeness(self) -> None:
        markets = [make_market(condition_id="a")]
        snapshots = [make_snapshot(h, condition_id="a") for h in (300, 60)]
        stats = compute_stats(markets, snapshots, horizons=(300, 120, 60, 1))
        assert stats.timeline_completeness == pytest.approx(0.5)

    def test_coverage_and_missing_by_source(self) -> None:
        snapshot = make_snapshot(
            observations=(
                make_observation(name="pm_mid", value=0.5, source="polymarket_gamma"),
                make_observation(name="ref_price", value=None, source="binance_spot",
                                 flags=(QualityFlag.MISSING,)),
            )
        )
        stats = compute_stats([make_market()], [snapshot])
        assert stats.feature_coverage["pm_mid"] == 1.0
        assert stats.feature_coverage["ref_price"] == 0.0
        assert stats.missing_pct_by_source["binance_spot"] == 1.0
        assert stats.missing_pct_by_source["polymarket_gamma"] == 0.0

    def test_latency_by_source(self) -> None:
        instant = SETTLE_MS - 60_000
        snapshot = make_snapshot(
            observations=(
                make_observation(
                    name="pm_mid", event_time_ms=instant - 1_000,
                    observation_time_ms=instant - 500,
                ),
            )
        )
        stats = compute_stats([make_market()], [snapshot])
        assert stats.mean_latency_by_source["polymarket_gamma"] == pytest.approx(500)

    def test_quality_distribution_and_flags(self) -> None:
        snapshot = make_snapshot(
            observations=(
                make_observation(name="a", flags=(QualityFlag.STALE,)),
                make_observation(name="b", flags=(QualityFlag.OK,)),
            )
        )
        stats = compute_stats([make_market()], [snapshot])
        assert stats.flag_counts.get("stale") == 1
        assert sum(stats.quality_distribution.values()) == 1

    def test_void_markets_are_counted_separately(self) -> None:
        void = make_market(condition_id="v", outcome=None).model_copy(
            update={"resolved_at_ms": SETTLE_MS + 1_000}
        )
        stats = compute_stats([void], [])
        assert stats.void == 1
        assert stats.unlabelled == 0

    def test_report_dict_is_serialisable(self) -> None:
        stats = compute_stats([make_market()], [make_snapshot()])
        assert json.dumps(stats.as_dict())
