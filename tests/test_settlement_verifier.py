"""Verifier, store, and report tests.

The assertions that matter: the bot trades the Chainlink 5-minute market and
**refuses** every near-miss — right provider wrong interval, right family wrong
provider, tampered metadata, and a series that silently changed its resolution
source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pmbtc.config import load_config
from pmbtc.constants import SettlementSource
from pmbtc.exceptions import SettlementSourceMismatch
from pmbtc.settlement import (
    SettlementReport,
    SettlementSpecStore,
    SettlementVerifier,
    VerificationStatus,
    parse_settlement_spec,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gamma"
SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def spec_of(name: str):  # type: ignore[no-untyped-def]
    return parse_settlement_spec(load(name))


@pytest.fixture
def config():  # type: ignore[no-untyped-def]
    return load_config(SHIPPED_CONFIG)


@pytest.fixture
def verifier(config):  # type: ignore[no-untyped-def]
    return SettlementVerifier(config)


class TestHappyPath:
    def test_chainlink_5m_verifies(self, verifier) -> None:  # type: ignore[no-untyped-def]
        result = verifier.verify(spec_of("btc_5m_live_2026_07"))
        assert result.status is VerificationStatus.VERIFIED
        assert result.trading_enabled is True
        assert result.failures == ()

    def test_every_gate_ran(self, verifier) -> None:  # type: ignore[no-untyped-def]
        result = verifier.verify(spec_of("btc_5m_live_2026_07"))
        assert {c.name for c in result.checks} == {
            "no_source_conflict",
            "required_fields_present",
            "provider_matches_config",
            "trading_pair_supported",
            "interval_matches_config",
            "window_timing_consistent",
            "tie_rule_known",
            "detection_confidence",
            "settlement_unchanged",
        }

    def test_historical_market_also_verifies(self, verifier) -> None:  # type: ignore[no-untyped-def]
        assert verifier.verify(spec_of("btc_5m_resolved_2026_04")).trading_enabled is True


class TestRefusals:
    def test_wrong_provider_same_product_line(self, verifier) -> None:  # type: ignore[no-untyped-def]
        # The hourly "Bitcoin Up or Down" market resolves off Binance.
        result = verifier.verify(spec_of("btc_hourly_binance"))
        assert result.trading_enabled is False
        assert result.status is VerificationStatus.REJECTED_PROVIDER_MISMATCH
        assert any("binance_spot" in r for r in result.reasons)

    def test_right_provider_wrong_interval(self, verifier) -> None:  # type: ignore[no-untyped-def]
        result = verifier.verify(spec_of("btc_15m_live_2026_07"))
        assert result.trading_enabled is False
        assert result.status is VerificationStatus.REJECTED_INTERVAL_MISMATCH
        assert any("900s" in r for r in result.reasons)

    def test_prose_only_market_is_refused_on_confidence(self, verifier) -> None:  # type: ignore[no-untyped-def]
        result = verifier.verify(spec_of("btc_quarterly_binance_2025"))
        assert result.trading_enabled is False

    def test_metadata_conflict_is_refused(self, verifier) -> None:  # type: ignore[no-untyped-def]
        tampered = dict(load("btc_5m_live_2026_07"))
        tampered["resolutionSource"] = "https://www.binance.com/en/trade/BTC_USDT"
        result = verifier.verify(parse_settlement_spec(tampered))
        assert result.status is VerificationStatus.REJECTED_CONFLICT
        assert result.trading_enabled is False

    def test_unaligned_window_is_refused(self, verifier) -> None:  # type: ignore[no-untyped-def]
        tampered = json.loads(json.dumps(load("btc_5m_live_2026_07")))
        tampered["slug"] = "btc-updown-5m-1785503130"          # 30s off the grid
        tampered["events"][0]["startTime"] = "2026-07-31T13:05:30Z"
        tampered["endDate"] = "2026-07-31T13:10:30Z"
        result = verifier.verify(parse_settlement_spec(tampered))
        assert result.trading_enabled is False
        assert result.status is VerificationStatus.REJECTED_TIMING

    def test_confidence_gate_can_be_tightened_but_not_bypassed(self) -> None:
        # Lowering the gate to 0.7 would let a prose-only market through, which
        # is precisely why the default is 1.0.
        cfg = load_config(SHIPPED_CONFIG, settlement={"min_detection_confidence": 0.7})
        strict = load_config(SHIPPED_CONFIG)
        loose_result = SettlementVerifier(cfg).verify(spec_of("btc_quarterly_binance_2025"))
        strict_result = SettlementVerifier(strict).verify(spec_of("btc_quarterly_binance_2025"))
        # Still refused -- but now on provider/interval grounds, not confidence.
        assert loose_result.trading_enabled is False
        assert strict_result.trading_enabled is False

    def test_verify_or_raise(self, verifier) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(SettlementSourceMismatch) as exc:
            verifier.verify_or_raise(spec_of("btc_hourly_binance"))
        assert exc.value.halts_trading is True
        assert "binance_spot" in str(exc.value)

    def test_verify_or_raise_passes_good_market(self, verifier) -> None:  # type: ignore[no-untyped-def]
        assert verifier.verify_or_raise(spec_of("btc_5m_live_2026_07")).trading_enabled


class TestBacktestRelaxation:
    def test_backtest_may_replay_unverifiable_markets(self) -> None:
        cfg = load_config(
            SHIPPED_CONFIG,
            app={"mode": "backtest"},
            settlement={"require_verified": False, "allow_unknown_source": True},
        )
        result = SettlementVerifier(cfg).verify(spec_of("btc_quarterly_binance_2025"))
        assert result.status is not VerificationStatus.VERIFIED
        assert result.trading_enabled is True  # simulated only; no money can move

    def test_paper_mode_gets_no_such_relaxation(self, verifier) -> None:  # type: ignore[no-untyped-def]
        assert verifier.verify(spec_of("btc_quarterly_binance_2025")).trading_enabled is False


class TestSpecStore:
    def test_records_and_reads_back(self, tmp_path: Path) -> None:
        store = SettlementSpecStore(tmp_path / "specs.jsonl")
        spec = spec_of("btc_5m_live_2026_07")
        assert store.record(spec) is None
        assert len(store) == 1
        assert store.get(spec.condition_id) is not None
        assert store.get(spec.condition_id).spec_hash == spec.spec_hash  # type: ignore[union-attr]

    def test_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "specs.jsonl"
        spec = spec_of("btc_5m_live_2026_07")
        SettlementSpecStore(path).record(spec)
        reopened = SettlementSpecStore(path)
        assert reopened.get(spec.condition_id) is not None
        assert reopened.known_hash("btc-up-or-down-5m") == spec.spec_hash

    def test_recording_the_same_market_twice_is_idempotent(self, tmp_path: Path) -> None:
        store = SettlementSpecStore(tmp_path / "specs.jsonl")
        spec = spec_of("btc_5m_live_2026_07")
        store.record(spec)
        assert store.record(spec) is None
        assert len(store) == 1

    def test_consecutive_markets_share_a_fingerprint(self, tmp_path: Path) -> None:
        store = SettlementSpecStore(tmp_path / "specs.jsonl")
        store.record(spec_of("btc_5m_resolved_2026_04"))
        change = store.record(spec_of("btc_5m_live_2026_07"))
        assert change is None  # same mechanics, four months apart

    def test_detects_a_provider_change_within_a_series(self, tmp_path: Path) -> None:
        store = SettlementSpecStore(tmp_path / "specs.jsonl")
        store.record(spec_of("btc_5m_live_2026_07"))

        # Same series slug, but now resolving off Binance.
        moved = json.loads(json.dumps(load("btc_5m_live_2026_07")))
        moved["conditionId"] = "0xdifferentmarket"
        moved["slug"] = "btc-updown-5m-1785503400"
        moved["events"][0]["startTime"] = "2026-07-31T13:10:00Z"
        moved["endDate"] = "2026-07-31T13:15:00Z"
        moved["resolutionSource"] = "https://www.binance.com/en/trade/BTC_USDT"
        moved["description"] = (
            'This market will resolve to "Up" if the close price is greater than or '
            "equal to the open price for the BTC/USDT 1 hour candle. The resolution "
            "source for this market is information from Binance, specifically the "
            "BTC/USDT pair (https://www.binance.com/en/trade/BTC_USDT)."
        )
        change = store.record(parse_settlement_spec(moved))
        assert change is not None
        assert change.is_provider_change
        assert change.previous_provider == "chainlink"
        assert change.current_provider == "binance_spot"

    def test_verifier_blocks_on_a_changed_fingerprint(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        spec = spec_of("btc_5m_live_2026_07")
        stale = SettlementVerifier(config, known_hash="0000deadbeef0000")
        result = stale.verify(spec)
        assert result.status is VerificationStatus.REJECTED_PROVIDER_CHANGED
        assert result.trading_enabled is False

    def test_verifier_accepts_a_matching_fingerprint(self, config) -> None:  # type: ignore[no-untyped-def]
        spec = spec_of("btc_5m_live_2026_07")
        assert SettlementVerifier(config, known_hash=spec.spec_hash).verify(spec).trading_enabled

    def test_corrupt_line_does_not_destroy_history(self, tmp_path: Path) -> None:
        path = tmp_path / "specs.jsonl"
        store = SettlementSpecStore(path)
        store.record(spec_of("btc_5m_live_2026_07"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        assert len(SettlementSpecStore(path)) == 1

    def test_historical_replay_uses_the_stored_spec(self, tmp_path: Path) -> None:
        # A backtest asks the store, not the live API, so a provider change
        # today cannot retroactively relabel yesterday's markets.
        store = SettlementSpecStore(tmp_path / "specs.jsonl")
        historical = spec_of("btc_quarterly_binance_2025")
        store.record(historical)
        replayed = store.get(historical.condition_id)
        assert replayed is not None
        assert replayed.provider is SettlementSource.BINANCE_SPOT


class TestReport:
    def test_mixed_report(self, verifier) -> None:  # type: ignore[no-untyped-def]
        report = SettlementReport(
            tuple(
                verifier.verify(spec_of(name))
                for name in ("btc_5m_live_2026_07", "btc_15m_live_2026_07", "btc_hourly_binance")
            )
        )
        assert len(report.verified) == 1
        assert len(report.rejected) == 2
        assert report.all_passed is False
        assert report.any_tradeable is True

    def test_clean_report_passes(self, verifier) -> None:  # type: ignore[no-untyped-def]
        report = SettlementReport((verifier.verify(spec_of("btc_5m_live_2026_07")),))
        assert report.all_passed is True

    def test_empty_scan_is_not_a_pass(self) -> None:
        # "Found nothing" must never read as "all clear".
        assert SettlementReport(()).all_passed is False

    def test_rows_carry_every_agreed_field(self, verifier) -> None:  # type: ignore[no-untyped-def]
        row = SettlementReport((verifier.verify(spec_of("btc_5m_live_2026_07")),)).rows()[0]
        for key in (
            "market_id",
            "resolution_source",
            "provider",
            "venue",
            "trading_pair",
            "interval",
            "settlement_timestamp",
            "confidence",
            "status",
            "trading_enabled",
        ):
            assert row[key], f"{key} empty"
        assert row["trading_enabled"] == "YES"
        assert row["settlement_timestamp"] == "2026-07-31T13:10:00.000Z"

    def test_rejected_row_names_the_failure(self, verifier) -> None:  # type: ignore[no-untyped-def]
        row = SettlementReport((verifier.verify(spec_of("btc_hourly_binance")),)).rows()[0]
        assert row["trading_enabled"] == "NO"
        assert "provider_matches_config" in row["failures"]

    def test_render_and_log_do_not_raise(self, tmp_config, verifier) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.logging_setup import configure_logging

        configure_logging(tmp_config, force=True)
        report = SettlementReport((verifier.verify(spec_of("btc_5m_live_2026_07")),))
        report.render()
        report.emit_log()
