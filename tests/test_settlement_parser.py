"""Parser tests against **real archived Gamma payloads**.

Fixtures in ``tests/fixtures/gamma`` were captured from the live API on
2026-07-31 and span two providers, three intervals, and four rule dialects:

===============================  ========  ========  =========================
fixture                          provider  interval  why it is here
===============================  ========  ========  =========================
btc_5m_live_2026_07              Chainlink  5m       the instrument we trade
btc_5m_resolved_2026_04          Chainlink  5m       same family, 4 months older
btc_15m_live_2026_07             Chainlink  15m      right provider, wrong size
btc_hourly_binance               Binance    1h       *same product line, other
                                                     provider* -- the trap
btc_quarterly_binance_2025       Binance    -        prose-only, 50-50 tie rule
===============================  ========  ========  =========================

The hourly fixture is the important one. "Bitcoin Up or Down" markets exist that
resolve off Binance BTCUSDT while the 5-minute ones resolve off Chainlink
BTC/USD. Any parser that trusts the title or the slug family trades the wrong
series eventually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pmbtc.constants import SettlementSource
from pmbtc.settlement.parser import SettlementParseError, parse_settlement_spec
from pmbtc.settlement.spec import EvidenceSource, TieRule

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gamma"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _strip_structured_source(market: dict[str, Any]) -> dict[str, Any]:
    """Deep copy with resolutionSource cleared on *both* market and event.

    Gamma mirrors the field onto the embedded event, so a shallow copy leaves
    the structured evidence intact and the test silently exercises the wrong
    code path.
    """
    copy = json.loads(json.dumps(market))
    copy["resolutionSource"] = ""
    for event in copy.get("events") or []:
        event["resolutionSource"] = ""
    return copy


@pytest.fixture
def btc_5m() -> dict[str, Any]:
    return load("btc_5m_live_2026_07")


class TestChainlink5m:
    """The market the bot actually trades."""

    def test_provider_is_chainlink(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        assert spec.provider is SettlementSource.CHAINLINK
        assert spec.venue == "Chainlink"
        assert spec.resolution_source_url == "https://data.chain.link/streams/btc-usd"

    def test_provider_is_corroborated_not_merely_matched(self, btc_5m: dict[str, Any]) -> None:
        # Structured resolutionSource AND the prose must independently agree
        # before the field can reach full confidence.
        evidence = parse_settlement_spec(btc_5m).evidence["provider"]
        assert evidence.source is EvidenceSource.STRUCTURED_FIELD
        assert evidence.confidence == 1.0
        assert not evidence.conflict
        assert "resolutionSource" in evidence.locator and "description" in evidence.locator

    def test_trading_pair(self, btc_5m: dict[str, Any]) -> None:
        assert parse_settlement_spec(btc_5m).trading_pair == "BTC/USD"

    def test_interval_from_series_recurrence(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        assert spec.interval_seconds == 300
        assert spec.evidence["interval_seconds"].confidence == 1.0

    def test_window_times_are_utc_millis(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        # slug btc-updown-5m-1785503100 -> 2026-07-31T13:05:00Z, endDate 13:10:00Z
        assert spec.window_open_ms == 1785503100 * 1000
        assert spec.window_close_ms == 1785503400 * 1000
        assert spec.duration_seconds == 300
        assert spec.timezone == "UTC"

    def test_slug_epoch_corroborates_start_time(self, btc_5m: dict[str, Any]) -> None:
        assert parse_settlement_spec(btc_5m).evidence["window_open_ms"].confidence == 1.0

    def test_settlement_timestamp_alias(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        assert spec.settlement_timestamp_ms == spec.window_close_ms

    def test_tie_resolves_up(self, btc_5m: dict[str, Any]) -> None:
        # "greater than or equal to" -- an exact tie is a win for Up.
        assert parse_settlement_spec(btc_5m).tie_rule is TieRule.TIE_UP

    def test_reference_rule(self, btc_5m: dict[str, Any]) -> None:
        assert parse_settlement_spec(btc_5m).reference_rule == "range_open_price"

    def test_rounding_is_provider_declared_not_invented(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        evidence = spec.evidence["price_decimals"]
        assert evidence.source is EvidenceSource.PROVIDER_DEFAULT
        assert spec.price_decimals == 18
        assert spec.rounding_mode == "half_even"

    def test_market_mechanics_captured(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        assert spec.outcomes == ("Up", "Down")
        assert spec.tick_size == 0.01
        assert spec.min_order_size == 5
        assert spec.fees_enabled is True

    def test_series_identity(self, btc_5m: dict[str, Any]) -> None:
        assert parse_settlement_spec(btc_5m).series_slug == "btc-up-or-down-5m"

    def test_fully_confident(self, btc_5m: dict[str, Any]) -> None:
        spec = parse_settlement_spec(btc_5m)
        assert spec.confidence == 1.0
        assert spec.conflicts == {}
        assert spec.missing_fields == ()


class TestHistoricalStability:
    """Same family, four months apart, must parse identically."""

    def test_april_market_matches_july_mechanics(self) -> None:
        april = parse_settlement_spec(load("btc_5m_resolved_2026_04"))
        july = parse_settlement_spec(load("btc_5m_live_2026_07"))
        assert april.provider is july.provider
        assert april.tie_rule is july.tie_rule
        assert april.interval_seconds == july.interval_seconds
        # Identical settlement mechanics must produce an identical fingerprint,
        # so a real change in the family is unmistakable.
        assert april.spec_hash == july.spec_hash

    def test_tick_size_did_change_between_eras(self) -> None:
        # Mechanics are stable; market parameters are not. This is why tick size
        # is read per-market rather than configured.
        april = parse_settlement_spec(load("btc_5m_resolved_2026_04"))
        july = parse_settlement_spec(load("btc_5m_live_2026_07"))
        assert april.tick_size == 0.001
        assert july.tick_size == 0.01

    def test_april_window(self) -> None:
        spec = parse_settlement_spec(load("btc_5m_resolved_2026_04"))
        assert spec.window_open_ms == 1775181000 * 1000
        assert spec.duration_seconds == 300
        assert spec.confidence == 1.0


class TestProviderDivergenceInsideOneProductLine:
    """The failure mode this whole module exists to prevent."""

    def test_hourly_sibling_resolves_off_binance(self) -> None:
        spec = parse_settlement_spec(load("btc_hourly_binance"))
        assert spec.provider is SettlementSource.BINANCE_SPOT
        assert spec.venue == "Binance"
        assert spec.trading_pair == "BTC/USDT"

    def test_hourly_sibling_has_different_fingerprint(self) -> None:
        hourly = parse_settlement_spec(load("btc_hourly_binance"))
        five_min = parse_settlement_spec(load("btc_5m_live_2026_07"))
        assert hourly.spec_hash != five_min.spec_hash

    def test_hourly_interval_is_3600s(self) -> None:
        assert parse_settlement_spec(load("btc_hourly_binance")).interval_seconds == 3_600

    def test_15m_sibling_right_provider_wrong_interval(self) -> None:
        spec = parse_settlement_spec(load("btc_15m_live_2026_07"))
        assert spec.provider is SettlementSource.CHAINLINK
        assert spec.interval_seconds == 900
        assert spec.confidence == 1.0  # correctly parsed; the verifier rejects it


class TestProseOnlyMarket:
    """Older markets publish no structured resolutionSource at all."""

    def test_provider_detected_from_text_only(self) -> None:
        spec = parse_settlement_spec(load("btc_quarterly_binance_2025"))
        assert spec.provider is SettlementSource.BINANCE_SPOT
        assert spec.evidence["provider"].source is EvidenceSource.TEXT_PATTERN

    def test_text_only_cannot_reach_the_trading_gate(self) -> None:
        # Capped at 0.7 by design: prose alone is never sufficient to trade.
        spec = parse_settlement_spec(load("btc_quarterly_binance_2025"))
        assert spec.evidence["provider"].confidence == 0.7
        assert spec.confidence < 1.0

    def test_fifty_fifty_tie_rule_detected(self) -> None:
        # A different tie dialect entirely -- exact equality voids to 50-50
        # rather than resolving Up.
        spec = parse_settlement_spec(load("btc_quarterly_binance_2025"))
        assert spec.tie_rule is TieRule.TIE_FIFTY_FIFTY


class TestConflictDetection:
    def test_structured_and_prose_disagreement_is_a_conflict(
        self, btc_5m: dict[str, Any]
    ) -> None:
        # Simulate Gamma metadata drifting away from the published rules.
        tampered = dict(btc_5m)
        tampered["resolutionSource"] = "https://www.binance.com/en/trade/BTC_USDT"
        spec = parse_settlement_spec(tampered)
        assert spec.provider is SettlementSource.UNKNOWN
        assert spec.confidence == 0.0
        assert "provider" in spec.conflicts
        assert "Binance" in spec.conflicts["provider"]
        assert "Chainlink" in spec.conflicts["provider"]

    def test_prose_naming_two_providers_is_a_conflict(self, btc_5m: dict[str, Any]) -> None:
        tampered = _strip_structured_source(btc_5m)
        tampered["description"] = (
            "Resolves per Chainlink data stream BTC/USD and also per Binance BTCUSDT."
        )
        spec = parse_settlement_spec(tampered)
        assert spec.provider is SettlementSource.UNKNOWN
        assert "multiple providers" in spec.conflicts["provider"]

    def test_structured_source_is_read_from_the_event_when_absent_on_the_market(
        self, btc_5m: dict[str, Any]
    ) -> None:
        # Gamma publishes resolutionSource on both objects; either is authoritative.
        tampered = json.loads(json.dumps(btc_5m))
        tampered["resolutionSource"] = ""
        spec = parse_settlement_spec(tampered)
        assert spec.provider is SettlementSource.CHAINLINK
        assert spec.confidence == 1.0

    def test_slug_epoch_disagreeing_with_start_time_is_a_conflict(
        self, btc_5m: dict[str, Any]
    ) -> None:
        tampered = json.loads(json.dumps(btc_5m))
        tampered["events"][0]["startTime"] = "2026-07-31T14:05:00Z"  # one hour out
        spec = parse_settlement_spec(tampered)
        assert spec.evidence["window_open_ms"].confidence == 0.0
        assert "window_open_ms" in spec.conflicts

    def test_interval_sources_disagreeing_is_a_conflict(self, btc_5m: dict[str, Any]) -> None:
        tampered = json.loads(json.dumps(btc_5m))
        tampered["events"][0]["series"][0]["recurrence"] = "15m"
        spec = parse_settlement_spec(tampered)
        assert spec.interval_seconds == 0
        assert "interval_seconds" in spec.conflicts

    def test_unknown_provider_yields_no_rounding_rule(self, btc_5m: dict[str, Any]) -> None:
        tampered = _strip_structured_source(btc_5m)
        tampered["description"] = "Resolves somehow."
        spec = parse_settlement_spec(tampered)
        assert spec.provider is SettlementSource.UNKNOWN
        assert "price_decimals" in spec.missing_fields


class TestMissingData:
    def test_no_condition_id_is_not_a_market(self) -> None:
        with pytest.raises(SettlementParseError):
            parse_settlement_spec({"slug": "whatever"})

    def test_missing_tie_clause(self, btc_5m: dict[str, Any]) -> None:
        tampered = dict(btc_5m)
        tampered["description"] = (
            "The resolution source for this market is information from Chainlink, "
            "specifically the BTC/USD data stream available at "
            "https://data.chain.link/streams/btc-usd."
        )
        spec = parse_settlement_spec(tampered)
        assert spec.tie_rule is TieRule.UNKNOWN
        assert "tie_rule" in spec.missing_fields

    def test_missing_end_date(self, btc_5m: dict[str, Any]) -> None:
        tampered = json.loads(json.dumps(btc_5m))
        tampered.pop("endDate")
        tampered["events"][0].pop("endDate", None)
        spec = parse_settlement_spec(tampered)
        assert spec.window_close_ms == 0
        assert "window_close_ms" in spec.missing_fields

    def test_outcomes_json_string_is_decoded(self, btc_5m: dict[str, Any]) -> None:
        assert parse_settlement_spec(btc_5m).outcomes == ("Up", "Down")
