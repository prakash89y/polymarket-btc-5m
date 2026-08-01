"""Schema drift, lifecycle, health, liquidity, discovery, and replay.

Built on the same real archived payloads as the settlement tests, so every
assertion here is about behaviour on data Polymarket actually served.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pmbtc.clock import ClockSample, ClockService
from pmbtc.config import load_config
from pmbtc.gamma import (
    DriftSeverity,
    HealthIssue,
    LifecycleTracker,
    LiquidityStore,
    MarketDiscovery,
    MarketHealthChecker,
    MarketState,
    ReplaySession,
    SchemaChecker,
    SchemaRegistry,
    SchemaViolation,
    classify,
    interpretation_of,
    snapshot_from_market,
)
from pmbtc.settlement import SettlementSpecStore, parse_settlement_spec
from pmbtc.utils.timeutils import utc_now_ms

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gamma"
SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def config(tmp_path: Path):  # type: ignore[no-untyped-def]
    # base_dir must be the temp dir: discovery persists learned slug hints and
    # the schema registry under app.data_dir, and tests that leak those into the
    # repo's real data/ directory contaminate each other.
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})


@pytest.fixture
def btc_5m() -> dict[str, Any]:
    return load("btc_5m_live_2026_07")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class TestSchema:
    @pytest.fixture
    def checker(self, tmp_path: Path) -> SchemaChecker:
        return SchemaChecker(SchemaRegistry(tmp_path / "schema.json"))

    def test_real_payload_is_compatible(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        result = checker.check(btc_5m)
        assert result.breaking == ()
        assert result.changed == ()
        assert result.compatible

    def test_versions_are_assigned_and_remembered(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        first = checker.check(btc_5m)
        second = checker.check(btc_5m)
        assert first.version == second.version
        assert first.known is False
        assert second.known is True

    def test_different_shapes_get_different_versions(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        other = dict(btc_5m)
        other["brandNewField"] = 1
        assert checker.check(btc_5m).version != checker.check(other).version

    def test_removed_required_field_is_breaking(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        broken = dict(btc_5m)
        broken.pop("endDate")
        result = checker.check(broken)
        assert any(d.field_name == "endDate" for d in result.breaking)
        assert not result.compatible

    def test_retyped_field_is_detected(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        broken = dict(btc_5m)
        broken["orderMinSize"] = "5"  # number -> string
        result = checker.check(broken)
        assert any(d.field_name == "orderMinSize" for d in result.drifts)

    def test_new_field_is_additive_only(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        # Polymarket adds fields routinely; halting on that would be a
        # self-inflicted outage.
        extended = dict(btc_5m)
        extended["somethingNew"] = "x"
        result = checker.check(extended)
        assert result.compatible
        assert any(d.severity is DriftSeverity.ADDITIVE for d in result.additive)

    def test_check_or_raise_blocks_incompatible_payloads(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        broken = dict(btc_5m)
        broken.pop("resolutionSource")
        with pytest.raises(SchemaViolation) as exc:
            checker.check_or_raise(broken)
        assert exc.value.halts_trading is True

    def test_check_or_raise_passes_good_payloads(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        assert checker.check_or_raise(btc_5m).compatible

    def test_missing_series_is_breaking(
        self, checker: SchemaChecker, btc_5m: dict[str, Any]
    ) -> None:
        broken = json.loads(json.dumps(btc_5m))
        broken["events"][0].pop("series")
        assert any(d.field_name == "series" for d in checker.check(broken).breaking)

    def test_registry_persists_across_restarts(
        self, tmp_path: Path, btc_5m: dict[str, Any]
    ) -> None:
        path = tmp_path / "schema.json"
        first = SchemaChecker(SchemaRegistry(path)).check(btc_5m)
        second = SchemaChecker(SchemaRegistry(path)).check(btc_5m)
        assert second.known is True
        assert second.version == first.version


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def test_classification_across_a_window(self) -> None:
        open_ms, close_ms = 1_000_000, 1_300_000  # 300s window
        kwargs = {"settlement_ms": close_ms, "window_open_ms": open_ms, "safety_window_s": 20}
        assert classify(now_ms=open_ms - 60_000, **kwargs) is MarketState.DISCOVERED
        assert classify(now_ms=open_ms + 10_000, **kwargs) is MarketState.OPEN
        assert classify(now_ms=close_ms - 10_000, **kwargs) is MarketState.NEAR_SETTLEMENT
        assert classify(now_ms=close_ms + 1, **kwargs) is MarketState.SETTLED
        assert classify(now_ms=open_ms + 10_000, closed=True, **kwargs) is MarketState.ARCHIVED

    def test_suspended_market_is_not_open(self) -> None:
        assert (
            classify(
                settlement_ms=1_300_000,
                window_open_ms=1_000_000,
                now_ms=1_100_000,
                safety_window_s=20,
                accepting_orders=False,
            )
            is MarketState.DISCOVERED
        )

    def test_only_transitions_are_persisted(self, tmp_path: Path) -> None:
        tracker = LifecycleTracker(tmp_path / "life.jsonl")
        assert tracker.observe("c1", "s1", MarketState.OPEN, 1) is not None
        assert tracker.observe("c1", "s1", MarketState.OPEN, 2) is None
        assert tracker.observe("c1", "s1", MarketState.SETTLED, 3) is not None
        assert len(tracker.history("c1")) == 2

    def test_backwards_transitions_are_refused(self, tmp_path: Path) -> None:
        tracker = LifecycleTracker(tmp_path / "life.jsonl")
        tracker.observe("c1", "s1", MarketState.SETTLED, 1)
        assert tracker.observe("c1", "s1", MarketState.OPEN, 2) is None
        assert tracker.state_of("c1") is MarketState.SETTLED

    def test_skipping_forward_is_allowed(self, tmp_path: Path) -> None:
        # A slow scan can legitimately miss a state.
        tracker = LifecycleTracker(tmp_path / "life.jsonl")
        tracker.observe("c1", "s1", MarketState.DISCOVERED, 1)
        assert tracker.observe("c1", "s1", MarketState.SETTLED, 2) is not None

    def test_history_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "life.jsonl"
        LifecycleTracker(path).observe("c1", "s1", MarketState.OPEN, 1)
        reopened = LifecycleTracker(path)
        assert reopened.state_of("c1") is MarketState.OPEN
        assert reopened.observe("c1", "s1", MarketState.OPEN, 2) is None

    def test_timestamps_for_feature_engineering(self, tmp_path: Path) -> None:
        tracker = LifecycleTracker(tmp_path / "life.jsonl")
        tracker.observe("c1", "s1", MarketState.OPEN, 1_000)
        tracker.observe("c1", "s1", MarketState.SETTLED, 2_000)
        assert tracker.timestamps("c1") == {"open": 1_000, "settled": 2_000}


# --------------------------------------------------------------------------- #
# Liquidity
# --------------------------------------------------------------------------- #
class TestLiquidity:
    def test_snapshot_from_real_payload(self, btc_5m: dict[str, Any]) -> None:
        snapshot = snapshot_from_market(btc_5m, captured_at_ms=1)
        assert snapshot.best_bid == 0.5
        assert snapshot.best_ask == 0.51
        assert snapshot.spread == pytest.approx(0.01)
        assert snapshot.mid == pytest.approx(0.505)
        assert snapshot.two_sided is True
        assert snapshot.liquidity_usdc == pytest.approx(17068.2705)

    def test_mid_is_clamped_away_from_the_boundaries(self) -> None:
        # This value is divided by in the Kelly sizing; 0 would be fatal.
        snapshot = snapshot_from_market(
            {"conditionId": "c", "bestBid": 0.0, "bestAsk": 0.0}, captured_at_ms=1
        )
        assert snapshot.mid is not None
        assert snapshot.mid > 0

    def test_one_sided_book(self) -> None:
        snapshot = snapshot_from_market(
            {"conditionId": "c", "bestBid": 0.4}, captured_at_ms=1
        )
        assert snapshot.two_sided is False
        assert snapshot.mid is None

    def test_string_numbers_are_parsed(self, btc_5m: dict[str, Any]) -> None:
        # Gamma serialises volume and liquidity as strings.
        assert snapshot_from_market(btc_5m, captured_at_ms=1).volume_usdc is not None

    def test_store_round_trip(self, tmp_path: Path, btc_5m: dict[str, Any]) -> None:
        store = LiquidityStore(tmp_path / "liq.jsonl")
        store.write(snapshot_from_market(btc_5m, captured_at_ms=123))
        rows = store.read_all()
        assert len(rows) == 1
        assert rows[0]["mid"] == pytest.approx(0.505)
        assert rows[0]["captured_at_ms"] == 123


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class TestHealth:
    def _check(self, config, market: dict[str, Any], **kwargs: Any):  # type: ignore[no-untyped-def]
        spec = parse_settlement_spec(market)
        snapshot = snapshot_from_market(market, captured_at_ms=0)
        now_ms = kwargs.pop("now_ms", spec.window_open_ms + 60_000)
        return MarketHealthChecker(config).check(
            market, spec, snapshot, now_ms=now_ms, **kwargs
        )

    def test_live_market_is_healthy(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        assert self._check(config, btc_5m).healthy is True

    def test_duplicate_condition_id(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        spec = parse_settlement_spec(btc_5m)
        result = self._check(config, btc_5m, seen_condition_ids={spec.condition_id})
        assert HealthIssue.DUPLICATE_IDENTIFIER in result.issues

    def test_settled_market_is_rejected(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        spec = parse_settlement_spec(btc_5m)
        result = self._check(config, btc_5m, now_ms=spec.window_close_ms + 1_000)
        assert HealthIssue.ALREADY_SETTLED in result.issues

    def test_wrong_recurrence(self, config) -> None:  # type: ignore[no-untyped-def]
        result = self._check(config, load("btc_15m_live_2026_07"))
        assert HealthIssue.UNEXPECTED_RECURRENCE in result.issues

    def test_suspended_trading(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        market = dict(btc_5m)
        market["acceptingOrders"] = False
        assert HealthIssue.TRADING_SUSPENDED in self._check(config, market).issues

    def test_thin_liquidity(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        market = dict(btc_5m)
        market["liquidity"] = "3.0"
        assert HealthIssue.MISSING_LIQUIDITY in self._check(config, market).issues

    def test_one_sided_quote(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        market = dict(btc_5m)
        market.pop("bestAsk")
        assert HealthIssue.ONE_SIDED_QUOTE in self._check(config, market).issues

    def test_wide_spread(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        market = dict(btc_5m)
        market["bestBid"], market["bestAsk"] = 0.2, 0.8
        assert HealthIssue.SPREAD_TOO_WIDE in self._check(config, market).issues

    def test_settlement_ambiguity(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        market = dict(btc_5m)
        market["resolutionSource"] = "https://www.binance.com/en/trade/BTC_USDT"
        assert HealthIssue.SETTLEMENT_AMBIGUITY in self._check(config, market).issues

    def test_unaligned_window(self, config, btc_5m: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        market = json.loads(json.dumps(btc_5m))
        market["slug"] = "btc-updown-5m-1785503130"
        market["events"][0]["startTime"] = "2026-07-31T13:05:30Z"
        market["endDate"] = "2026-07-31T13:10:30Z"
        result = self._check(config, market, now_ms=1785503140 * 1000)
        assert HealthIssue.INVALID_TIMESTAMP in result.issues


# --------------------------------------------------------------------------- #
# Discovery pipeline (network stubbed at the client boundary)
# --------------------------------------------------------------------------- #
class _StubClient:
    """Returns canned events; records which calls the pipeline made."""

    def __init__(
        self, events: list[dict[str, Any]], slug_markets: list[dict[str, Any]] | None = None
    ):
        self.events = events
        self.slug_markets = slug_markets or []
        self.series_calls = 0
        self.slug_calls = 0

    async def events_by_series(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        self.series_calls += 1
        return self.events

    async def markets_by_slug(self, slug: str) -> list[dict[str, Any]]:
        self.slug_calls += 1
        return [m for m in self.slug_markets if m.get("slug") == slug]


def _event_wrapping(market: dict[str, Any]) -> dict[str, Any]:
    """Turn a /markets-shaped fixture into an /events-shaped payload."""
    event = dict((market.get("events") or [{}])[0])
    inner = {k: v for k, v in market.items() if k != "events"}
    event["markets"] = [inner]
    return event


def _discovery(config, tmp_path: Path, client: Any) -> MarketDiscovery:  # type: ignore[no-untyped-def]
    clock = ClockService(config)
    clock.add_sample(ClockSample("binance", 0.0, 50.0, utc_now_ms()))
    return MarketDiscovery(
        config,
        client,  # type: ignore[arg-type]
        clock,
        spec_store=SettlementSpecStore(tmp_path / "specs.jsonl"),
        lifecycle=LifecycleTracker(tmp_path / "life.jsonl"),
        liquidity=LiquidityStore(tmp_path / "liq.jsonl"),
        schema=SchemaChecker(SchemaRegistry(tmp_path / "schema.json")),
    )


class TestDiscovery:
    def _future(self, market: dict[str, Any], seconds_ahead: int = 200) -> dict[str, Any]:
        """Shift a fixture so its window is live relative to now."""
        from pmbtc.utils.timeutils import window_open

        now = utc_now_ms()
        open_ms = window_open(now, 300_000)
        close_ms = open_ms + 300_000
        shifted = json.loads(json.dumps(market))
        shifted["slug"] = f"btc-updown-5m-{open_ms // 1000}"
        shifted["endDate"] = _iso(close_ms)
        shifted["events"][0]["startTime"] = _iso(open_ms)
        shifted["events"][0]["endDate"] = _iso(close_ms)
        del seconds_ahead
        return shifted

    async def test_accepts_a_live_market(self, config, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        market = self._future(btc_5m)
        client = _StubClient([_event_wrapping(market)])
        result = await _discovery(config, tmp_path, client).scan()
        assert result.source == "series"
        assert result.accepted_count == 1
        assert result.candidates[0].state in {MarketState.OPEN, MarketState.NEAR_SETTLEMENT}

    async def test_rejects_the_15m_sibling(self, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        client = _StubClient([_event_wrapping(load("btc_15m_live_2026_07"))])
        result = await _discovery(config, tmp_path, client).scan()
        assert result.accepted_count == 0
        # Health reports every issue it found; the recurrence mismatch is the
        # one that matters here, whichever gate happens to be listed first.
        assert "unexpected_recurrence" in result.rejected[0].detail

    async def test_liquidity_is_captured_for_rejected_markets_too(
        self, config, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        client = _StubClient([_event_wrapping(load("btc_15m_live_2026_07"))])
        await _discovery(config, tmp_path, client).scan()
        assert LiquidityStore(tmp_path / "liq.jsonl").read_all()

    async def test_falls_back_to_slugs_when_the_series_is_empty(
        self, config, tmp_path: Path, btc_5m
    ) -> None:  # type: ignore[no-untyped-def]
        market = self._future(btc_5m)
        client = _StubClient([], slug_markets=[market])
        result = await _discovery(config, tmp_path, client).scan()
        assert client.slug_calls > 0
        assert result.source == "slug_fallback"
        assert result.accepted_count == 1

    async def test_schema_violation_rejects_before_parsing(
        self, config, tmp_path: Path, btc_5m
    ) -> None:  # type: ignore[no-untyped-def]
        market = self._future(btc_5m)
        market.pop("resolutionSource")
        client = _StubClient([_event_wrapping(market)])
        result = await _discovery(config, tmp_path, client).scan()
        assert "schema_incompatible" in result.rejection_reasons

    async def test_duplicate_markets_in_one_scan(self, config, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        market = self._future(btc_5m)
        client = _StubClient([_event_wrapping(market), _event_wrapping(market)])
        result = await _discovery(config, tmp_path, client).scan()
        assert result.accepted_count == 1
        assert "duplicate_identifier" in result.rejection_reasons

    async def test_next_tradeable_prefers_the_soonest_open_market(
        self, config, tmp_path: Path, btc_5m
    ) -> None:  # type: ignore[no-untyped-def]
        market = self._future(btc_5m)
        client = _StubClient([_event_wrapping(market)])
        result = await _discovery(config, tmp_path, client).scan()
        assert result.next_tradeable() is not None

    async def test_slug_shape_is_learned_not_assumed(
        self, config, tmp_path: Path, btc_5m
    ) -> None:  # type: ignore[no-untyped-def]
        market = self._future(btc_5m)
        discovery = _discovery(config, tmp_path, _StubClient([_event_wrapping(market)]))
        await discovery.scan()
        assert discovery.slug_prefix == "btc-updown-5m-"


def _iso(ms: int) -> str:
    from pmbtc.utils.timeutils import isoformat

    return isoformat(ms).replace(".000Z", "Z")


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
class TestReplay:
    def _archive(self, tmp_path: Path, markets: list[dict[str, Any]]) -> Path:
        root = tmp_path / "archive"
        day = root / "2026-07-31"
        day.mkdir(parents=True)
        (day / "markets-1.json").write_text(
            json.dumps({"captured_at_ms": 1, "endpoint": "/markets", "payload": markets}),
            encoding="utf-8",
        )
        return root

    def test_interpretation_is_stable_across_runs(self, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        root = self._archive(tmp_path, [btc_5m])
        session = ReplaySession(root)
        baseline = tmp_path / "baseline.json"
        assert session.write_baseline(baseline) == 1
        report = session.compare_to_baseline(baseline)
        assert report.stable is True
        assert report.parsed == 1

    def test_a_changed_interpretation_is_caught(self, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        # Simulates a parser change altering how a historical payload reads.
        root = self._archive(tmp_path, [btc_5m])
        session = ReplaySession(root)
        baseline = tmp_path / "baseline.json"
        session.write_baseline(baseline)

        data = json.loads(baseline.read_text(encoding="utf-8"))
        only = next(iter(data["markets"]))
        data["markets"][only]["tie_rule"] = "tie_down"
        baseline.write_text(json.dumps(data), encoding="utf-8")

        report = session.compare_to_baseline(baseline)
        assert report.stable is False
        assert any(d.field_name == "tie_rule" for d in report.differences)

    def test_new_markets_do_not_break_stability(self, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        root = self._archive(tmp_path, [btc_5m])
        session = ReplaySession(root)
        baseline = tmp_path / "baseline.json"
        session.write_baseline(baseline)

        (root / "2026-07-31" / "markets-2.json").write_text(
            json.dumps(
                {
                    "captured_at_ms": 2,
                    "endpoint": "/markets",
                    "payload": [load("btc_5m_resolved_2026_04")],
                }
            ),
            encoding="utf-8",
        )
        report = session.compare_to_baseline(baseline)
        assert report.stable is True
        assert len(report.new_markets) == 1

    def test_event_shaped_archives_are_understood(self, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        root = self._archive(tmp_path, [_event_wrapping(btc_5m)])
        assert len(ReplaySession(root).iter_markets()) == 1

    def test_interpretation_excludes_volatile_fields(self, btc_5m) -> None:  # type: ignore[no-untyped-def]
        # A baseline that fails on a timestamp trains everyone to ignore it.
        entry = interpretation_of(parse_settlement_spec(btc_5m))
        assert "detected_at_ms" not in entry
        assert entry["spec_hash"]

    def test_duplicate_captures_collapse_to_the_latest(self, tmp_path: Path, btc_5m) -> None:  # type: ignore[no-untyped-def]
        root = self._archive(tmp_path, [btc_5m, btc_5m])
        assert len(ReplaySession(root).iter_markets()) == 1


class TestClientQuerySemantics:
    """Regression for a defect Module 4 validation exposed.

    Gamma's ``/markets`` excludes closed markets unless ``closed=true`` is
    passed explicitly. An unfiltered lookup therefore cannot see a settled
    market — exactly the ones label backfill needs — so the client must ask for
    the closed set first.
    """

    class _RecordingClient:
        def __init__(self, closed_market: dict[str, Any] | None) -> None:
            self.closed_market = closed_market
            self.calls: list[dict[str, Any]] = []

        async def _request(self, _base: str, _endpoint: str, params: dict[str, Any]):  # type: ignore[no-untyped-def]
            self.calls.append(params)
            if params.get("closed") == "true" and self.closed_market is not None:
                return [self.closed_market]
            return []

    async def test_closed_set_is_queried_first(self, config, btc_5m) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.gamma.client import GammaClient

        client = GammaClient(config)
        recorder = self._RecordingClient(btc_5m)
        client._request = recorder._request  # type: ignore[method-assign]

        found = await client.market_by_condition_id("0xabc")
        assert found is not None
        assert recorder.calls[0]["closed"] == "true"

    async def test_falls_back_to_open_markets(self, config) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.gamma.client import GammaClient

        client = GammaClient(config)
        recorder = self._RecordingClient(None)
        client._request = recorder._request  # type: ignore[method-assign]

        assert await client.market_by_condition_id("0xabc") is None
        assert len(recorder.calls) == 2
        assert "closed" not in recorder.calls[1]

    async def test_explicit_closed_flag_issues_one_query(self, config, btc_5m) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.gamma.client import GammaClient

        client = GammaClient(config)
        recorder = self._RecordingClient(btc_5m)
        client._request = recorder._request  # type: ignore[method-assign]

        await client.market_by_condition_id("0xabc", closed=True)
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["closed"] == "true"


class TestCommittedBaseline:
    """The regression guarantee, enforced by the normal test run.

    ``tests/fixtures/archive`` holds real captured Gamma responses and
    ``replay_baseline.json`` records how this parser reads them. If a parser
    change alters any historical interpretation, this test fails with the market
    and field named. Accepting the change means regenerating the baseline
    (``pmbtc replay --update``) -- a reviewable diff, not a silent behaviour
    change.
    """

    ARCHIVE = Path(__file__).resolve().parent / "fixtures" / "archive"
    BASELINE = Path(__file__).resolve().parent / "fixtures" / "replay_baseline.json"

    def test_fixtures_are_present(self) -> None:
        assert self.ARCHIVE.is_dir(), "archived Gamma responses are missing"
        assert self.BASELINE.is_file(), "replay baseline is missing"

    def test_parser_reinterprets_history_identically(self) -> None:
        report = ReplaySession(self.ARCHIVE).compare_to_baseline(self.BASELINE)
        assert report.parsed > 0
        assert report.stable, "\n".join(str(d) for d in report.differences)

    def test_baseline_covers_a_meaningful_sample(self) -> None:
        report = ReplaySession(self.ARCHIVE).compare_to_baseline(self.BASELINE)
        assert report.parsed >= 10
        assert report.missing_markets == []
