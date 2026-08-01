"""Module 8 — the simulator, its metrics, and the deployment gate.

The tests that matter most here are the ones asserting the backtest *refuses* to
flatter itself: no fills without depth, no conclusions from small samples, no
profit for a strategy that only knows what the book already knows.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pmbtc.backtest import (
    BacktestEngine,
    ColumnMap,
    ConstantModel,
    FillModel,
    MarketProbabilityModel,
    compute_metrics,
    evaluate_backtest,
    quote_from_row,
    run_walk_forward,
    slice_recent,
)
from pmbtc.backtest.metrics import max_drawdown, sharpe_ratio
from pmbtc.config import Config
from pmbtc.constants import Outcome, SkipReason
from pmbtc.trading.costs import CostModel, Quote

SETTLEMENT_0 = 1_785_000_000_000
WINDOW_MS = 300_000
HORIZONS = (300, 240, 180, 120, 60, 30, 15, 5)


@pytest.fixture
def config() -> Config:
    return Config()


def make_row(
    index: int,
    horizon: int,
    *,
    label: int = 1,
    bid: float = 0.49,
    ask: float = 0.51,
    depth: float = 500.0,
    latency_ms: int = 200,
) -> dict[str, Any]:
    settlement = SETTLEMENT_0 + index * WINDOW_MS
    return {
        "condition_id": f"market-{index:04d}",
        "slug": f"btc-updown-5m-{settlement // 1000}",
        "horizon_seconds": horizon,
        "settlement_time_ms": settlement,
        "snapshot_time_ms": settlement - horizon * 1000,
        "label": label,
        "official_outcome": "up" if label == 1 else "down",
        "f_ob_best_bid": bid,
        "f_ob_best_ask": ask,
        "f_ob_mid": (bid + ask) / 2.0,
        "f_lq_depth_bid_usdc": depth,
        "f_lq_depth_ask_usdc": depth,
        "lat_ob_best_bid": latency_ms,
        "lat_ob_best_ask": latency_ms,
    }


def make_dataset(n_markets: int = 40, **kwargs: Any) -> list[dict[str, Any]]:
    """``n_markets`` back-to-back windows, alternating outcomes."""
    rows: list[dict[str, Any]] = []
    for i in range(n_markets):
        label = kwargs.pop("label", None)
        outcome = label if label is not None else i % 2
        for horizon in HORIZONS:
            rows.append(make_row(i, horizon, label=outcome, **kwargs))
    return rows


# --------------------------------------------------------------------------- #
# Row plumbing
# --------------------------------------------------------------------------- #
class TestQuoteFromRow:
    def test_reads_the_book_and_its_age(self) -> None:
        quote = quote_from_row(make_row(0, 180, latency_ms=350))
        assert quote.best_bid == pytest.approx(0.49)
        assert quote.best_ask == pytest.approx(0.51)
        assert quote.age_ms == 350
        assert quote.valid

    def test_missing_book_columns_yield_an_invalid_quote(self) -> None:
        assert not quote_from_row({}).valid

    def test_missing_depth_is_zero_not_infinite(self) -> None:
        row = make_row(0, 180)
        del row["f_lq_depth_ask_usdc"]
        assert quote_from_row(row).ask_depth_usdc == 0.0


# --------------------------------------------------------------------------- #
# Fills
# --------------------------------------------------------------------------- #
class TestFillModel:
    def test_pessimistic_fill_refuses_an_unknown_book(self, config: Config) -> None:
        """A backtest that cannot see the depth does not get to assume it."""
        model = FillModel(CostModel(config.costs), pessimistic=True)
        quote = Quote(best_bid=0.49, best_ask=0.51, bid_depth_usdc=0.0, ask_depth_usdc=0.0)
        outcome = model.execute(outcome=Outcome.UP, quote=quote, stake_usdc=50.0)
        assert not outcome.filled
        assert outcome.skip_reason is SkipReason.INSUFFICIENT_LIQUIDITY

    def test_stake_is_capped_by_resting_depth(self, config: Config) -> None:
        model = FillModel(CostModel(config.costs), pessimistic=True)
        quote = Quote(best_bid=0.49, best_ask=0.51, ask_depth_usdc=25.0, bid_depth_usdc=25.0)
        outcome = model.execute(outcome=Outcome.UP, quote=quote, stake_usdc=200.0)
        assert outcome.fill is not None
        assert outcome.fill.notional_usdc == pytest.approx(25.0)
        assert outcome.fill.partial

    def test_mid_fills_are_strictly_cheaper_than_touch(self, config: Config) -> None:
        """The gap is the honest cost of assuming we were the maker."""
        quote = Quote(best_bid=0.49, best_ask=0.51, ask_depth_usdc=500.0, bid_depth_usdc=500.0)
        touch = FillModel(CostModel(config.costs), style="touch")
        mid = FillModel(CostModel(config.costs), style="mid")
        aggressive = FillModel(CostModel(config.costs), style="aggressive")
        p_mid = mid.price_for(Outcome.UP, quote)
        p_touch = touch.price_for(Outcome.UP, quote)
        p_agg = aggressive.price_for(Outcome.UP, quote)
        assert p_mid is not None and p_touch is not None and p_agg is not None
        assert p_mid < p_touch < p_agg


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class TestBacktestEngine:
    def test_the_market_model_places_no_trades(self, config: Config) -> None:
        """The null control. A strategy that forecasts exactly what the book
        forecasts has no edge by construction; if it trades, the edge
        calculation is wrong."""
        engine = BacktestEngine(config)
        result = engine.run(make_dataset(), MarketProbabilityModel(), model_name="market")
        assert result.evaluated == 40
        assert result.trades == []
        assert result.ending_bankroll_usdc == result.starting_bankroll_usdc

    def test_every_window_is_recorded_traded_or_not(self, config: Config) -> None:
        """No survivorship: 40 markets in, 40 windows out."""
        result = BacktestEngine(config).run(make_dataset(), MarketProbabilityModel())
        assert result.evaluated == 40
        assert sum(result.skip_histogram().values()) == 40

    def test_default_horizon_respects_the_timing_gates(self, config: Config) -> None:
        """window=300, min_into=45, min_to_settlement=30 -> horizons 30..255.
        The latest permitted snapshot in the grid is 30 seconds out."""
        result = BacktestEngine(config).run(make_dataset(), MarketProbabilityModel())
        assert result.decision_horizon_seconds == 30

    def test_an_always_up_model_wins_on_always_up_data(self, config: Config) -> None:
        rows = [make_row(i, h, label=1) for i in range(40) for h in HORIZONS]
        result = BacktestEngine(config).run(rows, ConstantModel(0.95), model_name="always-up")
        assert len(result.trades) > 0
        assert all(w.won for w in result.trades)
        assert result.ending_bankroll_usdc > result.starting_bankroll_usdc

    def test_an_always_up_model_loses_on_always_down_data(self, config: Config) -> None:
        rows = [make_row(i, h, label=0) for i in range(40) for h in HORIZONS]
        result = BacktestEngine(config).run(rows, ConstantModel(0.95), model_name="always-up")
        assert len(result.trades) > 0
        assert not any(w.won for w in result.trades)
        assert result.ending_bankroll_usdc < result.starting_bankroll_usdc

    def test_a_confident_model_stands_down_after_a_losing_streak(
        self, config: Config
    ) -> None:
        """The risk layer must stop a confidently wrong model, not follow it down."""
        rows = [make_row(i, h, label=0) for i in range(40) for h in HORIZONS]
        result = BacktestEngine(config).run(rows, ConstantModel(0.95))
        assert len(result.trades) <= config.risk.max_consecutive_losses + 1
        assert SkipReason.RISK_LIMIT.value in result.skip_histogram()

    def test_the_label_never_reaches_the_decision(self, config: Config) -> None:
        """Flipping every label must not change a single decision — only the
        P&L that follows from it."""
        engine = BacktestEngine(config)
        up = engine.run([make_row(i, h, label=1) for i in range(20) for h in HORIZONS],
                        ConstantModel(0.95))
        down = BacktestEngine(config).run(
            [make_row(i, h, label=0) for i in range(20) for h in HORIZONS],
            ConstantModel(0.95),
        )
        # The first decision is taken before any P&L exists, so it must match.
        assert up.windows[0].decision.as_dict() == down.windows[0].decision.as_dict()
        assert up.windows[0].pnl_usdc != down.windows[0].pnl_usdc

    def test_runs_are_deterministic(self, config: Config) -> None:
        rows = make_dataset()
        first = BacktestEngine(config).run(rows, ConstantModel(0.8))
        second = BacktestEngine(config).run(list(reversed(rows)), ConstantModel(0.8))
        assert [w.as_dict() for w in first.windows] == [w.as_dict() for w in second.windows]

    def test_thin_books_are_refused_not_filled(self, config: Config) -> None:
        rows = make_dataset(depth=1.0)
        result = BacktestEngine(config).run(rows, ConstantModel(0.95))
        assert result.trades == []
        assert SkipReason.INSUFFICIENT_LIQUIDITY.value in result.skip_histogram()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_max_drawdown(self) -> None:
        assert max_drawdown([100.0, 120.0, 60.0, 90.0]) == pytest.approx(0.5)

    def test_sharpe_of_a_constant_return_is_zero(self) -> None:
        assert sharpe_ratio([0.1] * 10, periods_per_year=100.0) == 0.0

    def test_breakeven_win_rate_is_the_average_price_paid(self, config: Config) -> None:
        """A contract bought at ~0.515 must win ~51.5% of the time to break even —
        which is the whole reason a 50/50 market is not a free bet."""
        rows = [make_row(i, h, label=i % 2) for i in range(40) for h in HORIZONS]
        result = BacktestEngine(config).run(rows, ConstantModel(0.95))
        metrics = compute_metrics(result)
        assert metrics.trades > 0
        assert metrics.breakeven_win_rate == pytest.approx(metrics.avg_entry_price)
        assert metrics.breakeven_win_rate > 0.5

    def test_forecast_metrics_cover_every_window_not_just_trades(
        self, config: Config
    ) -> None:
        result = BacktestEngine(config).run(make_dataset(), MarketProbabilityModel())
        metrics = compute_metrics(result)
        assert metrics.trades == 0
        assert metrics.evaluated_windows == 40
        assert not math.isnan(metrics.brier)

    def test_no_trades_yields_a_flat_equity_curve(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_dataset(), MarketProbabilityModel())
        metrics = compute_metrics(result)
        assert metrics.net_pnl_usdc == pytest.approx(0.0)
        assert metrics.equity_curve == [result.starting_bankroll_usdc]

    def test_profit_factor_without_losses_is_finite(self, config: Config) -> None:
        rows = [make_row(i, h, label=1) for i in range(10) for h in HORIZONS]
        metrics = compute_metrics(BacktestEngine(config).run(rows, ConstantModel(0.95)))
        assert metrics.gross_loss_usdc == pytest.approx(0.0)
        assert metrics.profit_factor == pytest.approx(metrics.gross_profit_usdc)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
class TestDeploymentGate:
    def test_a_small_sample_is_inconclusive_however_good(self, config: Config) -> None:
        """Ten perfect trades is not evidence. This is the check that matters
        while the dataset is still hours old."""
        rows = [make_row(i, h, label=1) for i in range(10) for h in HORIZONS]
        result = BacktestEngine(config).run(rows, ConstantModel(0.95))
        report = evaluate_backtest(config, result)
        assert result.ending_bankroll_usdc > result.starting_bankroll_usdc
        assert not report.conclusive
        assert not report.approved
        assert "sample_size" in [c.name for c in report.failures]

    def test_a_no_trade_run_fails_rather_than_passing_vacuously(
        self, config: Config
    ) -> None:
        result = BacktestEngine(config).run(make_dataset(), MarketProbabilityModel())
        report = evaluate_backtest(config, result)
        assert not report.approved
        assert report.metrics.trades == 0

    def test_the_gate_reports_every_check(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_dataset(), ConstantModel(0.8))
        report = evaluate_backtest(config, result)
        names = {c.name for c in report.checks}
        assert {
            "sample_size", "roi", "profit_factor", "sharpe", "max_drawdown",
            "accuracy", "brier", "beats_market_forecast", "win_rate_significant",
        } <= names

    def test_report_serialises(self, config: Config) -> None:
        import json

        result = BacktestEngine(config).run(make_dataset(), ConstantModel(0.8))
        payload = evaluate_backtest(config, result).as_dict()
        assert json.loads(json.dumps(payload))["model"] == "model"


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
class TestWalkForward:
    def test_slice_recent_uses_the_data_not_the_wall_clock(self) -> None:
        rows = make_dataset(n_markets=40)
        recent = slice_recent(rows, days=1)
        assert len(recent) == len(rows)  # 40 windows span ~3.3 hours
        assert slice_recent([], days=30) == []

    def test_every_lookback_is_gated(self, config: Config) -> None:
        report = run_walk_forward(
            config, make_dataset(), MarketProbabilityModel(), model_name="market"
        )
        assert set(report.windows) == set(config.backtest.windows_days)
        assert not report.approved

    def test_a_longer_lookback_than_the_history_still_evaluates(self) -> None:
        """A 10-year lookback over 3 hours of data is the whole dataset, not an
        error — but it must not silently become a different sample."""
        config = Config(backtest={"windows_days": (1, 3650)})
        rows = make_dataset()
        report = run_walk_forward(config, rows, MarketProbabilityModel())
        assert set(report.windows) == {1, 3650}
        assert report.skipped_days == []
        assert (
            report.windows[1].metrics.evaluated_windows
            == report.windows[3650].metrics.evaluated_windows
        )


# --------------------------------------------------------------------------- #
# Column mapping
# --------------------------------------------------------------------------- #
def test_column_map_matches_the_live_feature_names() -> None:
    """The engine reads the real registry's names, not invented ones.

    If a feature is renamed, this fails here rather than silently backtesting
    against an empty book.
    """
    from pmbtc.features.engine import FeatureEngine

    names = set(FeatureEngine().feature_names)
    columns = ColumnMap()
    for column in (
        columns.best_bid, columns.best_ask, columns.bid_depth, columns.ask_depth
    ):
        assert column.removeprefix("f_") in names
