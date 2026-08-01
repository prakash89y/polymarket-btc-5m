"""Edge decomposition, execution metrics, and strict walk-forward.

The decomposition's whole value is that it reconciles. If the ladder does not
sum to the P&L the ledger actually recorded, it is attributing money to causes
that did not produce it — which is worse than not decomposing at all, because it
looks authoritative. Most of these tests exist to defend that identity.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pmbtc.backtest import (
    BacktestEngine,
    ConstantModel,
    MarketProbabilityModel,
    compute_metrics,
    decompose,
    evaluate_backtest,
    run_strict_walk_forward,
)
from pmbtc.config import Config
from pmbtc.constants import SkipReason
from pmbtc.trading import DecisionEngine, Quote
from pmbtc.trading.decision import shrink_toward_fair

SETTLE0 = 1_785_600_000_000
WINDOW_MS = 300_000
HORIZONS = (300, 240, 180, 120, 60, 30, 15, 5)


@pytest.fixture
def config() -> Config:
    return Config()


def make_rows(
    n_markets: int,
    *,
    label: int | None = None,
    depth: float = 500.0,
    bid: float = 0.49,
    ask: float = 0.51,
    latency_ms: int = 200,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n_markets):
        outcome = label if label is not None else i % 2
        settlement = SETTLE0 + i * WINDOW_MS
        for horizon in HORIZONS:
            rows.append(
                {
                    "condition_id": f"m{i:04d}",
                    "slug": f"btc-updown-5m-{settlement // 1000}",
                    "horizon_seconds": horizon,
                    "settlement_time_ms": settlement,
                    "label": outcome,
                    "official_outcome": "up" if outcome == 1 else "down",
                    "f_ob_best_bid": bid,
                    "f_ob_best_ask": ask,
                    "f_ob_mid": (bid + ask) / 2.0,
                    "f_lq_depth_bid_usdc": depth,
                    "f_lq_depth_ask_usdc": depth,
                    "lat_ob_best_bid": latency_ms,
                    "lat_ob_best_ask": latency_ms,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# The identity
# --------------------------------------------------------------------------- #
class TestReconciliation:
    @pytest.mark.parametrize("label", [0, 1, None])
    @pytest.mark.parametrize("depth", [500.0, 12.0])
    def test_the_ladder_sums_to_the_realised_pnl(
        self, config: Config, label: int | None, depth: float
    ) -> None:
        """Across winning, losing and mixed runs, thick book and thin."""
        result = BacktestEngine(config).run(
            make_rows(40, label=label, depth=depth), ConstantModel(0.9)
        )
        decomposition = decompose(result, config.costs)
        metrics = compute_metrics(result)
        assert decomposition.reconciles()
        assert decomposition.reconciles(metrics.net_pnl_usdc)
        assert decomposition.net_profit_usdc == pytest.approx(
            metrics.net_pnl_usdc, abs=1e-6
        )

    def test_walking_the_lines_by_hand_reaches_net_profit(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40), ConstantModel(0.85))
        d = decompose(result, config.costs)
        walked = (
            d.raw_edge_usdc
            - d.spread_cost_usdc
            - d.slippage_cost_usdc
            - d.fee_cost_usdc
            - d.risk_limit_usdc
            - d.missed_fill_usdc
        )
        assert walked == pytest.approx(d.net_profit_usdc, abs=1e-9)

    def test_a_run_with_no_intents_decomposes_to_nothing(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40), MarketProbabilityModel())
        d = decompose(result, config.costs)
        assert d.intents == 0
        assert d.net_profit_usdc == pytest.approx(0.0)
        assert d.reconciles()


# --------------------------------------------------------------------------- #
# What the lines mean
# --------------------------------------------------------------------------- #
class TestAttributionSemantics:
    @pytest.mark.parametrize("prob", [0.9, 0.1])
    @pytest.mark.parametrize("label", [0, 1, None])
    def test_crossing_the_spread_is_never_a_benefit(
        self, config: Config, prob: float, label: int | None
    ) -> None:
        """Regression test for a real bug found on live data.

        The fair price was recorded as the UP mid regardless of which side was
        bought. A DOWN trade's fair price is ``1 - mid_up``, so the
        decomposition reported crossing the spread as a **profit** of 146 USDC
        on the real dataset. A mechanical cost that comes out negative is always
        a bug in the attribution.

        The bound is ``>= 0`` rather than ``> 0`` deliberately: a losing trade
        forfeits its whole stake whatever it paid, so the entry price only
        changes the winning branch. Spread cost is therefore exactly zero on a
        run with no winners — and never below it.
        """
        result = BacktestEngine(config).run(
            make_rows(40, label=label), ConstantModel(prob)
        )
        d = decompose(result, config.costs)
        assert d.intents > 0
        assert d.spread_cost_usdc >= 0.0
        assert d.slippage_cost_usdc >= 0.0
        assert d.reconciles()

    def test_the_spread_bites_whenever_a_trade_wins(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40, label=1), ConstantModel(0.9))
        d = decompose(result, config.costs)
        assert d.filled > 0
        assert d.spread_cost_usdc > 0.0

    def test_the_fair_price_follows_the_side_being_bought(self, config: Config) -> None:
        """``touch >= mid`` must hold for DOWN as well as UP.

        Checked on every priced window, not only the ones that traded, because
        the pricing is what the bug corrupted.
        """
        result = BacktestEngine(config).run(
            make_rows(20, bid=0.05, ask=0.06), ConstantModel(0.001)
        )
        priced = [w for w in result.windows if w.decision.outcome is not None]
        assert priced
        for window in priced:
            assert window.touch_price >= window.mid_price
            # A DOWN buy on a 0.05/0.06 book is priced near 0.945, not near 0.055.
            assert window.mid_price == pytest.approx(1.0 - 0.055)

    def test_risk_limits_that_block_losers_show_as_a_saving(
        self, config: Config
    ) -> None:
        """A negative cost is information, not a bug: standing down after a
        losing streak on an always-losing tape saves money."""
        result = BacktestEngine(config).run(make_rows(40, label=0), ConstantModel(0.95))
        d = decompose(result, config.costs)
        assert d.blocked_by_risk > 0
        assert d.risk_limit_usdc < 0.0
        assert d.reconciles()

    def test_the_depth_gate_fires_before_the_fill_model_ever_sees_a_thin_book(
        self, config: Config
    ) -> None:
        """Under stock config the decision gate's 200-USDC depth floor is
        stricter than any stake the sizer can produce, so a thin book is an
        abstention rather than a missed fill. The fill model's depth check is a
        second line of defence, not the first."""
        result = BacktestEngine(config).run(
            make_rows(40, label=1, depth=1.0), ConstantModel(0.9)
        )
        d = decompose(result, config.costs)
        assert d.intents == 0
        assert SkipReason.INSUFFICIENT_LIQUIDITY.value in result.skip_histogram()

    def test_depth_below_the_stake_becomes_a_partial_fill(self) -> None:
        """A stake larger than the book is holding.

        Risk limits are lifted deliberately: with stock limits the worst-case
        daily-loss check refuses any stake above 100 USDC long before the book's
        depth becomes the binding constraint, so leaving them in place would
        test the risk engine rather than the fill model.
        """
        config = Config(
            execution={"min_book_depth_usdc": 40.0},
            risk={
                "max_daily_loss_usdc": 100_000.0,
                "max_daily_loss_fraction": 0.5,
                "max_weekly_loss_fraction": 0.8,
                "max_total_exposure_usdc": 100_000.0,
            },
        )
        result = BacktestEngine(config).run(
            make_rows(40, label=1, depth=50.0),
            ConstantModel(0.9),
            starting_bankroll_usdc=10_000.0,
        )
        d = decompose(result, config.costs)
        assert d.partial_fills > 0
        assert d.missed_fill_usdc > 0.0  # edge we wanted but could not buy
        assert d.reconciles()

    def test_an_empty_book_is_attributed_to_missed_fills(self) -> None:
        """With the decision gate's depth floor removed, the fill model is what
        refuses — and that refusal lands on the missed-fill line."""
        config = Config(execution={"min_book_depth_usdc": 0.0})
        result = BacktestEngine(config).run(
            make_rows(40, label=1, depth=0.0), ConstantModel(0.9)
        )
        d = decompose(result, config.costs)
        assert d.intents > 0
        assert d.filled == 0
        assert d.blocked_by_depth > 0
        assert d.net_profit_usdc == pytest.approx(0.0)
        assert d.reconciles()

    def test_raw_edge_exceeds_net_profit_when_costs_are_real(
        self, config: Config
    ) -> None:
        """The point of the whole module: forecast edge is not profit."""
        result = BacktestEngine(config).run(make_rows(40, label=1), ConstantModel(0.9))
        d = decompose(result, config.costs)
        assert d.raw_edge_usdc > d.net_profit_usdc

    def test_render_is_readable_and_flags_a_break(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40), ConstantModel(0.9))
        d = decompose(result, config.costs)
        text = d.render()
        for line in ("raw_edge", "spread_cost", "missed_fills", "net_profit"):
            assert line in text
        assert "does not reconcile" not in text


# --------------------------------------------------------------------------- #
# Execution metrics
# --------------------------------------------------------------------------- #
class TestExecutionMetrics:
    def test_all_required_execution_metrics_are_populated(
        self, config: Config
    ) -> None:
        result = BacktestEngine(config).run(make_rows(40, label=1), ConstantModel(0.9))
        m = compute_metrics(result)
        assert m.trades > 0
        assert m.fill_attempts >= m.trades
        assert 0.0 < m.fill_rate <= 1.0
        assert m.avg_holding_seconds > 0.0
        assert m.avg_quoted_spread == pytest.approx(0.02)
        # Half the 2-cent spread is crossed to reach the touch.
        assert m.avg_spread_paid == pytest.approx(0.01)
        assert m.avg_slippage == pytest.approx(config.costs.slippage)
        assert m.trades_per_day > 0.0

    def test_calibration_and_reliability_come_from_the_model_layer(
        self, config: Config
    ) -> None:
        """Reused from pmbtc.models.calibration rather than reimplemented, so
        the backtest and the promotion gate can never disagree on a Brier."""
        import numpy as np

        from pmbtc.models.calibration import brier_score

        result = BacktestEngine(config).run(make_rows(40), MarketProbabilityModel())
        m = compute_metrics(result)
        labels = np.array([w.label for w in result.windows], dtype=float)
        probs = np.array([w.prob_up for w in result.windows], dtype=float)
        assert m.brier == pytest.approx(brier_score(labels, probs))
        assert m.reliability is not None
        assert m.reliability.samples == len(result.windows)
        assert not math.isnan(m.ece)

    def test_fill_rate_ignores_risk_refusals(self, config: Config) -> None:
        """A trade the risk engine refused never reached the book, so counting
        it as a fill failure would blame the venue for our own limit."""
        result = BacktestEngine(config).run(make_rows(40, label=0), ConstantModel(0.95))
        m = compute_metrics(result)
        risk_blocked = result.skip_histogram().get(SkipReason.RISK_LIMIT.value, 0)
        assert risk_blocked > 0
        assert m.fill_attempts == m.trades  # everything that reached the book filled
        assert m.fill_rate == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The EV gate
# --------------------------------------------------------------------------- #
class TestExpectedValueGate:
    def test_the_gate_requires_positive_ev_after_costs(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40, label=0), ConstantModel(0.95))
        report = evaluate_backtest(config, result)
        check = next(c for c in report.checks if c.name == "positive_ev_after_costs")
        assert not check.passed

    def test_the_gate_checks_that_the_decomposition_reconciles(
        self, config: Config
    ) -> None:
        result = BacktestEngine(config).run(make_rows(40, label=1), ConstantModel(0.9))
        report = evaluate_backtest(config, result)
        check = next(c for c in report.checks if c.name == "decomposition_reconciles")
        assert check.passed

    def test_a_no_intent_run_cannot_pass_the_ev_gate(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40), MarketProbabilityModel())
        report = evaluate_backtest(config, result)
        check = next(c for c in report.checks if c.name == "positive_ev_after_costs")
        assert not check.passed

    def test_the_report_renders_the_decomposition(self, config: Config) -> None:
        result = BacktestEngine(config).run(make_rows(40, label=1), ConstantModel(0.9))
        text = evaluate_backtest(config, result).render()
        assert "edge decomposition" in text
        assert "execution quality" in text
        assert "reliability (model)" in text


# --------------------------------------------------------------------------- #
# Latency and confidence shrinkage
# --------------------------------------------------------------------------- #
class TestLatencyAndShrinkage:
    def _decide(self, config: Config, quote: Quote, prob: float, **kwargs: Any):
        params: dict[str, Any] = {
            "model_prob_up": prob,
            "quote": quote,
            "seconds_into_window": 120.0,
            "seconds_to_settlement": 180.0,
        }
        params.update(kwargs)
        return DecisionEngine(config).decide(**params)

    def test_latency_is_added_to_book_age(self) -> None:
        """A 1.8s-old book is inside the 2s budget until a 0.5s round trip is
        added; then the book we priced against is gone by the time we arrive."""
        config = Config(costs={"assumed_latency_ms": 500})
        quote = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=1_800,
        )
        decision = self._decide(config, quote, 0.9)
        assert not decision.trade
        assert decision.skip_reason is SkipReason.STALE_DATA
        assert "latency" in decision.detail

    def test_zero_latency_restores_the_old_behaviour(self) -> None:
        config = Config(costs={"assumed_latency_ms": 0})
        quote = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=1_800,
        )
        assert self._decide(config, quote, 0.9).trade

    def test_shrinkage_pulls_toward_a_coin_flip(self) -> None:
        assert shrink_toward_fair(0.9, 0.0) == pytest.approx(0.9)
        assert shrink_toward_fair(0.9, 0.5) == pytest.approx(0.7)
        assert shrink_toward_fair(0.9, 1.0) == pytest.approx(0.5)
        assert shrink_toward_fair(0.1, 0.5) == pytest.approx(0.3)

    def test_shrinkage_is_off_by_default(self, config: Config) -> None:
        quote = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0
        )
        decision = self._decide(config, quote, 0.75)
        assert decision.model_prob_up == pytest.approx(decision.raw_prob_up)

    def test_shrinkage_can_turn_a_trade_into_an_abstention(self) -> None:
        """The purpose of the knob: distrusting the model costs trades, which is
        the point rather than a side effect."""
        quote = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0
        )
        trusting = Config()
        assert self._decide(trusting, quote, 0.70).trade
        wary = Config(prediction={"confidence_shrinkage": 0.6})
        shrunk = self._decide(wary, quote, 0.70)
        assert not shrunk.trade
        assert shrunk.raw_prob_up == pytest.approx(0.70)
        assert shrunk.model_prob_up < 0.70


# --------------------------------------------------------------------------- #
# Strict walk-forward
# --------------------------------------------------------------------------- #
class TestStrictWalkForward:
    def test_it_refits_each_fold_and_never_tests_before_training(
        self, config: Config
    ) -> None:
        seen: list[int] = []

        def fit(train_rows):
            seen.append(len(train_rows))
            return MarketProbabilityModel()

        report = run_strict_walk_forward(
            config, make_rows(120), fit, model_name="market", n_folds=3
        )
        assert len(report.folds) == len(seen) > 0
        for fold in report.folds:
            assert fold.train_end_ms < fold.test_start_ms
            assert fold.train_markets > 0 and fold.test_markets > 0

    def test_folds_are_purged_between_train_and_test(self, config: Config) -> None:
        """Adjacent 5-minute windows share microstructure state; the embargo
        comes from TimeSeriesSplitter rather than being re-derived here."""
        report = run_strict_walk_forward(
            config, make_rows(120), lambda _: MarketProbabilityModel(), n_folds=3
        )
        assert any(fold.purged_markets > 0 for fold in report.folds)

    def test_too_little_history_yields_no_folds_rather_than_a_guess(
        self, config: Config
    ) -> None:
        report = run_strict_walk_forward(
            config, make_rows(4), lambda _: MarketProbabilityModel(), n_folds=6
        )
        assert report.folds == []
        assert not report.approved

    def test_the_pooled_record_is_gated(self, config: Config) -> None:
        report = run_strict_walk_forward(
            config, make_rows(120), lambda _: MarketProbabilityModel(), n_folds=3
        )
        assert report.combined is not None
        assert not report.approved  # the null strategy places no trades

    def test_it_is_deterministic(self, config: Config) -> None:
        rows = make_rows(120)
        runs = [
            run_strict_walk_forward(
                config, rows, lambda _: ConstantModel(0.85), n_folds=3
            ).as_dict()
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
