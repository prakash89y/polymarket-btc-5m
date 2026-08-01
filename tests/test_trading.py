"""Module 8 — the shared trading layer: costs, decisions, sizing, risk.

These tests exist to pin down the arithmetic that decides whether the system
makes money. Where a test asserts a specific number, the number is derived in
the docstring rather than copied from a run, so a change in behaviour reads as a
disagreement with the reasoning rather than as a new expected value.
"""

from __future__ import annotations

import pytest

from pmbtc.config import Config
from pmbtc.constants import Outcome, SkipReason
from pmbtc.trading import CostModel, DecisionEngine, PositionSizer, Quote, RiskLedger
from pmbtc.trading.sizing import kelly_fraction


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def quote() -> Quote:
    """A realistic 5m book: 2-cent spread, a few hundred USDC a side."""
    return Quote(
        best_bid=0.49,
        best_ask=0.51,
        bid_depth_usdc=500.0,
        ask_depth_usdc=500.0,
        age_ms=200,
    )


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
class TestCostModel:
    def test_buying_up_lifts_the_ask(self, config: Config, quote: Quote) -> None:
        costs = CostModel(config.costs)
        assert costs.touch_price(Outcome.UP, quote) == pytest.approx(0.51)

    def test_buying_down_pays_the_complement_of_the_bid(
        self, config: Config, quote: Quote
    ) -> None:
        """DOWN's ask is ``1 - bid_up`` = 0.51, not ``1 - ask_up`` = 0.49.

        Getting this backwards would make every DOWN trade look 2 cents cheaper
        than it is — i.e. would invent the spread as profit.
        """
        costs = CostModel(config.costs)
        assert costs.touch_price(Outcome.DOWN, quote) == pytest.approx(0.51)

    def test_both_directions_cross_the_spread(self, config: Config, quote: Quote) -> None:
        costs = CostModel(config.costs)
        mid = quote.mid
        assert mid is not None
        for outcome in (Outcome.UP, Outcome.DOWN):
            touch = costs.touch_price(outcome, quote)
            assert touch is not None and touch > mid

    def test_entry_price_adds_slippage(self, config: Config, quote: Quote) -> None:
        costs = CostModel(config.costs)
        expected = 0.51 + config.costs.slippage
        assert costs.entry_price(Outcome.UP, quote) == pytest.approx(expected)

    def test_round_trip_is_one_leg_when_holding_to_settlement(
        self, config: Config, quote: Quote
    ) -> None:
        """Half-spread (0.01) + slippage, charged once rather than twice."""
        costs = CostModel(config.costs)
        assert config.costs.assume_hold_to_settlement is True
        assert costs.round_trip_cost(quote) == pytest.approx(0.01 + config.costs.slippage)

    def test_one_sided_and_crossed_books_are_invalid(self) -> None:
        assert not Quote(best_bid=0.49, best_ask=None).valid
        assert not Quote(best_bid=0.52, best_ask=0.51).valid
        assert Quote(best_bid=0.49, best_ask=0.51).valid

    def test_depth_for_down_is_the_bid_side(self, quote: Quote) -> None:
        """A DOWN buy is filled by the resting UP bid, so bid depth is what binds."""
        q = Quote(best_bid=0.49, best_ask=0.51, bid_depth_usdc=10.0, ask_depth_usdc=900.0)
        assert q.depth_for(Outcome.DOWN) == 10.0
        assert q.depth_for(Outcome.UP) == 900.0

    def test_fill_payoff_and_pnl(self, config: Config, quote: Quote) -> None:
        """100 USDC at 0.515 buys ~194.17 shares; winning returns ~194.17."""
        costs = CostModel(config.costs)
        fill = costs.fill(Outcome.UP, quote, 100.0)
        assert fill is not None
        assert fill.price == pytest.approx(0.515)
        assert fill.shares == pytest.approx(100.0 / 0.515)
        assert fill.pnl_usdc(Outcome.UP) == pytest.approx(fill.shares - 100.0)
        assert fill.pnl_usdc(Outcome.DOWN) == pytest.approx(-100.0)


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
class TestDecisionEngine:
    def _decide(self, config: Config, quote: Quote, prob: float, **kwargs: object):
        engine = DecisionEngine(config)
        params: dict = {
            "model_prob_up": prob,
            "quote": quote,
            "seconds_into_window": 120.0,
            "seconds_to_settlement": 180.0,
        }
        params.update(kwargs)
        return engine.decide(**params)  # type: ignore[arg-type]

    def test_the_market_price_alone_is_never_a_trade(
        self, config: Config, quote: Quote
    ) -> None:
        """Forecasting the mid exactly means zero edge — the null control."""
        decision = self._decide(config, quote, 0.5)
        assert not decision.trade

    def test_a_large_edge_trades(self, config: Config, quote: Quote) -> None:
        decision = self._decide(config, quote, 0.75)
        assert decision.trade
        assert decision.outcome is Outcome.UP
        assert decision.entry_price == pytest.approx(0.515)
        assert decision.edge == pytest.approx(0.75 - 0.515)

    def test_a_large_downside_edge_trades_down(self, config: Config, quote: Quote) -> None:
        decision = self._decide(config, quote, 0.25)
        assert decision.trade
        assert decision.outcome is Outcome.DOWN
        assert decision.confidence == pytest.approx(0.75)

    def test_edge_is_measured_against_the_price_paid_not_the_mid(self) -> None:
        """An edge over the mid is not an edge on an 8-cent spread.

        With ``min_edge = 0.08`` on a 0.46/0.54 book, a forecast of 0.62 looks
        like a 12-point edge against the mid (0.50) and would trade. Against the
        0.545 price actually paid it is 7.5 points, and must not. The difference
        is the spread, and mistaking one for the other is precisely how a
        backtest invents profit.
        """
        wide = Quote(
            best_bid=0.46, best_ask=0.54, bid_depth_usdc=500.0, ask_depth_usdc=500.0
        )
        # A wider spread must also carry a bigger min_edge; the config layer
        # refuses any pairing where min_edge sits below the round-trip cost.
        loose = Config(execution={"max_spread": 0.10}, prediction={"min_edge": 0.08})
        decision = self._decide(loose, wide, 0.62)
        assert decision.edge == pytest.approx(0.62 - 0.545)
        assert 0.62 - decision.market_prob_up > 0.08  # would have traded on mid
        assert not decision.trade
        assert decision.skip_reason is SkipReason.EDGE_TOO_SMALL

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"seconds_into_window": 10.0}, SkipReason.TOO_EARLY_IN_WINDOW),
            (
                {"seconds_into_window": 295.0, "seconds_to_settlement": 5.0},
                SkipReason.TOO_CLOSE_TO_SETTLEMENT,
            ),
            ({"vol_zscore": 9.0}, SkipReason.ABNORMAL_VOLATILITY),
            ({"calibration_error": 0.5}, SkipReason.MODEL_UNCALIBRATED),
            ({"model_agreement": 0.1}, SkipReason.LOW_CONFIDENCE),
        ],
    )
    def test_each_gate_names_its_own_refusal(
        self, config: Config, quote: Quote, kwargs: dict, expected: SkipReason
    ) -> None:
        decision = self._decide(config, quote, 0.9, **kwargs)
        assert not decision.trade
        assert decision.skip_reason is expected

    def test_stale_book_is_refused_as_stale_not_as_a_bad_price(
        self, config: Config
    ) -> None:
        stale = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=60_000,
        )
        decision = self._decide(config, stale, 0.9)
        assert decision.skip_reason is SkipReason.STALE_DATA

    def test_wide_spread_is_refused(self, config: Config) -> None:
        wide = Quote(
            best_bid=0.40, best_ask=0.60, bid_depth_usdc=500.0, ask_depth_usdc=500.0
        )
        decision = self._decide(config, wide, 0.95)
        assert decision.skip_reason is SkipReason.SPREAD_TOO_WIDE

    def test_thin_book_is_refused(self, config: Config) -> None:
        thin = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=5.0, ask_depth_usdc=5.0
        )
        # 0.80 rather than 0.90: Module 8.5's disagreement ceiling would
        # otherwise reject 0.90 against a 0.50 book before the depth gate.
        decision = self._decide(config, thin, 0.80)
        assert decision.skip_reason is SkipReason.INSUFFICIENT_LIQUIDITY

    def test_low_confidence_is_refused(self, config: Config, quote: Quote) -> None:
        decision = self._decide(config, quote, 0.55)
        assert decision.skip_reason is SkipReason.LOW_CONFIDENCE

    def test_late_window_demands_a_bigger_edge(self, config: Config, quote: Quote) -> None:
        """Between the 30s gate and the clock's 20s no-submit zone the bar is
        ``late_entry_min_edge`` (0.12), not ``min_edge`` (0.04)."""
        late = {"seconds_into_window": 275.0, "seconds_to_settlement": 25.0}
        modest = self._decide(config, quote, 0.60, **late)
        assert modest.skip_reason is SkipReason.EDGE_TOO_SMALL
        assert modest.trade is False
        strong = self._decide(config, quote, 0.80, **late)
        assert strong.trade

    def test_clock_no_submit_window_cannot_be_overridden_by_edge(
        self, config: Config, quote: Quote
    ) -> None:
        decision = self._decide(
            config, quote, 0.99, seconds_into_window=290.0, seconds_to_settlement=10.0
        )
        assert decision.skip_reason is SkipReason.TOO_CLOSE_TO_SETTLEMENT

    def test_missing_diagnostics_do_not_fire_their_gates(
        self, config: Config, quote: Quote
    ) -> None:
        """A diagnostic we could not compute is not evidence of a problem."""
        decision = self._decide(
            config, quote, 0.75, vol_zscore=None, calibration_error=None,
            model_agreement=None,
        )
        assert decision.trade


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
class TestSizing:
    def test_kelly_formula(self) -> None:
        """p=0.6, c=0.5 -> (0.6-0.5)/(1-0.5) = 0.2 of bankroll at full Kelly."""
        assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)

    def test_kelly_refuses_a_negative_edge_outright(self) -> None:
        assert kelly_fraction(0.4, 0.5) == 0.0

    def test_quarter_kelly_is_capped_by_max_risk_per_trade(self, config: Config) -> None:
        """Full Kelly here is 0.2; a quarter is 0.05; the 0.02 cap binds.

        This is the cap that stands between a confident model and the account.
        """
        stake = PositionSizer(config).size(
            bankroll_usdc=1_000.0, model_prob=0.6, entry_price=0.5
        )
        assert stake.accepted
        assert stake.capped_by == "max_risk_per_trade"
        assert stake.usdc == pytest.approx(20.0)

    def test_no_edge_means_no_stake(self, config: Config) -> None:
        stake = PositionSizer(config).size(
            bankroll_usdc=1_000.0, model_prob=0.5, entry_price=0.55
        )
        assert not stake.accepted
        assert stake.skip_reason is SkipReason.NEGATIVE_EV

    def test_dust_is_refused_rather_than_rounded_up(self, config: Config) -> None:
        stake = PositionSizer(config).size(
            bankroll_usdc=20.0, model_prob=0.6, entry_price=0.5
        )
        assert not stake.accepted
        assert stake.skip_reason is SkipReason.SIZE_BELOW_MINIMUM

    def test_position_cap_binds_on_a_large_bankroll(self, config: Config) -> None:
        stake = PositionSizer(config).size(
            bankroll_usdc=1_000_000.0, model_prob=0.6, entry_price=0.5
        )
        assert stake.usdc == pytest.approx(config.sizing.max_position_usdc)
        assert stake.capped_by == "max_position_usdc"

    def test_poor_calibration_shrinks_the_stake(self, config: Config) -> None:
        """Chosen so the per-trade cap does not bind and the haircut is visible:
        p=0.53 at c=0.50 is full-Kelly 0.06, a quarter of which is 0.015 — under
        the 0.02 cap."""
        sizer = PositionSizer(config)
        full = sizer.size(bankroll_usdc=1_000.0, model_prob=0.53, entry_price=0.5)
        haircut = sizer.size(
            bankroll_usdc=1_000.0, model_prob=0.53, entry_price=0.5,
            calibration_error=config.prediction.max_calibration_error * 0.5,
        )
        assert full.capped_by == "kelly"
        assert 0.0 < haircut.usdc < full.usdc

    def test_rounding_never_increases_the_stake(self) -> None:
        chunky = Config(sizing={"stake_rounding_usdc": 10.0})
        stake = PositionSizer(chunky).size(
            bankroll_usdc=1_000.0, model_prob=0.6, entry_price=0.5
        )
        assert stake.usdc <= 20.0 + 1e-9


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
DAY = 86_400_000


class TestRiskLedger:
    def _ledger(self, config: Config, bankroll: float = 1_000.0) -> RiskLedger:
        return RiskLedger.from_config(
            config, starting_bankroll_usdc=bankroll, honour_kill_switch=False
        )

    def test_a_fresh_ledger_allows_a_trade(self, config: Config) -> None:
        assert self._ledger(config).check(now_ms=DAY, stake_usdc=10.0).allowed

    def test_concurrent_position_limit(self, config: Config) -> None:
        ledger = self._ledger(config)
        ledger.register_open(cost_usdc=10.0)
        verdict = ledger.check(now_ms=DAY, stake_usdc=10.0)
        assert not verdict.allowed
        assert verdict.reason is SkipReason.DUPLICATE_POSITION

    def test_consecutive_losses_trigger_a_cooldown(self, config: Config) -> None:
        ledger = self._ledger(config)
        for i in range(config.risk.max_consecutive_losses):
            ledger.register_open(cost_usdc=10.0)
            ledger.register_close(now_ms=DAY + i, cost_usdc=10.0, payoff_usdc=0.0)
        verdict = ledger.check(now_ms=DAY + 100, stake_usdc=10.0)
        assert not verdict.allowed
        assert verdict.reason is SkipReason.RISK_LIMIT

    def test_the_cooldown_expires_rather_than_stopping_forever(
        self, config: Config
    ) -> None:
        """A stand-down that only a win could lift would be permanent, because
        a stopped bot cannot win.

        Stakes are kept small so the streak rule fires alone: five 10-USDC
        losses on a 1,000 bankroll would also trip the 5% daily-loss limit,
        which is a different stand-down with different expiry.
        """
        ledger = self._ledger(config)
        for i in range(config.risk.max_consecutive_losses):
            ledger.register_open(cost_usdc=5.0)
            ledger.register_close(now_ms=DAY + i, cost_usdc=5.0, payoff_usdc=0.0)
        after = DAY + config.risk.cooldown_minutes * 60_000 + 1_000
        assert ledger.check(now_ms=after, stake_usdc=10.0).allowed

    def test_daily_loss_limit_halts_trading(self) -> None:
        # A high streak limit isolates the daily-loss rule from the streak rule.
        config = Config(risk={"max_consecutive_losses": 100})
        ledger = self._ledger(config, bankroll=10_000.0)
        cost = config.risk.max_daily_loss_usdc + 50.0
        ledger.register_open(cost_usdc=cost)
        ledger.register_close(now_ms=DAY, cost_usdc=cost, payoff_usdc=0.0)
        assert ledger.state.halted
        assert not ledger.check(now_ms=DAY + 1, stake_usdc=5.0).allowed

    def test_a_daily_halt_lifts_the_next_day(self) -> None:
        config = Config(risk={"max_consecutive_losses": 100})
        ledger = self._ledger(config, bankroll=10_000.0)
        cost = config.risk.max_daily_loss_usdc + 50.0
        ledger.register_open(cost_usdc=cost)
        ledger.register_close(now_ms=DAY, cost_usdc=cost, payoff_usdc=0.0)
        assert ledger.state.halted
        assert ledger.check(now_ms=DAY * 3, stake_usdc=5.0).allowed

    def test_exposure_limit(self) -> None:
        config = Config(risk={"max_concurrent_positions": 10})
        ledger = self._ledger(config, bankroll=10_000.0)
        ledger.register_open(cost_usdc=config.risk.max_total_exposure_usdc)
        verdict = ledger.check(now_ms=DAY, stake_usdc=50.0)
        assert not verdict.allowed
        assert verdict.reason is SkipReason.RISK_LIMIT

    def test_bookkeeping_is_conserved(self, config: Config) -> None:
        """Bankroll after a settled win equals start - cost + payoff, exactly."""
        ledger = self._ledger(config)
        ledger.register_open(cost_usdc=100.0)
        assert ledger.state.bankroll_usdc == pytest.approx(900.0)
        pnl = ledger.register_close(now_ms=DAY, cost_usdc=100.0, payoff_usdc=180.0)
        assert pnl == pytest.approx(80.0)
        assert ledger.state.bankroll_usdc == pytest.approx(1_080.0)
        assert ledger.state.open_positions == 0
        assert ledger.state.open_exposure_usdc == pytest.approx(0.0)

    def test_the_ledger_never_reads_the_clock(self, config: Config) -> None:
        """Same inputs, same state — the property a backtest depends on."""
        states = []
        for _ in range(2):
            ledger = self._ledger(config)
            ledger.register_open(cost_usdc=50.0)
            ledger.register_close(now_ms=DAY, cost_usdc=50.0, payoff_usdc=0.0)
            states.append(ledger.state.as_dict())
        assert states[0] == states[1]
