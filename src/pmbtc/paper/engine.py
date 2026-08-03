"""The paper trading engine — Module 9.

Runs the *production* decision path against the *live* Polymarket book and
records what would have happened, without ever sending an order. No wallet, no
Polygon transaction, no credential: the only thing separating this from live
trading is that the fill is computed rather than requested.

It answers the question the backtest structurally cannot. A backtest fills
against a book it reconstructed from its own archive, so its fill model is an
assumption checked against itself. Here the order is priced against a book that
is arriving in real time, and then the *subsequent tape* is watched to see
whether trades actually printed at or through our price. That is direct evidence
about whether the fill model is optimistic — and it is the reason paper trading
must precede risking money, not merely a rehearsal of it.

Reuse is total by construction: the decision, sizing, risk and cost layers are
imported, and the decide -> size -> risk -> fill sequence is
:func:`pmbtc.trading.pipeline.evaluate_opportunity` — the identical call the
backtest makes. A divergence between backtest and paper can therefore only come
from the market, never from the code.

Quotes come from an injectable source so the same engine runs against the live
CLOB or against archived frames, which is what keeps replay deterministic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pmbtc.backtest.fills import FillModel
from pmbtc.clock import ClockService
from pmbtc.config import Config
from pmbtc.constants import Outcome, TradeStatus
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.paper.journal import (
    PaperDecision,
    PaperJournal,
    PaperSettlement,
    decision_from_outcome,
)
from pmbtc.trading.costs import CostModel, Quote
from pmbtc.trading.decision import DecisionEngine
from pmbtc.trading.pipeline import evaluate_opportunity
from pmbtc.trading.risk import RiskLedger
from pmbtc.trading.sizing import PositionSizer

log = get_logger("pmbtc.paper.engine")

paper_decisions = METRICS.counter("pmbtc_paper_decisions_total", "Paper windows evaluated.")
paper_trades = METRICS.counter("pmbtc_paper_trades_total", "Paper trades opened.")

#: How far above the decision horizon an evaluation may still fire. The run
#: loop ticks twice a second, so a few seconds is ample; the band must stay
#: above the horizon so the timing gates see the same side of the boundary
#: the backtest does.
EVALUATION_TOLERANCE_S = 3.0


class QuoteSource(Protocol):
    """Where the book comes from: the live CLOB, or an archive replay."""

    def quote_for(self, condition_id: str, now_ms: int) -> Quote | None: ...


@dataclass
class LiveQuoteSource:
    """Reads the book from the Module 5 CLOB feeds the service already runs."""

    feeds: dict[str, Any] = field(default_factory=dict)

    def register(self, condition_id: str, feed: Any) -> None:
        self.feeds[condition_id] = feed

    def unregister(self, condition_id: str) -> None:
        self.feeds.pop(condition_id, None)

    def quote_for(self, condition_id: str, now_ms: int) -> Quote | None:
        feed = self.feeds.get(condition_id)
        if feed is None:
            return None
        book = feed.books.up
        if not book.is_two_sided:
            return None
        # Depth is taken over the configured book levels rather than the touch
        # alone: an order larger than the top level walks into the next one, and
        # a fill model that only sees the touch would silently assume it did not.
        return Quote(
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            bid_depth_usdc=book.notional_depth(10, "bid"),
            ask_depth_usdc=book.notional_depth(10, "ask"),
            age_ms=max(0, book.age_ms(now_ms)),
        )


@dataclass
class PaperStats:
    evaluated: int = 0
    traded: int = 0
    skipped: int = 0
    settled: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "traded": self.traded,
            "skipped": self.skipped,
            "settled": self.settled,
            "errors": self.errors,
        }


class PaperTradingEngine:
    """Evaluates live windows through the production pipeline and records them.

    Holds no network resources of its own — the caller supplies the quote source
    and the clock, so the engine is exercised in tests with neither.
    """

    def __init__(
        self,
        config: Config,
        journal: PaperJournal,
        *,
        quotes: QuoteSource,
        clock: ClockService | None = None,
        model: Callable[[str, Quote], float] | None = None,
        model_name: str = "market",
        ledger: RiskLedger | None = None,
    ) -> None:
        self.config = config
        self.journal = journal
        self.quotes = quotes
        self.clock = clock
        self.costs = CostModel(config.costs)
        self.decisions = DecisionEngine(config, self.costs)
        self.sizer = PositionSizer(config)
        # The paper fill model is the backtest's, with the same pessimism: a
        # different one here would make the two runs incomparable, which is the
        # entire point of doing this before risking money.
        self.fills = FillModel(self.costs, pessimistic=config.backtest.pessimistic_fill)
        self.ledger = ledger or RiskLedger.from_config(
            config,
            starting_bankroll_usdc=config.paper.initial_bankroll_usdc,
            honour_kill_switch=True,
        )
        #: Probability of UP. Defaults to the market's own mid, which produces
        #: exactly zero edge and therefore no trades — a deliberate default, so
        #: an unconfigured paper run records abstentions rather than noise.
        self.model = model or (lambda _cid, quote: quote.mid or 0.5)
        self.model_name = model_name
        self.stats = PaperStats()

    # ------------------------------------------------------------------ #
    def now_ms(self) -> int:
        from pmbtc.utils.timeutils import utc_now_ms

        return self.clock.now_ms() if self.clock else utc_now_ms()

    def decision_horizon(self) -> int:
        """The last snapshot horizon the timing gates still permit.

        Identical rule to the backtest's default, so the two evaluate the same
        moment in the window and their results stay comparable.
        """
        window = self.config.app.window_seconds
        execution = self.config.execution
        allowed = [
            h
            for h in self.config.dataset.snapshot_horizons_s
            if window - h >= execution.min_seconds_into_window
            and h >= execution.min_seconds_to_settlement
        ]
        return min(allowed) if allowed else window

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        *,
        condition_id: str,
        slug: str,
        settlement_time_ms: int,
        calibration_error: float | None = None,
    ) -> PaperDecision | None:
        """Evaluate one live window. Returns the record, or None if too early.

        Called repeatedly by the run loop; it decides for itself whether this
        market is at its decision horizon yet, and refuses to evaluate the same
        market twice.
        """
        now = self.now_ms()
        horizon = self.decision_horizon()
        seconds_left = (settlement_time_ms - now) / 1000.0

        # Evaluate in a narrow band *at or just above* the horizon, never below
        # it.
        #
        # Found by live validation: firing on `seconds_left <= horizon` meant
        # every decision landed a fraction of a second inside the horizon, so
        # `seconds_to_settlement < execution.min_seconds_to_settlement` was
        # always true and the late-window bar (0.12) applied where the backtest,
        # evaluating at exactly T-30, uses the normal bar (0.04). Paper was
        # therefore three times stricter than the backtest it exists to
        # validate — a divergence in the code rather than in the market, which
        # is the one thing this module must never have.
        if not (horizon <= seconds_left <= horizon + EVALUATION_TOLERANCE_S):
            return None
        if self.journal.has_decision(condition_id, horizon):
            return None

        started = time.perf_counter()
        quote = self.quotes.quote_for(condition_id, now)
        if quote is None:
            log.debug("paper.no_quote", slug=slug)
            return None

        outcome = evaluate_opportunity(
            decisions=self.decisions,
            sizer=self.sizer,
            ledger=self.ledger,
            fills=self.fills,
            model_prob_up=self.model(condition_id, quote),
            quote=quote,
            seconds_into_window=self.config.app.window_seconds - seconds_left,
            seconds_to_settlement=seconds_left,
            now_ms=now,
            calibration_error=calibration_error,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        chosen = outcome.decision.outcome
        mid = quote.mid or 0.0
        record = decision_from_outcome(
            outcome,
            condition_id=condition_id,
            slug=slug,
            settlement_time_ms=settlement_time_ms,
            horizon_seconds=horizon,
            decided_at_ms=now,
            quote=quote,
            mid_price=mid if chosen is not Outcome.DOWN else 1.0 - mid,
            touch_price=(self.costs.touch_price(chosen, quote) or 0.0) if chosen else 0.0,
            bankroll_usdc=self.ledger.state.bankroll_usdc,
            model_name=self.model_name,
            clock_status=self.clock.status().value if self.clock else "unknown",
            clock_offset_ms=self.clock.offset_ms if self.clock else 0.0,
            decision_latency_ms=latency_ms,
        )
        self.journal.record_decision(record)
        paper_decisions.inc()
        self.stats.evaluated += 1

        if outcome.fill is not None:
            # The stake leaves the bankroll now and returns at settlement, so
            # concurrent-position and exposure limits bind exactly as they would
            # with real money.
            self.ledger.register_open(cost_usdc=outcome.fill.cost_usdc)
            self.stats.traded += 1
            paper_trades.inc()
            log.info(
                "paper.trade_opened",
                slug=slug,
                outcome=chosen.value if chosen else "?",
                price=round(outcome.fill.price, 4),
                stake=round(outcome.fill.cost_usdc, 2),
                edge=round(outcome.decision.edge, 4),
            )
        else:
            self.stats.skipped += 1
            log.info(
                "paper.skipped",
                slug=slug,
                reason=record.skip_reason,
                stage=record.stage,
                detail=outcome.decision.detail,
            )
        return record

    # ------------------------------------------------------------------ #
    def settle(
        self,
        decision: PaperDecision,
        official_outcome: Outcome | None,
        *,
        settled_at_ms: int | None = None,
        quality: dict[str, Any] | None = None,
    ) -> PaperSettlement:
        """Resolve one recorded decision against the official outcome.

        The label is Polymarket's own resolution and nothing else — the same
        rule Module 4 applies to training labels. A paper P&L computed from a
        reconstructed outcome would be measuring our own reconstruction.
        """
        now = settled_at_ms if settled_at_ms is not None else self.now_ms()
        settlement = PaperSettlement(
            condition_id=decision.condition_id,
            slug=decision.slug,
            settled_at_ms=now,
            settlement_time_ms=decision.settlement_time_ms,
            official_outcome=official_outcome.value if official_outcome else None,
            bankroll_after_usdc=self.ledger.state.bankroll_usdc,
        )
        if quality:
            settlement.fill_probability = quality.get("fill_probability")
            settlement.observed_best_price = quality.get("observed_best_price")
            settlement.price_error = quality.get("price_error")
            settlement.tape_trades_observed = int(quality.get("tape_trades_observed", 0))

        if not decision.traded:
            settlement.status = TradeStatus.REJECTED.value
            self.journal.record_settlement(settlement)
            return settlement

        if official_outcome is None:
            # Unresolvable market: return the stake rather than book a loss.
            self.ledger.register_close(
                now_ms=now,
                cost_usdc=decision.expected_cost_usdc,
                payoff_usdc=decision.expected_cost_usdc,
            )
            settlement.status = TradeStatus.VOIDED.value
            settlement.bankroll_after_usdc = self.ledger.state.bankroll_usdc
            self.journal.record_settlement(settlement)
            return settlement

        won = decision.outcome == official_outcome.value
        payoff = decision.expected_shares if won else 0.0
        net_payoff = payoff - (self.config.costs.gas_cost_usdc if payoff > 0 else 0.0)
        pnl = self.ledger.register_close(
            now_ms=now, cost_usdc=decision.expected_cost_usdc, payoff_usdc=net_payoff
        )
        settlement.status = (
            TradeStatus.SETTLED_WIN.value if won else TradeStatus.SETTLED_LOSS.value
        )
        settlement.won = won
        settlement.payoff_usdc = payoff
        settlement.cost_usdc = decision.expected_cost_usdc
        settlement.pnl_usdc = pnl
        settlement.bankroll_after_usdc = self.ledger.state.bankroll_usdc
        self.journal.record_settlement(settlement)
        self.stats.settled += 1
        log.info(
            "paper.settled",
            slug=decision.slug,
            outcome=official_outcome.value,
            won=won,
            pnl=round(pnl, 4),
            bankroll=round(self.ledger.state.bankroll_usdc, 2),
        )
        return settlement
