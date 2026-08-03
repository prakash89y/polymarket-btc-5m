"""Execution quality: did the simulation tell the truth?

A backtest fills against a book reconstructed from its own archive, so its fill
model is an assumption validated against itself. Paper trading can do better,
because after a simulated order is priced at time T the *real tape keeps
printing*. Trades that occur at or through our price after T are direct evidence
that the order would have filled; their absence is evidence it would not.

That single measurement is the reason paper trading precedes risking money
rather than merely rehearsing it. Everything else here — slippage, spread paid,
latency — the backtest already estimates. Fill probability it cannot.

Sign convention, stated once because it is easy to invert:

``price_error = expected_price - observed_best_price``

Positive means the simulation assumed a **worse** price than the market offered,
so the backtest was conservative. Negative means it assumed a better price than
was available — the dangerous direction, and the one that turns a profitable
backtest into a losing strategy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pmbtc.constants import Outcome
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.paper.execution")


@dataclass(frozen=True, slots=True)
class TapePrint:
    """One execution seen on the tape after a decision."""

    price: float
    size: float
    timestamp_ms: int


def would_have_filled(outcome: Outcome, expected_price: float, price: float) -> bool:
    """Would a resting order at ``expected_price`` have been hit by this print?

    Buying UP means lifting the ask, so any print at or **below** our price is a
    fill. Buying DOWN is the complement — the tape quotes UP, so a DOWN buy at
    ``p`` fills when UP prints at or **above** ``1 - p``.
    """
    if outcome is Outcome.UP:
        return price <= expected_price
    return price >= 1.0 - expected_price


@dataclass
class ExecutionQuality:
    """What the tape said about one simulated order."""

    tape_trades_observed: int = 0
    filling_trades: int = 0
    fill_probability: float | None = None
    observed_best_price: float | None = None
    price_error: float | None = None
    #: Notional that printed at or through our price, in UP-token terms.
    filling_notional: float = 0.0

    @property
    def optimistic(self) -> bool:
        """Did the simulation assume a better price than the market offered?"""
        return self.price_error is not None and self.price_error < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tape_trades_observed": self.tape_trades_observed,
            "filling_trades": self.filling_trades,
            "fill_probability": self.fill_probability,
            "observed_best_price": self.observed_best_price,
            "price_error": self.price_error,
            "filling_notional": round(self.filling_notional, 4),
            "optimistic": self.optimistic,
        }


def measure_execution(
    *,
    outcome: Outcome | None,
    expected_price: float,
    decided_at_ms: int,
    prints: Iterable[TapePrint],
) -> ExecutionQuality:
    """Score a simulated order against the tape that followed it.

    Only prints strictly after the decision instant count. Including the print
    that triggered the decision would let the measurement confirm itself.
    """
    quality = ExecutionQuality()
    if outcome is None or expected_price <= 0.0:
        return quality

    after = [p for p in prints if p.timestamp_ms > decided_at_ms]
    quality.tape_trades_observed = len(after)
    if not after:
        # No prints is not evidence of a bad fill; it is no evidence at all, and
        # reporting 0.0 would quietly become "never fills" in the average.
        return quality

    filling = [p for p in after if would_have_filled(outcome, expected_price, p.price)]
    quality.filling_trades = len(filling)
    quality.fill_probability = len(filling) / len(after)
    quality.filling_notional = sum(p.price * p.size for p in filling)

    # Best price available to us after the decision, in our own direction.
    if outcome is Outcome.UP:
        best = min(p.price for p in after)
    else:
        best = 1.0 - max(p.price for p in after)
    quality.observed_best_price = best
    quality.price_error = expected_price - best
    return quality


# --------------------------------------------------------------------------- #
@dataclass
class ExecutionSummary:
    """Aggregate execution quality across many paper trades."""

    trades: int = 0
    measured: int = 0
    mean_fill_probability: float = 0.0
    mean_price_error: float = 0.0
    optimistic_trades: int = 0
    mean_expected_price: float = 0.0
    mean_spread_paid: float = 0.0
    mean_slippage_usdc: float = 0.0
    mean_decision_latency_ms: float = 0.0
    mean_book_age_ms: float = 0.0
    partial_fills: int = 0
    errors: list[float] = field(default_factory=list)

    @property
    def simulation_is_optimistic(self) -> bool:
        """Is the backtest's fill model, on average, too kind to itself?"""
        return self.measured > 0 and self.mean_price_error < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "measured": self.measured,
            "mean_fill_probability": round(self.mean_fill_probability, 4),
            "mean_price_error": round(self.mean_price_error, 6),
            "optimistic_trades": self.optimistic_trades,
            "mean_expected_price": round(self.mean_expected_price, 4),
            "mean_spread_paid": round(self.mean_spread_paid, 6),
            "mean_slippage_usdc": round(self.mean_slippage_usdc, 6),
            "mean_decision_latency_ms": round(self.mean_decision_latency_ms, 2),
            "mean_book_age_ms": round(self.mean_book_age_ms, 1),
            "partial_fills": self.partial_fills,
            "simulation_is_optimistic": self.simulation_is_optimistic,
        }

    def render(self) -> str:
        if not self.trades:
            return "  no paper trades to score"
        verdict = (
            "OPTIMISTIC — the backtest assumed better prices than the market offered"
            if self.simulation_is_optimistic
            else "conservative — the backtest assumed prices no better than reality"
        )
        return (
            f"  trades={self.trades} measured={self.measured}\n"
            f"  fill probability   {self.mean_fill_probability:.1%}\n"
            f"  price error        {self.mean_price_error:+.5f}  ({verdict})\n"
            f"  expected price     {self.mean_expected_price:.4f}\n"
            f"  half-spread paid   {self.mean_spread_paid:.5f}\n"
            f"  decision latency   {self.mean_decision_latency_ms:.1f} ms\n"
            f"  book age at entry  {self.mean_book_age_ms:.0f} ms\n"
            f"  partial fills      {self.partial_fills}"
        )


def summarise_execution(runs: Sequence[Any]) -> ExecutionSummary:
    """Aggregate execution quality over :class:`PaperRun` objects."""
    summary = ExecutionSummary()
    traded = [r for r in runs if r.decision.traded]
    summary.trades = len(traded)
    if not traded:
        return summary

    probabilities: list[float] = []
    for run in traded:
        decision = run.decision
        summary.mean_expected_price += decision.expected_price
        summary.mean_spread_paid += max(0.0, decision.touch_price - decision.mid_price)
        summary.mean_slippage_usdc += decision.expected_slippage_usdc
        summary.mean_decision_latency_ms += decision.decision_latency_ms
        summary.mean_book_age_ms += decision.book_age_ms
        summary.partial_fills += 1 if decision.partial else 0

        settlement = run.settlement
        if settlement is None:
            continue
        if settlement.fill_probability is not None:
            probabilities.append(settlement.fill_probability)
        if settlement.price_error is not None:
            summary.errors.append(settlement.price_error)
            if settlement.price_error < 0:
                summary.optimistic_trades += 1

    n = len(traded)
    summary.mean_expected_price /= n
    summary.mean_spread_paid /= n
    summary.mean_slippage_usdc /= n
    summary.mean_decision_latency_ms /= n
    summary.mean_book_age_ms /= n
    summary.measured = len(summary.errors)
    if probabilities:
        summary.mean_fill_probability = sum(probabilities) / len(probabilities)
    if summary.errors:
        summary.mean_price_error = sum(summary.errors) / len(summary.errors)
    return summary
