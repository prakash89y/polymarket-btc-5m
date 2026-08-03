"""The one path from an opportunity to an order.

Backtest, paper and live must answer "do we trade this window, at what size, at
what price" *identically*, or the comparison between them means nothing. Paper
trading exists to test whether the backtest's assumptions survive contact with a
real book; if the two ran even slightly different sequences, a divergence in the
results could never be attributed to the market rather than to the code.

So the sequence lives here, once:

    decide  -> DecisionEngine   trade, or a named SkipReason
    size    -> PositionSizer    fractional Kelly, capped
    risk    -> RiskLedger       portfolio limits, drawdown, streaks
    fill    -> FillModel        what the book would actually have given us

The ordering is load-bearing and easy to get subtly wrong, which is the reason
it is not left to each caller to reproduce:

*Risk is checked before the fill, not after.* A trade blocked by a daily-loss
limit never reached the book, so it must not be counted as a fill attempt — the
fill-rate statistic would otherwise be diluted by orders that were never sent.

*The first refusal wins and keeps its own reason.* A window rejected for
``edge_too_small`` must not be relabelled ``insufficient_liquidity`` by a later
stage that never should have run.

What this module deliberately does **not** do is settle. The backtest settles
instantly from a stored label; paper waits for Polymarket to resolve the market
hours later. That difference is real, and forcing both through one function
would mean inventing a fake settlement for the live case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pmbtc.constants import Outcome, SkipReason
from pmbtc.trading.costs import Fill, Quote
from pmbtc.trading.decision import Decision, DecisionEngine
from pmbtc.trading.risk import RiskLedger
from pmbtc.trading.sizing import PositionSizer, Stake


class Stage(StrEnum):
    """How far an opportunity got before it was refused, or that it filled.

    Reported so a run can be diagnosed by *where* opportunities die, which is a
    different and more useful question than how many did.
    """

    DECISION = "decision"
    SIZING = "sizing"
    RISK = "risk"
    FILL = "fill"
    FILLED = "filled"


class SupportsExecute(Protocol):
    """The fill model's contract, kept structural so the paper engine can
    supply a live-book implementation without importing the backtest."""

    def execute(self, *, outcome: Outcome, quote: Quote, stake_usdc: float) -> Any: ...


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """Everything a caller needs to record the attempt, settled or not."""

    decision: Decision
    stage: Stage
    stake: Stake | None = None
    fill: Fill | None = None
    skip_reason: SkipReason | None = None
    #: True only when an order actually reached the fill model. Orders stopped
    #: by the decision gate, sizing or risk never became attempts.
    fill_attempted: bool = False

    @property
    def traded(self) -> bool:
        return self.fill is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "traded": self.traded,
            "fill_attempted": self.fill_attempted,
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
            "decision": self.decision.as_dict(),
            "stake": self.stake.as_dict() if self.stake else None,
            "fill": (
                {
                    "outcome": self.fill.outcome.value,
                    "price": round(self.fill.price, 6),
                    "shares": round(self.fill.shares, 6),
                    "cost_usdc": round(self.fill.cost_usdc, 6),
                    "fee_usdc": round(self.fill.fee_usdc, 6),
                    "slippage_usdc": round(self.fill.slippage_usdc, 6),
                    "partial": self.fill.partial,
                }
                if self.fill
                else None
            ),
        }


def evaluate_opportunity(
    *,
    decisions: DecisionEngine,
    sizer: PositionSizer,
    ledger: RiskLedger,
    fills: SupportsExecute,
    model_prob_up: float,
    quote: Quote,
    seconds_into_window: float,
    seconds_to_settlement: float,
    now_ms: int,
    vol_zscore: float | None = None,
    calibration_error: float | None = None,
    model_agreement: float | None = None,
) -> PipelineOutcome:
    """Run one opportunity through the full gauntlet.

    Pure with respect to everything except ``ledger``, which is asked whether
    the trade is permitted but is **not** debited here — opening a position is
    the caller's decision, because the backtest opens and settles in the same
    instant while paper opens now and settles later.
    """
    decision = decisions.decide(
        model_prob_up=model_prob_up,
        quote=quote,
        seconds_into_window=seconds_into_window,
        seconds_to_settlement=seconds_to_settlement,
        vol_zscore=vol_zscore,
        calibration_error=calibration_error,
        model_agreement=model_agreement,
    )
    if not decision.trade or decision.outcome is None:
        return PipelineOutcome(
            decision=decision, stage=Stage.DECISION, skip_reason=decision.skip_reason
        )

    stake = sizer.size(
        bankroll_usdc=ledger.state.bankroll_usdc,
        model_prob=decision.confidence,
        entry_price=decision.entry_price,
        calibration_error=calibration_error,
    )
    if not stake.accepted:
        return PipelineOutcome(
            decision=decision,
            stage=Stage.SIZING,
            stake=stake,
            skip_reason=stake.skip_reason,
        )

    verdict = ledger.check(now_ms=now_ms, stake_usdc=stake.usdc)
    if not verdict.allowed:
        return PipelineOutcome(
            decision=decision,
            stage=Stage.RISK,
            stake=stake,
            skip_reason=verdict.reason,
        )

    # Only from here is an order genuinely attempted against a book.
    filled = fills.execute(outcome=decision.outcome, quote=quote, stake_usdc=stake.usdc)
    if filled.fill is None:
        return PipelineOutcome(
            decision=decision,
            stage=Stage.FILL,
            stake=stake,
            skip_reason=filled.skip_reason,
            fill_attempted=True,
        )

    return PipelineOutcome(
        decision=decision,
        stage=Stage.FILLED,
        stake=stake,
        fill=filled.fill,
        fill_attempted=True,
    )
