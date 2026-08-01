"""The trading layer: the path from a probability to a position.

Modules 1-7 answer "what will happen". This package answers "does that belong
in a position, at what price, and for how much" — and it is deliberately
separate from :mod:`pmbtc.backtest`, because the same four questions have to be
answered identically in simulation, in paper, and with real money. A cost model
that lives inside the simulator is a cost model the live path will eventually
re-implement, slightly differently, and the difference will only show up in the
P&L.

    Quote + model probability
      -> DecisionEngine   gates -> trade or a named SkipReason
      -> CostModel        entry price, fees, round-trip cost
      -> PositionSizer    fractional Kelly, capped
      -> RiskLedger       portfolio limits, drawdown, kill switch

Nothing here touches the network, the clock, or a random number generator.
Every function is pure in its inputs, so the same market state produces the same
decision in a backtest as it does at 3 a.m. against the live book.
"""

from __future__ import annotations

from pmbtc.trading.costs import CostModel, Fill, Quote
from pmbtc.trading.decision import Decision, DecisionEngine
from pmbtc.trading.risk import RiskLedger, RiskState
from pmbtc.trading.sizing import PositionSizer, Stake
from pmbtc.trading.validation import (
    DisagreementMonitor,
    EdgeValidation,
    MarketEdgeValidator,
    blend_toward_market,
    disagreement_logits,
    inv_logit,
    logit,
)

__all__ = [
    "CostModel",
    "Decision",
    "DecisionEngine",
    "DisagreementMonitor",
    "EdgeValidation",
    "Fill",
    "MarketEdgeValidator",
    "PositionSizer",
    "Quote",
    "RiskLedger",
    "RiskState",
    "Stake",
    "blend_toward_market",
    "disagreement_logits",
    "inv_logit",
    "logit",
]
