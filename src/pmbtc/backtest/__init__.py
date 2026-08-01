"""Module 8 — backtesting: does the edge survive contact with the book?

Every gate before this one scores *forecasts*. Brier score and log loss say how
well the model knows the world; they say nothing about money. On a market where
the spread is 1-3 cents and the whole edge is a few percentage points, the
difference is decisive: a model can beat the market's own forecast on Brier and
still lose on every single trade after paying to cross the spread. This module
is the first thing in the system that measures P&L, and therefore the first
thing that can answer whether to trade at all.

    labelled rows (Module 4/6)
      -> BacktestEngine     one decision per window, chronological
           DecisionEngine   trade or a named SkipReason      (pmbtc.trading)
           PositionSizer    fractional Kelly, capped         (pmbtc.trading)
           FillModel        what the book would actually have given us
           RiskLedger       daily/weekly/streak stand-downs  (pmbtc.trading)
      -> BacktestMetrics    ROI, profit factor, Sharpe, drawdown, Brier
      -> BacktestReport     the deployment gate + walk-forward stability

Three properties are deliberate:

*It is pessimistic by construction.* Fills cross the spread, pay slippage, and
are capped by the depth that was actually resting. Missing depth data is treated
as no depth, not as infinite depth.

*It is deterministic.* No clock reads, no RNG, no set iteration order. The same
dataset and config produce byte-identical results, which is what makes a
regression in the strategy detectable rather than arguable.

*It cannot conclude from noise.* A result computed on fewer than
``backtest.min_trades`` trades is reported as inconclusive and fails the gate,
however good the numbers look.
"""

from __future__ import annotations

from pmbtc.backtest.adapters import (
    ConstantModel,
    EstimatorModel,
    MarketProbabilityModel,
)
from pmbtc.backtest.attribution import EdgeDecomposition, EdgeLine, decompose
from pmbtc.backtest.edgescan import (
    CANDIDATE_HORIZONS,
    DisagreementReport,
    EdgeScanReport,
    HorizonResult,
    disagreement_distribution,
    scan_horizons,
)
from pmbtc.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    ColumnMap,
    ProbabilityModel,
    WindowResult,
    quote_from_row,
)
from pmbtc.backtest.fills import FillModel, FillOutcome
from pmbtc.backtest.metrics import BacktestMetrics, compute_metrics
from pmbtc.backtest.report import BacktestReport, GateCheck, evaluate_backtest
from pmbtc.backtest.walkforward import (
    StrategyFactory,
    StrictWalkForwardReport,
    WalkForwardFold,
    WalkForwardReport,
    run_strict_walk_forward,
    run_walk_forward,
    slice_recent,
)

__all__ = [
    "CANDIDATE_HORIZONS",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestReport",
    "BacktestResult",
    "ColumnMap",
    "ConstantModel",
    "DisagreementReport",
    "EdgeDecomposition",
    "EdgeLine",
    "EdgeScanReport",
    "EstimatorModel",
    "FillModel",
    "FillOutcome",
    "GateCheck",
    "HorizonResult",
    "MarketProbabilityModel",
    "ProbabilityModel",
    "StrategyFactory",
    "StrictWalkForwardReport",
    "WalkForwardFold",
    "WalkForwardReport",
    "WindowResult",
    "compute_metrics",
    "decompose",
    "disagreement_distribution",
    "evaluate_backtest",
    "quote_from_row",
    "run_strict_walk_forward",
    "run_walk_forward",
    "scan_horizons",
    "slice_recent",
]
