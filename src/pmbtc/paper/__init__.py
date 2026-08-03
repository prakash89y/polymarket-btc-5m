"""Module 9 — paper trading against the live Polymarket book.

Runs the production decision path against real-time market data and records
what would have happened, without ever sending an order. No wallet, no Polygon
transaction, no credential, no new integration: the only difference from live
trading is that the fill is computed rather than requested.

    live CLOB book (Module 5)
      -> LiveQuoteSource
      -> evaluate_opportunity   the *same* call the backtest makes
      -> PaperJournal           every decision, traded or skipped
      -> settlement             Polymarket's official outcome, nothing else
      -> measure_execution      did the tape confirm our fill?
      -> build_report           daily / weekly / cumulative + promotion gate

Two properties make this worth doing rather than simply trusting Module 8:

*It shares one implementation with the backtest.* The decide -> size -> risk ->
fill sequence lives in :mod:`pmbtc.trading.pipeline` and both engines call it, so
a divergence between backtest and paper can only come from the market.

*It measures what a backtest cannot.* A backtest fills against a book it
reconstructed from its own archive — its fill model is an assumption checked
against itself. Here the tape keeps printing after the order is priced, and
those prints are direct evidence about whether the fill would have happened.
:attr:`~pmbtc.paper.execution.ExecutionSummary.simulation_is_optimistic` is the
answer, and the promotion gate refuses to arm live trading when it is true.
"""

from __future__ import annotations

from pmbtc.paper.engine import (
    LiveQuoteSource,
    PaperStats,
    PaperTradingEngine,
    QuoteSource,
)
from pmbtc.paper.execution import (
    ExecutionQuality,
    ExecutionSummary,
    TapePrint,
    measure_execution,
    summarise_execution,
    would_have_filled,
)
from pmbtc.paper.journal import (
    PaperDecision,
    PaperJournal,
    PaperRun,
    PaperSettlement,
    open_journal,
    skip_histogram,
)
from pmbtc.paper.report import PaperReport, PeriodReport, build_report

__all__ = [
    "ExecutionQuality",
    "ExecutionSummary",
    "LiveQuoteSource",
    "PaperDecision",
    "PaperJournal",
    "PaperReport",
    "PaperRun",
    "PaperSettlement",
    "PaperStats",
    "PaperTradingEngine",
    "PeriodReport",
    "QuoteSource",
    "TapePrint",
    "build_report",
    "measure_execution",
    "open_journal",
    "skip_histogram",
    "summarise_execution",
    "would_have_filled",
]
