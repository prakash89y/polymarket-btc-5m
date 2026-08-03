"""Paper-trading performance reports.

Three periods — daily, weekly, cumulative — and one gate.

The gate is the point. ``config.paper`` already states what a paper record must
achieve before live mode may arm: ``min_trades_before_live``,
``min_roi_before_live`` and ``required_significance``. Those thresholds existed
from Module 1 and nothing consumed them until now; this module is what makes
them binding. It reuses :mod:`pmbtc.models.statistics` for the significance test
and :mod:`pmbtc.models.calibration` for forecast scoring, so a paper record is
judged by exactly the arithmetic that judges a model.

Every report scores the model **and the market** on the same windows. On a
market this close to fair that comparison is the whole question: a paper record
that is profitable but worse than the book's own forecast has not found an edge,
it has found a lucky streak, and the distinction is invisible if only one of the
two is reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.models.calibration import EvaluationMetrics, evaluate
from pmbtc.models.statistics import TestResult, binomial_vs_breakeven, paired_bootstrap
from pmbtc.paper.execution import ExecutionSummary, summarise_execution
from pmbtc.paper.journal import PaperRun, skip_histogram

log = get_logger("pmbtc.paper.report")


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, UTC).strftime("%Y-%m-%d")


def _week(ms: int) -> str:
    iso = datetime.fromtimestamp(ms / 1000.0, UTC).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


@dataclass
class PeriodReport:
    """One period's paper record."""

    period: str
    label: str
    decisions: int = 0
    traded: int = 0
    settled: int = 0
    wins: int = 0
    losses: int = 0
    net_pnl_usdc: float = 0.0
    staked_usdc: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.0

    @property
    def return_on_stake(self) -> float:
        return self.net_pnl_usdc / self.staked_usdc if self.staked_usdc > 0 else 0.0

    @property
    def trade_rate(self) -> float:
        return self.traded / self.decisions if self.decisions else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "label": self.label,
            "decisions": self.decisions,
            "traded": self.traded,
            "settled": self.settled,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "net_pnl_usdc": round(self.net_pnl_usdc, 4),
            "staked_usdc": round(self.staked_usdc, 2),
            "return_on_stake": round(self.return_on_stake, 6),
            "trade_rate": round(self.trade_rate, 4),
        }

    def render(self) -> str:
        return (
            f"  {self.label:<12} decisions={self.decisions:<5} traded={self.traded:<4} "
            f"settled={self.settled:<4} win={self.win_rate:>5.1%} "
            f"pnl={self.net_pnl_usdc:>+8.2f} RoS={self.return_on_stake:>+7.2%}"
        )


def _bucket(runs: Sequence[PaperRun], period: str) -> list[PeriodReport]:
    key = _day if period == "daily" else _week
    buckets: dict[str, PeriodReport] = {}
    for run in runs:
        label = key(run.decision.decided_at_ms)
        report = buckets.setdefault(label, PeriodReport(period=period, label=label))
        _accumulate(report, run)
    return [buckets[k] for k in sorted(buckets)]


def _accumulate(report: PeriodReport, run: PaperRun) -> None:
    report.decisions += 1
    if not run.decision.traded:
        return
    report.traded += 1
    report.staked_usdc += run.decision.expected_cost_usdc
    settlement = run.settlement
    if settlement is None:
        return
    report.settled += 1
    report.net_pnl_usdc += settlement.pnl_usdc
    if settlement.won:
        report.wins += 1
    else:
        report.losses += 1


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class PaperReport:
    """The full paper record: periods, forecast quality, execution, gate."""

    cumulative: PeriodReport
    daily: list[PeriodReport] = field(default_factory=list)
    weekly: list[PeriodReport] = field(default_factory=list)
    execution: ExecutionSummary = field(default_factory=ExecutionSummary)
    model: EvaluationMetrics | None = None
    market: EvaluationMetrics | None = None
    significance: TestResult | None = None
    versus_market: TestResult | None = None
    skips: dict[str, int] = field(default_factory=dict)
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def promotion_approved(self) -> bool:
        """May live mode arm? Every check, no override."""
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def brier_skill(self) -> float:
        if not self.model or not self.market or self.market.brier <= 0:
            return float("nan")
        return 1.0 - self.model.brier / self.market.brier

    def as_dict(self) -> dict[str, Any]:
        return {
            "cumulative": self.cumulative.as_dict(),
            "daily": [d.as_dict() for d in self.daily],
            "weekly": [w.as_dict() for w in self.weekly],
            "execution": self.execution.as_dict(),
            "model": self.model.as_dict() if self.model else None,
            "market": self.market.as_dict() if self.market else None,
            "brier_skill": None if math.isnan(self.brier_skill) else round(self.brier_skill, 6),
            "significance": self.significance.as_dict() if self.significance else None,
            "versus_market": self.versus_market.as_dict() if self.versus_market else None,
            "skips": self.skips,
            "promotion_approved": self.promotion_approved,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }

    def render(self) -> str:
        lines = ["paper trading record", "=" * 60, "", "cumulative:"]
        lines.append(self.cumulative.render())
        if self.weekly:
            lines += ["", "weekly:"] + [w.render() for w in self.weekly]
        if self.daily:
            lines += ["", "daily (last 7):"] + [d.render() for d in self.daily[-7:]]
        lines += ["", "execution quality:", self.execution.render()]
        if self.model and self.market:
            lines += [
                "",
                "forecast quality (all evaluated windows):",
                f"  model  {self.model}",
                f"  market {self.market}",
                f"  brier skill vs market: {self.brier_skill:+.4f}",
            ]
        if self.skips:
            lines += ["", "abstentions by reason:"]
            lines += [f"  {k:<28}{v}" for k, v in self.skips.items()]
        lines += ["", "promotion gate:"] + [f"  {c}" for c in self.checks]
        lines += [
            "",
            "VERDICT: "
            + ("APPROVED for live arming" if self.promotion_approved else "NOT approved"),
        ]
        return "\n".join(lines)


def build_report(config: Config, runs: Sequence[PaperRun]) -> PaperReport:
    """Score a paper record and apply ``config.paper``'s promotion gate."""
    cumulative = PeriodReport(period="cumulative", label="all")
    for run in runs:
        _accumulate(cumulative, run)

    report = PaperReport(
        cumulative=cumulative,
        daily=_bucket(runs, "daily"),
        weekly=_bucket(runs, "weekly"),
        execution=summarise_execution(runs),
        skips=skip_histogram(list(runs)),
    )

    # Forecast quality over every settled window, traded or not: abstaining is a
    # forecast too, and scoring only trades would flatter a model that happened
    # to skip its worst calls.
    scored = [r for r in runs if r.label is not None]
    if scored:
        y = np.array([r.label for r in scored], dtype=float)
        model_p = np.array([r.decision.adjusted_prob_up for r in scored], dtype=float)
        market_p = np.array([r.decision.market_prob_up for r in scored], dtype=float)
        report.model = evaluate(y, model_p)
        report.market = evaluate(y, market_p)
        if len(scored) >= 8:
            report.versus_market = paired_bootstrap(
                y, model_p, market_p, metric="brier", seed=config.model.random_seed
            )

    settled_trades = [r for r in runs if r.decision.traded and r.settlement is not None]
    if settled_trades:
        wins = sum(1 for r in settled_trades if r.settlement and r.settlement.won)
        # Break-even is the average price paid, not 0.5: a contract bought at
        # 0.62 must win 62% of the time merely to return the stake.
        breakeven = sum(r.decision.expected_price for r in settled_trades) / len(settled_trades)
        report.significance = binomial_vs_breakeven(wins, len(settled_trades), breakeven)

    report.checks = _gate(config, report)
    log.info(
        "paper.report",
        decisions=cumulative.decisions,
        traded=cumulative.traded,
        settled=cumulative.settled,
        pnl=round(cumulative.net_pnl_usdc, 2),
        approved=report.promotion_approved,
    )
    return report


def _gate(config: Config, report: PaperReport) -> list[GateCheck]:
    """The paper promotion gate. No override parameter, by design."""
    paper = config.paper
    cumulative = report.cumulative
    checks = [
        GateCheck(
            "min_trades",
            cumulative.settled >= paper.min_trades_before_live,
            f"{cumulative.settled} settled paper trades, need {paper.min_trades_before_live}",
        ),
        GateCheck(
            "min_roi",
            cumulative.return_on_stake >= paper.min_roi_before_live,
            f"return on stake {cumulative.return_on_stake:+.2%}, "
            f"need {paper.min_roi_before_live:+.2%}",
        ),
        GateCheck(
            "profitable",
            cumulative.net_pnl_usdc > 0,
            f"net P&L {cumulative.net_pnl_usdc:+.2f} USDC",
        ),
    ]

    significance = report.significance
    checks.append(
        GateCheck(
            "significance",
            significance is not None and significance.p_value < paper.required_significance,
            (
                f"p={significance.p_value:.4f} vs alpha {paper.required_significance} "
                f"({significance.detail})"
                if significance
                else "no settled trades to test"
            ),
        )
    )

    # Beating the book's own forecast is the thesis. Profit without it is
    # unexplained, and unexplained profit should not be given real money.
    skill = report.brier_skill
    checks.append(
        GateCheck(
            "beats_market_forecast",
            not math.isnan(skill) and skill > 0,
            f"brier skill vs market {skill:+.4f}" if not math.isnan(skill) else "not scored",
        )
    )

    # A simulation that flattered itself invalidates the backtest the live
    # thresholds were tuned on, whatever the paper P&L says.
    checks.append(
        GateCheck(
            "execution_not_optimistic",
            not report.execution.simulation_is_optimistic,
            f"mean price error {report.execution.mean_price_error:+.5f} "
            f"over {report.execution.measured} measured trade(s)",
        )
    )
    return checks
