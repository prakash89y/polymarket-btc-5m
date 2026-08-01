"""The deployment gate.

Modelled on :mod:`pmbtc.models.promotion` and for the same reason: every
condition must pass, there is no force flag, and the way to change a threshold
is a reviewable commit to ``config.backtest`` rather than an argument at the
call site.

Two conditions here do not exist in the model-promotion gate, because they are
about money rather than about forecasts:

*Beating the market's own forecast is necessary but not sufficient.* A strategy
must also clear break-even **at the prices it actually paid**. Those are
different bars, and the gap between them is exactly the spread.

*A result on too few trades is not a result.* Below ``backtest.min_trades`` the
run is marked inconclusive and fails, however good the numbers are. This is the
condition that matters most in the current state of the project, where the
dataset is hours old: it makes an encouraging early number impossible to mistake
for evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pmbtc.backtest.engine import BacktestResult
from pmbtc.backtest.metrics import BacktestMetrics, compute_metrics
from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.models.statistics import TestResult, binomial_vs_breakeven

log = get_logger("pmbtc.backtest.report")


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class BacktestReport:
    """Metrics, the gate's verdict, and the statistics behind it."""

    metrics: BacktestMetrics
    checks: list[GateCheck] = field(default_factory=list)
    tests: list[TestResult] = field(default_factory=list)
    model_name: str = ""
    decision_horizon_seconds: int = 0
    fill_style: str = "touch"

    @property
    def approved(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def conclusive(self) -> bool:
        """Whether the sample is large enough for the verdict to mean anything."""
        return next(
            (c.passed for c in self.checks if c.name == "sample_size"),
            False,
        )

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def summary(self) -> str:
        if self.approved:
            return f"backtest gate PASSED ({len(self.checks)} checks)"
        if not self.conclusive:
            return (
                f"backtest INCONCLUSIVE — {self.metrics.trades} trades is too few "
                f"to judge ({len(self.failures)} check(s) failed)"
            )
        return f"backtest gate FAILED ({len(self.failures)} of {len(self.checks)} checks)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "conclusive": self.conclusive,
            "model": self.model_name,
            "decision_horizon_seconds": self.decision_horizon_seconds,
            "fill_style": self.fill_style,
            "summary": self.summary(),
            "metrics": self.metrics.as_dict(),
            "checks": [check.as_dict() for check in self.checks],
            "tests": [test.as_dict() for test in self.tests],
        }

    def render(self) -> str:
        lines = [self.summary(), str(self.metrics), ""]
        lines += [str(check) for check in self.checks]
        if self.tests:
            lines += ["", *[str(test) for test in self.tests]]
        if self.metrics.skips:
            lines += ["", "abstentions:"]
            lines += [
                f"  {reason}: {count}" for reason, count in self.metrics.skips.items()
            ]
        return "\n".join(lines)


def _ok(value: float, floor: float) -> bool:
    return not math.isnan(value) and value >= floor


def evaluate_backtest(
    config: Config,
    result: BacktestResult,
    *,
    metrics: BacktestMetrics | None = None,
    alpha: float | None = None,
) -> BacktestReport:
    """Score a completed run against ``config.backtest``."""
    metrics = metrics or compute_metrics(result)
    cfg = config.backtest
    alpha = alpha if alpha is not None else config.training.promotion_significance

    report = BacktestReport(
        metrics=metrics,
        model_name=result.model_name,
        decision_horizon_seconds=result.decision_horizon_seconds,
        fill_style=result.fill_style,
    )
    add = report.checks.append

    # --- 1. Is there enough here to judge? ------------------------------ #
    add(
        GateCheck(
            "sample_size",
            metrics.trades >= cfg.min_trades,
            f"{metrics.trades} trades, need {cfg.min_trades}",
        )
    )

    # --- 2. Money ------------------------------------------------------- #
    add(
        GateCheck(
            "roi", _ok(metrics.roi, cfg.min_roi), f"{metrics.roi:+.2%}, need {cfg.min_roi:+.2%}"
        )
    )
    add(
        GateCheck(
            "profit_factor",
            _ok(metrics.profit_factor, cfg.min_profit_factor),
            f"{metrics.profit_factor:.3f}, need {cfg.min_profit_factor:.3f}",
        )
    )
    add(
        GateCheck(
            "sharpe",
            _ok(metrics.sharpe, cfg.min_sharpe),
            f"{metrics.sharpe:.3f}, need {cfg.min_sharpe:.3f}",
        )
    )
    add(
        GateCheck(
            "max_drawdown",
            metrics.max_drawdown <= cfg.max_drawdown,
            f"{metrics.max_drawdown:.2%}, limit {cfg.max_drawdown:.2%}",
        )
    )

    # --- 3. Forecast quality -------------------------------------------- #
    add(
        GateCheck(
            "accuracy",
            _ok(metrics.accuracy, cfg.min_accuracy),
            f"{metrics.accuracy:.4f}, need {cfg.min_accuracy:.4f}",
        )
    )
    add(
        GateCheck(
            "brier",
            not math.isnan(metrics.brier) and metrics.brier <= cfg.max_brier,
            f"{metrics.brier:.5f}, limit {cfg.max_brier:.5f}",
        )
    )
    # The benchmark that matters: the book's own forecast, on the same windows.
    beats_market = (
        not math.isnan(metrics.brier)
        and not math.isnan(metrics.market_brier)
        and metrics.brier < metrics.market_brier
    )
    add(
        GateCheck(
            "beats_market_forecast",
            beats_market,
            f"brier {metrics.brier:.5f} vs market {metrics.market_brier:.5f} "
            f"(skill {metrics.brier_skill:+.4f})",
        )
    )

    # --- 4. Did it beat break-even by more than luck? -------------------- #
    if metrics.trades > 0:
        test = binomial_vs_breakeven(
            wins=metrics.wins,
            total=metrics.trades,
            breakeven=min(0.999, max(0.001, metrics.breakeven_win_rate)),
        )
        report.tests.append(test)
        add(
            GateCheck(
                "win_rate_significant",
                test.significant(alpha) and test.effect > 0,
                f"{test} (alpha {alpha})",
            )
        )
    else:
        add(GateCheck("win_rate_significant", False, "no trades to test"))

    log.info(
        "backtest.gate",
        model=result.model_name,
        approved=report.approved,
        conclusive=report.conclusive,
        trades=metrics.trades,
        roi=round(metrics.roi, 6),
        failures=[c.name for c in report.failures],
    )
    return report
