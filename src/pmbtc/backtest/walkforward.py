"""Walk-forward evaluation over several lookbacks.

A single backtest over all available history answers "would this have worked on
average". That is the wrong question. A strategy whose entire profit came from
one favourable fortnight is not a strategy, and an average over the whole span
hides exactly that. So the gate in ``config.backtest.windows_days`` is applied
to each lookback separately and every one of them must pass.

The reference instant is the last settlement in the data, never the wall clock.
Re-running last month's backtest tomorrow must produce last month's answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pmbtc.backtest.engine import BacktestEngine, ProbabilityModel, Row
from pmbtc.backtest.report import BacktestReport, evaluate_backtest
from pmbtc.config import Config
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.backtest.walkforward")

MS_PER_DAY = 86_400_000


@dataclass
class WalkForwardReport:
    """One report per lookback, plus the combined verdict."""

    windows: dict[int, BacktestReport] = field(default_factory=dict)
    #: Lookbacks that had no rows at all — not failures, just absent history.
    skipped_days: list[int] = field(default_factory=list)
    model_name: str = ""

    @property
    def approved(self) -> bool:
        return bool(self.windows) and all(r.approved for r in self.windows.values())

    @property
    def conclusive(self) -> bool:
        return bool(self.windows) and all(r.conclusive for r in self.windows.values())

    def roi_spread(self) -> float:
        """Best minus worst ROI across lookbacks — a crude stability read."""
        rois = [r.metrics.roi for r in self.windows.values()]
        return max(rois) - min(rois) if len(rois) > 1 else 0.0

    def summary(self) -> str:
        if not self.windows:
            return "walk-forward produced no evaluable window"
        parts = [
            f"{days}d: {'PASS' if report.approved else 'FAIL'} "
            f"roi={report.metrics.roi:+.2%} n={report.metrics.trades}"
            for days, report in sorted(self.windows.items())
        ]
        verdict = "PASSED" if self.approved else "FAILED"
        if not self.conclusive:
            verdict = "INCONCLUSIVE"
        return f"walk-forward {verdict} | " + " | ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "conclusive": self.conclusive,
            "model": self.model_name,
            "roi_spread": round(self.roi_spread(), 6),
            "skipped_days": self.skipped_days,
            "summary": self.summary(),
            "windows": {str(d): r.as_dict() for d, r in sorted(self.windows.items())},
        }

    def render(self) -> str:
        blocks = [self.summary()]
        for days, report in sorted(self.windows.items()):
            blocks += ["", f"--- last {days} days ---", report.render()]
        return "\n".join(blocks)


def slice_recent(rows: Iterable[Row], days: int, *, reference_ms: int | None = None) -> list[Row]:
    """Rows settling within ``days`` of the last settlement in the data."""
    materialised = list(rows)
    if not materialised:
        return []
    reference = (
        reference_ms
        if reference_ms is not None
        else max(int(r["settlement_time_ms"]) for r in materialised)
    )
    cutoff = reference - days * MS_PER_DAY
    return [r for r in materialised if int(r["settlement_time_ms"]) >= cutoff]


def run_walk_forward(
    config: Config,
    rows: Iterable[Row],
    model: ProbabilityModel,
    *,
    engine: BacktestEngine | None = None,
    model_name: str = "model",
    decision_horizon_seconds: int | None = None,
) -> WalkForwardReport:
    """Run and gate the backtest over every configured lookback."""
    materialised = list(rows)
    engine = engine or BacktestEngine(config)
    report = WalkForwardReport(model_name=model_name)

    for days in sorted(config.backtest.windows_days):
        subset = slice_recent(materialised, days)
        if not subset:
            report.skipped_days.append(days)
            continue
        result = engine.run(
            subset,
            model,
            model_name=model_name,
            decision_horizon_seconds=decision_horizon_seconds,
        )
        report.windows[days] = evaluate_backtest(config, result)

    log.info(
        "backtest.walkforward",
        model=model_name,
        approved=report.approved,
        conclusive=report.conclusive,
        windows=sorted(report.windows),
        skipped=report.skipped_days,
    )
    return report
