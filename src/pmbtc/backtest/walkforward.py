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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pmbtc.backtest.engine import BacktestEngine, ProbabilityModel, Row
from pmbtc.backtest.report import BacktestReport, evaluate_backtest
from pmbtc.config import Config
from pmbtc.exceptions import InsufficientDataError
from pmbtc.logging_setup import get_logger
from pmbtc.models.validation import SplitScheme, TimeSeriesSplitter

log = get_logger("pmbtc.backtest.walkforward")

MS_PER_DAY = 86_400_000

#: Fits a strategy on the rows it is given. Strict walk-forward refits at every
#: fold, so it needs the recipe rather than an already-fitted model.
StrategyFactory = Callable[[Sequence[Row]], ProbabilityModel]


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


# --------------------------------------------------------------------------- #
# Strict walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardFold:
    """One refit-and-test step."""

    index: int
    train_markets: int
    test_markets: int
    purged_markets: int
    train_end_ms: int
    test_start_ms: int
    report: BacktestReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_markets": self.train_markets,
            "test_markets": self.test_markets,
            "purged_markets": self.purged_markets,
            "train_end_ms": self.train_end_ms,
            "test_start_ms": self.test_start_ms,
            "report": self.report.as_dict(),
        }


@dataclass
class StrictWalkForwardReport:
    """Out-of-sample results from a model refitted at every step.

    This is the only honest way to ask "would this have made money". A single
    train/test split answers a much weaker question: it fits once, on a split
    point chosen with hindsight, and tests on one contiguous stretch of market
    conditions.
    """

    folds: list[WalkForwardFold] = field(default_factory=list)
    combined: BacktestReport | None = None
    model_name: str = ""

    @property
    def approved(self) -> bool:
        return bool(self.combined and self.combined.approved)

    @property
    def conclusive(self) -> bool:
        return bool(self.combined and self.combined.conclusive)

    def profitable_folds(self) -> int:
        return sum(1 for f in self.folds if f.report.metrics.net_pnl_usdc > 0)

    def summary(self) -> str:
        if not self.folds:
            return "strict walk-forward produced no fold"
        verdict = "PASSED" if self.approved else "FAILED"
        if not self.conclusive:
            verdict = "INCONCLUSIVE"
        pooled = self.combined.metrics if self.combined else None
        tail = (
            f"pooled roi={pooled.roi:+.2%} n={pooled.trades}"
            if pooled
            else "no pooled result"
        )
        return (
            f"strict walk-forward {verdict} | {len(self.folds)} fold(s), "
            f"{self.profitable_folds()} profitable | {tail}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "conclusive": self.conclusive,
            "model": self.model_name,
            "folds": [fold.as_dict() for fold in self.folds],
            "profitable_folds": self.profitable_folds(),
            "combined": self.combined.as_dict() if self.combined else None,
            "summary": self.summary(),
        }

    def render(self) -> str:
        blocks = [self.summary(), ""]
        for fold in self.folds:
            metrics = fold.report.metrics
            blocks.append(
                f"fold {fold.index}: train={fold.train_markets} markets "
                f"(purged {fold.purged_markets}) test={fold.test_markets} | "
                f"{metrics.trades} trades pnl={metrics.net_pnl_usdc:+.2f} "
                f"brier={metrics.brier:.5f} vs market {metrics.market_brier:.5f}"
            )
        if self.combined:
            blocks += ["", "--- pooled out-of-sample ---", self.combined.render()]
        return "\n".join(blocks)


def run_strict_walk_forward(
    config: Config,
    rows: Iterable[Row],
    fit: StrategyFactory,
    *,
    engine: BacktestEngine | None = None,
    model_name: str = "model",
    n_folds: int | None = None,
    scheme: SplitScheme = SplitScheme.EXPANDING,
    decision_horizon_seconds: int | None = None,
) -> StrictWalkForwardReport:
    """Refit at every fold and trade only forward.

    Splitting is delegated to :class:`~pmbtc.models.validation.TimeSeriesSplitter`
    rather than reimplemented here. That class already groups by market so a
    market is never split across a boundary, and already purges the markets
    adjacent to the test period — adjacent 5-minute windows share microstructure
    state, so a model trained on the market immediately before the one it trades
    has seen a correlated draw. Re-deriving that logic in the backtester is
    exactly the kind of duplication that silently diverges.

    The bankroll does **not** carry across folds. Each fold starts from the
    configured initial bankroll, because a fold measures an edge, and
    compounding across refits would let one lucky early fold inflate the size of
    every trade that followed it.
    """
    import numpy as np

    materialised = list(rows)
    engine = engine or BacktestEngine(config)
    report = StrictWalkForwardReport(model_name=model_name)
    if not materialised:
        return report

    ordered = sorted(
        materialised,
        key=lambda r: (int(r["settlement_time_ms"]), str(r.get("condition_id", ""))),
    )
    timestamps = np.array([int(r["settlement_time_ms"]) for r in ordered], dtype=np.int64)
    groups = np.array([str(r.get("condition_id", "")) for r in ordered])

    requested = n_folds or config.model.cv_folds
    splitter = TimeSeriesSplitter(
        scheme=scheme,
        n_folds=requested,
        embargo_markets=max(1, config.model.embargo_windows // 4),
        min_train_markets=max(10, requested * 4),
    )
    try:
        folds = list(splitter.split(timestamps, groups))
    except InsufficientDataError as exc:
        log.warning("walkforward.strict_insufficient", detail=str(exc))
        return report

    pooled_windows: list[Any] = []
    for fold in folds:
        train_rows = [ordered[i] for i in fold.train]
        test_rows = [ordered[i] for i in fold.test]
        if not train_rows or not test_rows:
            continue
        strategy = fit(train_rows)
        result = engine.run(
            test_rows,
            strategy,
            model_name=f"{model_name}#{fold.index}",
            decision_horizon_seconds=decision_horizon_seconds,
        )
        pooled_windows.extend(result.windows)
        report.folds.append(
            WalkForwardFold(
                index=fold.index,
                train_markets=len({str(r.get("condition_id", "")) for r in train_rows}),
                test_markets=len({str(r.get("condition_id", "")) for r in test_rows}),
                purged_markets=fold.purged,
                train_end_ms=fold.train_end_ms,
                test_start_ms=fold.test_start_ms,
                report=evaluate_backtest(config, result),
            )
        )

    if pooled_windows:
        report.combined = _pool(config, report, pooled_windows, model_name)

    log.info(
        "backtest.strict_walkforward",
        model=model_name,
        folds=len(report.folds),
        profitable=report.profitable_folds(),
        approved=report.approved,
    )
    return report


def _pool(
    config: Config,
    report: StrictWalkForwardReport,
    windows: list[Any],
    model_name: str,
) -> BacktestReport:
    """Gate the pooled out-of-sample record.

    P&L is summed across folds rather than compounded, matching the decision
    above to reset the bankroll at every refit.
    """
    from pmbtc.backtest.engine import BacktestResult

    start = config.backtest.initial_bankroll_usdc
    net = sum(f.report.metrics.net_pnl_usdc for f in report.folds)
    pooled = BacktestResult(
        windows=sorted(windows, key=lambda w: w.settlement_time_ms),
        starting_bankroll_usdc=start,
        ending_bankroll_usdc=start + net,
        model_name=model_name,
        decision_horizon_seconds=(
            report.folds[0].report.decision_horizon_seconds if report.folds else 0
        ),
        fill_style=report.folds[0].report.fill_style if report.folds else "touch",
    )
    return evaluate_backtest(config, pooled)
