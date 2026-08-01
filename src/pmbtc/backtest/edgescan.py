"""Module 8.5 reports: horizon choice, edge stability, disagreement shape.

Three questions, none of which the deployment gate can answer on its own:

*Which horizon should we trade?* Answered by strict walk-forward at every
candidate horizon, never in-sample. Picking a horizon by looking at the whole
history and taking the best number is the exact overfitting this module exists
to prevent — with six candidates and a handful of trades each, the best one is
whichever got lucky.

*Is whatever edge exists stable?* A strategy whose entire profit arrives in one
fold is not a strategy. The stability report shows the per-fold record and the
sign consistency, and refuses to call anything stable that has not been positive
in most folds.

*What does the model's disagreement with the book look like?* This is the
diagnostic the statistical review needed and had to compute by hand. A healthy
model disagrees with a well-calibrated book by a fraction of a logit most of the
time. The deployed logistic baseline disagreed by a median of 0.83 *probability
points*, which is several logits, and the shape of that distribution is the
fastest way to see it.

Nothing here places, sizes, or simulates an order beyond what
:mod:`pmbtc.backtest.engine` already does. These are read-only views.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pmbtc.backtest.engine import BacktestEngine, ProbabilityModel, Row
from pmbtc.backtest.walkforward import StrategyFactory, run_strict_walk_forward
from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.trading.validation import disagreement_logits

log = get_logger("pmbtc.backtest.edgescan")

#: Candidate decision horizons, longest first. T-300 and T-240 are included
#: even though the stock timing gate forbids them, because "the gate forbids
#: the only horizon with edge" is a finding worth surfacing rather than hiding.
CANDIDATE_HORIZONS = (300, 240, 180, 120, 60, 30)


# --------------------------------------------------------------------------- #
# Disagreement distribution
# --------------------------------------------------------------------------- #
@dataclass
class DisagreementReport:
    """How far this model sits from the book, and how often."""

    horizon_seconds: int = 0
    samples: int = 0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    maximum: float = 0.0
    #: Share of windows beyond the configured ceiling.
    over_ceiling: float = 0.0
    ceiling: float = 0.0
    #: Histogram of |logit| disagreement, in one-logit buckets.
    histogram: dict[str, int] = field(default_factory=dict)
    #: Share of windows where the book itself was already confident.
    against_confident_book: float = 0.0

    @property
    def healthy(self) -> bool:
        """A model that clears the ceiling on almost every window is broken.

        The threshold is deliberately generous: this is a smoke alarm, not a
        promotion gate.
        """
        return self.samples > 0 and self.over_ceiling <= 0.10

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "samples": self.samples,
            "median_logits": round(self.median, 4),
            "p90_logits": round(self.p90, 4),
            "p99_logits": round(self.p99, 4),
            "max_logits": round(self.maximum, 4),
            "ceiling_logits": round(self.ceiling, 4),
            "share_over_ceiling": round(self.over_ceiling, 4),
            "share_against_confident_book": round(self.against_confident_book, 4),
            "healthy": self.healthy,
            "histogram": self.histogram,
        }

    def render(self) -> str:
        if not self.samples:
            return f"T-{self.horizon_seconds}: no priced windows"
        bars = "\n".join(
            f"    {k:>10}  {'#' * min(40, v)} {v}" for k, v in self.histogram.items()
        )
        verdict = "healthy" if self.healthy else "PATHOLOGICAL"
        return (
            f"  T-{self.horizon_seconds}: n={self.samples} median={self.median:.2f} "
            f"p90={self.p90:.2f} p99={self.p99:.2f} max={self.maximum:.2f} logits\n"
            f"    over ceiling {self.ceiling:.2f}: {self.over_ceiling:.1%}  "
            f"against a confident book: {self.against_confident_book:.1%}  [{verdict}]\n"
            f"{bars}"
        )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def disagreement_distribution(
    config: Config,
    rows: Iterable[Row],
    model: ProbabilityModel,
    *,
    horizon_seconds: int,
    confident_book: float = 0.8,
) -> DisagreementReport:
    """Measure model-versus-market disagreement over every priced window.

    Computed directly from the model and the book rather than from executed
    trades, so the picture is not filtered by the very gates being assessed.
    """
    from pmbtc.backtest.engine import quote_from_row

    ceiling = config.prediction.max_disagreement_logits
    report = DisagreementReport(horizon_seconds=horizon_seconds, ceiling=ceiling)

    values: list[float] = []
    confident = 0
    for row in rows:
        if int(row["horizon_seconds"]) != horizon_seconds:
            continue
        quote = quote_from_row(row)
        market = quote.mid
        if market is None:
            continue
        divergence = disagreement_logits(model(row), market)
        values.append(divergence)
        if market >= confident_book or market <= 1.0 - confident_book:
            confident += 1

    if not values:
        return report
    report.samples = len(values)
    report.median = _percentile(values, 0.50)
    report.p90 = _percentile(values, 0.90)
    report.p99 = _percentile(values, 0.99)
    report.maximum = max(values)
    report.over_ceiling = sum(1 for v in values if v > ceiling) / len(values)
    report.against_confident_book = confident / len(values)

    buckets: dict[str, int] = {}
    for value in values:
        lo = int(value)
        key = f"[{lo},{lo + 1})" if lo < 6 else "[6,inf)"
        buckets[key] = buckets.get(key, 0) + 1
    report.histogram = dict(sorted(buckets.items()))
    return report


# --------------------------------------------------------------------------- #
# Horizon scan + stability
# --------------------------------------------------------------------------- #
@dataclass
class HorizonResult:
    """Out-of-sample record for one candidate horizon."""

    horizon_seconds: int
    folds: int = 0
    trades: int = 0
    net_pnl_usdc: float = 0.0
    return_on_stake: float = 0.0
    win_rate: float = 0.0
    breakeven_win_rate: float = 0.0
    model_brier: float = float("nan")
    market_brier: float = float("nan")
    profitable_folds: int = 0
    fold_pnl: list[float] = field(default_factory=list)
    disagreement: DisagreementReport | None = None
    conclusive: bool = False

    @property
    def brier_skill(self) -> float:
        if math.isnan(self.model_brier) or math.isnan(self.market_brier):
            return float("nan")
        if self.market_brier <= 0:
            return float("nan")
        return 1.0 - self.model_brier / self.market_brier

    @property
    def sign_consistency(self) -> float:
        """Share of folds agreeing with the overall sign. 1.0 is perfect."""
        if not self.fold_pnl:
            return 0.0
        positive = sum(1 for p in self.fold_pnl if p > 0)
        return max(positive, len(self.fold_pnl) - positive) / len(self.fold_pnl)

    @property
    def stable(self) -> bool:
        """Profitable, in most folds, and beating the book's own forecast.

        All three are required. Profit from one fold is a coincidence; beating
        the market on Brier without profit is the trap Module 8 was built to
        expose; profit without beating the market is unexplained and should not
        be trusted with size.
        """
        return (
            self.net_pnl_usdc > 0
            and self.folds > 0
            and self.profitable_folds / self.folds >= 0.6
            and not math.isnan(self.brier_skill)
            and self.brier_skill > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "folds": self.folds,
            "trades": self.trades,
            "net_pnl_usdc": round(self.net_pnl_usdc, 4),
            "return_on_stake": round(self.return_on_stake, 6),
            "win_rate": round(self.win_rate, 6),
            "breakeven_win_rate": round(self.breakeven_win_rate, 6),
            "model_brier": round(self.model_brier, 6),
            "market_brier": round(self.market_brier, 6),
            "brier_skill": round(self.brier_skill, 6),
            "profitable_folds": self.profitable_folds,
            "sign_consistency": round(self.sign_consistency, 4),
            "fold_pnl": [round(p, 4) for p in self.fold_pnl],
            "stable": self.stable,
            "conclusive": self.conclusive,
            "disagreement": self.disagreement.as_dict() if self.disagreement else None,
        }


@dataclass
class EdgeScanReport:
    """The Module 8.5 verdict: is there evidence of edge anywhere, at all?"""

    horizons: list[HorizonResult] = field(default_factory=list)
    model_name: str = ""
    folds_requested: int = 0

    @property
    def best(self) -> HorizonResult | None:
        """Highest out-of-sample return on stake among *stable* horizons.

        Deliberately not "highest P&L overall". An unstable horizon with a big
        number is the thing this report exists to refuse.
        """
        candidates = [h for h in self.horizons if h.stable]
        if not candidates:
            return None
        return max(candidates, key=lambda h: h.return_on_stake)

    @property
    def any_evidence_of_edge(self) -> bool:
        return self.best is not None

    def summary(self) -> str:
        if not self.horizons:
            return "edge scan produced no evaluable horizon"
        best = self.best
        if best is None:
            return (
                f"NO EVIDENCE OF EDGE at any of {len(self.horizons)} horizons "
                f"— nothing was profitable, consistent, and better than the book"
            )
        return (
            f"best horizon T-{best.horizon_seconds}: RoS {best.return_on_stake:+.2%} "
            f"over {best.trades} trades, {best.profitable_folds}/{best.folds} folds "
            f"profitable, brier skill {best.brier_skill:+.3f}"
            + ("" if best.conclusive else "  (INCONCLUSIVE: too few trades)")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "folds_requested": self.folds_requested,
            "any_evidence_of_edge": self.any_evidence_of_edge,
            "best_horizon": self.best.horizon_seconds if self.best else None,
            "summary": self.summary(),
            "horizons": [h.as_dict() for h in self.horizons],
        }

    def render(self) -> str:
        lines = [self.summary(), ""]
        lines.append("  edge stability by horizon (strict walk-forward, out-of-sample)")
        lines.append(
            f"  {'horizon':>8}{'folds':>7}{'trades':>8}{'net':>10}{'RoS':>9}"
            f"{'win':>7}{'BE':>7}{'skill':>8}{'signcon':>9}  verdict"
        )
        for h in self.horizons:
            verdict = "STABLE" if h.stable else ("no edge" if h.trades else "no trades")
            lines.append(
                f"  T-{h.horizon_seconds:<6}{h.folds:>7}{h.trades:>8}"
                f"{h.net_pnl_usdc:>10.2f}{h.return_on_stake:>9.1%}{h.win_rate:>7.0%}"
                f"{h.breakeven_win_rate:>7.2f}{h.brier_skill:>8.3f}"
                f"{h.sign_consistency:>9.2f}  {verdict}"
            )
            if h.fold_pnl:
                pnl = " ".join(f"{p:+.1f}" for p in h.fold_pnl)
                lines.append(f"           per-fold P&L: {pnl}")
        lines += ["", "  disagreement distribution (model vs book, in log-odds)"]
        for h in self.horizons:
            if h.disagreement:
                lines.append(h.disagreement.render())
        return "\n".join(lines)


def scan_horizons(
    config: Config,
    rows: Iterable[Row],
    fit: StrategyFactory,
    *,
    engine: BacktestEngine | None = None,
    model_name: str = "model",
    horizons: Sequence[int] = CANDIDATE_HORIZONS,
    n_folds: int = 4,
) -> EdgeScanReport:
    """Evaluate every candidate horizon out-of-sample and report stability.

    ``fit`` is refit inside every fold at every horizon, so no horizon is ever
    chosen using data it was also evaluated on.
    """
    materialised = list(rows)
    engine = engine or BacktestEngine(config)
    report = EdgeScanReport(model_name=model_name, folds_requested=n_folds)

    for horizon in horizons:
        if not any(int(r["horizon_seconds"]) == horizon for r in materialised):
            continue
        walk = run_strict_walk_forward(
            config,
            materialised,
            fit,
            engine=engine,
            model_name=f"{model_name}@T-{horizon}",
            n_folds=n_folds,
            decision_horizon_seconds=horizon,
        )
        result = HorizonResult(horizon_seconds=horizon, folds=len(walk.folds))
        result.fold_pnl = [f.report.metrics.net_pnl_usdc for f in walk.folds]
        result.profitable_folds = walk.profitable_folds()
        if walk.combined is not None:
            m = walk.combined.metrics
            result.trades = m.trades
            result.net_pnl_usdc = m.net_pnl_usdc
            result.return_on_stake = m.return_on_stake
            result.win_rate = m.win_rate
            result.breakeven_win_rate = m.breakeven_win_rate
            result.model_brier = m.brier
            result.market_brier = m.market_brier
            result.conclusive = walk.combined.conclusive

        # The disagreement view is fitted on the earliest fold's training data
        # only, so it describes an out-of-sample model rather than one that has
        # seen every row it is being measured against.
        train = _earliest_training_slice(materialised, n_folds)
        if train:
            result.disagreement = disagreement_distribution(
                config, materialised, fit(train), horizon_seconds=horizon
            )
        report.horizons.append(result)

    log.info(
        "backtest.edge_scan",
        model=model_name,
        horizons=[h.horizon_seconds for h in report.horizons],
        evidence=report.any_evidence_of_edge,
        best=report.best.horizon_seconds if report.best else None,
    )
    return report


def _earliest_training_slice(rows: Sequence[Row], n_folds: int) -> list[Row]:
    """The first chunk of markets, matching the splitter's first training set."""
    order = sorted(
        {(int(r["settlement_time_ms"]), str(r.get("condition_id", ""))) for r in rows}
    )
    ids = [cid for _, cid in order]
    take = max(10, n_folds * 4)
    keep = set(ids[:take])
    return [r for r in rows if str(r.get("condition_id", "")) in keep]
