"""Model readiness gate.

Module 7 may not train until the dataset earns it. The failure mode this
prevents is specific and common: a model trained on two hundred markets shows a
beautiful validation score, gets believed, and loses money — because at that
sample size a 2% edge and pure noise are indistinguishable.

Every threshold is configurable and every failure names the number it wanted and
the number it got, so "not yet" is actionable rather than mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pmbtc.config import Config
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord
from pmbtc.dataset.stats import DatasetStats, compute_stats
from pmbtc.exceptions import InsufficientDataError
from pmbtc.features.provider import ENGINEERED_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    required: float
    actual: float
    detail: str = ""

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.name}: have {self.actual:g}, need {self.required:g}"
            + (f" — {self.detail}" if self.detail else "")
        )


@dataclass
class ReadinessReport:
    checks: list[ReadinessCheck] = field(default_factory=list)
    stats: DatasetStats | None = None

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[ReadinessCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if self.ready:
            return f"dataset is ready for training ({len(self.checks)} checks passed)"
        lines = [f"dataset is NOT ready for training ({len(self.failures)} check(s) failed):"]
        lines += [f"  {check}" for check in self.failures]
        return "\n".join(lines)

    def raise_if_not_ready(self) -> None:
        """Refuse to train, with a message that says exactly what is missing."""
        if self.ready:
            return
        raise InsufficientDataError(
            "Dataset does not meet the training readiness gate",
            context={
                "failures": "; ".join(
                    f"{c.name}(have={c.actual:g},need={c.required:g})" for c in self.failures
                )
            },
        )


def complete_timeline_count(
    markets: list[MarketRecord],
    snapshots: list[FeatureSnapshot],
    horizons: tuple[int, ...],
) -> int:
    """Markets that are labelled *and* have every scheduled snapshot."""
    wanted = set(horizons)
    by_market: dict[str, set[int]] = {}
    for snapshot in snapshots:
        by_market.setdefault(snapshot.condition_id, set()).add(snapshot.horizon_seconds)
    return sum(
        1
        for market in markets
        if market.is_labelled and wanted.issubset(by_market.get(market.condition_id, set()))
    )


def check_readiness(
    config: Config,
    markets: list[MarketRecord],
    snapshots: list[FeatureSnapshot],
) -> ReadinessReport:
    """Evaluate every gate. Runs all checks so the report is complete."""
    cfg = config.training
    horizons = tuple(config.dataset.snapshot_horizons_s)
    stats = compute_stats(markets, snapshots, horizons)
    report = ReadinessReport(stats=stats)

    def add(name: str, actual: float, required: float, detail: str = "") -> None:
        report.checks.append(
            ReadinessCheck(name, actual >= required, required, actual, detail)
        )

    add(
        "labelled_markets",
        stats.labelled,
        cfg.min_labelled_markets,
        "markets with an official Polymarket outcome",
    )
    add(
        "complete_timelines",
        complete_timeline_count(markets, snapshots, horizons),
        cfg.min_complete_timelines,
        f"labelled markets with all {len(horizons)} snapshots",
    )

    # Coverage is measured against the **engineered** feature set — the columns
    # a model will actually be trained on — not against the whole registry,
    # which still contains legacy raw measurements from earlier collection
    # generations. A feature that was never collected must score zero rather
    # than being invisible.
    engineered = set(ENGINEERED_FEATURE_NAMES)
    if engineered:
        observed = {
            name for name, coverage in stats.feature_coverage.items() if coverage > 0
        }
        present = observed & engineered
        coverage = len(present) / len(engineered)
        add(
            "feature_coverage",
            coverage,
            cfg.min_feature_coverage,
            f"{len(present)} of {len(engineered)} engineered features present",
        )

    up_rate = stats.up_rate
    if up_rate is not None:
        # Distance of the minority class from zero, so both skews are caught.
        minority = min(up_rate, 1.0 - up_rate)
        add(
            "class_balance",
            minority,
            cfg.min_class_balance,
            f"minority class share (up={up_rate:.1%})",
        )
    else:
        report.checks.append(
            ReadinessCheck("class_balance", False, cfg.min_class_balance, 0.0, "no labels yet")
        )

    total_quality = sum(stats.quality_distribution.values())
    if total_quality:
        low = stats.quality_distribution.get("0.0-0.5", 0) + stats.quality_distribution.get(
            "zero", 0
        )
        good_fraction = 1.0 - low / total_quality
        add(
            "quality",
            good_fraction,
            1.0 - cfg.max_low_quality_fraction,
            f"{low} of {total_quality} snapshots below the quality floor",
        )

    add(
        "min_samples",
        len(snapshots),
        cfg.min_samples_to_train,
        "total feature snapshots",
    )
    return report


def readiness_for_store(config: Config, store: object) -> ReadinessReport:
    """Convenience wrapper for a :class:`~pmbtc.dataset.store.DatasetStore`."""
    markets = store.markets()  # type: ignore[attr-defined]
    snapshots = store.snapshots()  # type: ignore[attr-defined]
    return check_readiness(config, markets, snapshots)
