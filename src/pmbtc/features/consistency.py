"""Cross-source consistency.

Three sources measure overlapping things: the Polymarket CLOB, Binance, and the
settlement provider. When they disagree, the right response is not to pick a
winner — it is to *annotate the affected features* so that downstream code knows
the input was contested, and to reduce confidence rather than trade through it.

Four checks:

``stale_feed``     a source has not updated within its budget
``abnormal_spread`` a venue's spread is far outside its normal range
``timestamp_drift`` a source's clock disagrees with ours
``price_divergence`` two price sources disagree beyond tolerance

The affected-feature mapping is derived from the dependency graph rather than
hand-listed, so a new feature reading a degraded source is annotated
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pmbtc.features.graph import FeatureGraph
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS

log = get_logger("pmbtc.features.consistency")

consistency_issues = METRICS.counter(
    "pmbtc_consistency_issues_total", "Cross-source consistency issues, by kind."
)


class InconsistencyKind(StrEnum):
    STALE_FEED = "stale_feed"
    ABNORMAL_SPREAD = "abnormal_spread"
    TIMESTAMP_DRIFT = "timestamp_drift"
    PRICE_DIVERGENCE = "price_divergence"


@dataclass(frozen=True, slots=True)
class Inconsistency:
    kind: InconsistencyKind
    source: str
    observed: float
    limit: float
    detail: str

    @property
    def severity(self) -> float:
        """How far past the limit, as a multiple. Drives confidence reduction."""
        return abs(self.observed) / self.limit if self.limit else 1.0

    def __str__(self) -> str:
        return f"{self.kind.value}[{self.source}]: {self.detail}"


@dataclass
class ConsistencyReport:
    issues: list[Inconsistency] = field(default_factory=list)
    #: Raw sources considered unreliable right now.
    degraded_sources: set[str] = field(default_factory=set)
    #: Features touching a degraded source, derived from the graph.
    affected_features: frozenset[str] = frozenset()

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def confidence_multiplier(self) -> float:
        """Factor to scale model confidence by, in (0, 1].

        Deliberately blunt and monotone: each issue reduces confidence, a severe
        one reduces it more, and the floor is 0.1 rather than 0 so the system
        degrades rather than silently stopping.
        """
        multiplier = 1.0
        for issue in self.issues:
            multiplier *= max(0.3, 1.0 - 0.2 * min(3.0, issue.severity))
        return max(0.1, multiplier)

    def annotate(self, values: dict[str, float | None]) -> dict[str, Any]:
        """Attach the contested flag to each affected feature."""
        return {
            name: {
                "value": value,
                "contested": name in self.affected_features,
            }
            for name, value in values.items()
        }

    def summary(self) -> str:
        if self.clean:
            return "sources consistent"
        return (
            f"{len(self.issues)} inconsistency(ies), "
            f"{len(self.affected_features)} feature(s) annotated, "
            f"confidence x{self.confidence_multiplier:.2f}: "
            + "; ".join(str(i) for i in self.issues)
        )


@dataclass(frozen=True, slots=True)
class ConsistencyLimits:
    max_feed_age_ms: int = 5_000
    max_pm_spread: float = 0.10
    max_ref_spread_bps: float = 10.0
    max_timestamp_drift_ms: int = 2_000
    #: Two reference price sources may differ by this much before it is an issue.
    max_price_divergence_bps: float = 25.0


def check_consistency(
    graph: FeatureGraph,
    *,
    now_ms: int,
    pm_book_age_ms: int | None = None,
    ref_quote_age_ms: int | None = None,
    pm_spread: float | None = None,
    ref_spread_bps: float | None = None,
    clock_offset_ms: float = 0.0,
    settlement_price: float | None = None,
    reference_price: float | None = None,
    limits: ConsistencyLimits | None = None,
) -> ConsistencyReport:
    """Compare the sources and annotate what a disagreement affects."""
    limits = limits or ConsistencyLimits()
    issues: list[Inconsistency] = []
    degraded: set[str] = set()

    if pm_book_age_ms is not None and pm_book_age_ms > limits.max_feed_age_ms:
        issues.append(
            Inconsistency(
                InconsistencyKind.STALE_FEED, "pm_book", float(pm_book_age_ms),
                float(limits.max_feed_age_ms),
                f"CLOB book is {pm_book_age_ms}ms old",
            )
        )
        degraded |= {"pm_book", "pm_trades"}

    if ref_quote_age_ms is not None and ref_quote_age_ms > limits.max_feed_age_ms:
        issues.append(
            Inconsistency(
                InconsistencyKind.STALE_FEED, "ref_quote", float(ref_quote_age_ms),
                float(limits.max_feed_age_ms),
                f"reference quote is {ref_quote_age_ms}ms old",
            )
        )
        degraded |= {"ref_quote", "ref_trades"}

    if pm_spread is not None and pm_spread > limits.max_pm_spread:
        issues.append(
            Inconsistency(
                InconsistencyKind.ABNORMAL_SPREAD, "pm_book", pm_spread,
                limits.max_pm_spread,
                f"CLOB spread {pm_spread:.3f} is abnormally wide",
            )
        )
        degraded.add("pm_book")

    if ref_spread_bps is not None and ref_spread_bps > limits.max_ref_spread_bps:
        issues.append(
            Inconsistency(
                InconsistencyKind.ABNORMAL_SPREAD, "ref_quote", ref_spread_bps,
                limits.max_ref_spread_bps,
                f"reference spread {ref_spread_bps:.1f}bps is abnormally wide",
            )
        )
        degraded.add("ref_quote")

    if abs(clock_offset_ms) > limits.max_timestamp_drift_ms:
        issues.append(
            Inconsistency(
                InconsistencyKind.TIMESTAMP_DRIFT, "clock", float(clock_offset_ms),
                float(limits.max_timestamp_drift_ms),
                f"local clock offset {clock_offset_ms:.0f}ms",
            )
        )
        # A bad clock contaminates every windowed computation.
        degraded |= {"clock", "pm_trades", "ref_trades"}

    if settlement_price and reference_price:
        divergence_bps = abs(settlement_price - reference_price) / reference_price * 10_000
        if divergence_bps > limits.max_price_divergence_bps:
            issues.append(
                Inconsistency(
                    InconsistencyKind.PRICE_DIVERGENCE, "settlement", divergence_bps,
                    limits.max_price_divergence_bps,
                    f"settlement source and reference differ by {divergence_bps:.1f}bps",
                )
            )
            degraded |= {"ref_trades", "window_open_price"}

    affected: set[str] = set()
    for source in degraded:
        affected |= graph.impact.get(source, frozenset())

    for issue in issues:
        consistency_issues.inc(kind=issue.kind.value, source=issue.source)
        log.warning("consistency.issue", detail=str(issue), affected=len(affected))

    del now_ms
    return ConsistencyReport(
        issues=issues, degraded_sources=degraded, affected_features=frozenset(affected)
    )
