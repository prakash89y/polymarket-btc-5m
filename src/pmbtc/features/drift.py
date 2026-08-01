"""Feature drift monitoring.

Two kinds of change, detected differently because they look different:

**Gradual drift** — the distribution moves slowly over weeks. Caught by
comparing a long reference window against a recent one with the
population-stability index (PSI), which is sensitive to shifts in the body of a
distribution.

**Sudden drift** — a level shift or a variance explosion within minutes,
typically a venue change or a broken feed. Caught by a z-score on the recent
mean against the reference mean and variance.

Alerts, never automatic retraining. A distribution shift might mean the market
changed, or it might mean a data source broke — and retraining on broken data is
strictly worse than not retraining at all. A human decides.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from math import log as ln
from math import sqrt
from typing import Any

from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS

# Named `logger`, not `log`: this module needs `math.log`, and a logger bound to
# that name silently shadows it.
logger = get_logger("pmbtc.features.drift")

drift_alerts = METRICS.counter("pmbtc_feature_drift_alerts_total", "Feature drift alerts.")


class DriftKind(StrEnum):
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    COVERAGE = "coverage"


@dataclass(frozen=True, slots=True)
class DriftAlert:
    feature: str
    kind: DriftKind
    statistic: float
    threshold: float
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.feature}: {self.kind.value} drift "
            f"({self.statistic:.3f} vs {self.threshold:.3f}) — {self.detail}"
        )


def population_stability_index(
    reference: list[float], recent: list[float], bins: int = 10
) -> float:
    """PSI between two samples, using quantile bins from the reference.

    Rules of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant.
    Quantile bins rather than equal-width, so a skewed feature does not put
    every observation in one bucket and report zero drift.
    """
    if len(reference) < bins * 2 or len(recent) < bins:
        return 0.0
    ordered = sorted(reference)
    edges = [ordered[int(i * len(ordered) / bins)] for i in range(1, bins)]

    def distribute(sample: list[float]) -> list[float]:
        counts = [0] * bins
        for value in sample:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        total = len(sample)
        # Floor at a small epsilon: an empty bucket would make the log diverge.
        return [max(c / total, 1e-6) for c in counts]

    expected, actual = distribute(reference), distribute(recent)
    return sum((a - e) * ln(a / e) for e, a in zip(expected, actual, strict=True))


@dataclass
class FeatureWindow:
    """Rolling samples for one feature.

    ``reference`` holds strictly *older* observations than ``recent``: a value
    only enters the reference once it has been pushed out of the recent window.
    Letting the two overlap would make drift mask itself — the shifted values
    would sit in both samples and the comparison would show nothing.
    """

    reference: deque[float] = field(default_factory=lambda: deque(maxlen=5_000))
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    missing: int = 0
    seen: int = 0
    recent_capacity: int = 500

    def add(self, value: float | None) -> None:
        self.seen += 1
        if value is None:
            self.missing += 1
            return
        if len(self.recent) >= self.recent_capacity:
            # Age the oldest recent value into the reference baseline.
            self.reference.append(self.recent.popleft())
        self.recent.append(value)

    @property
    def coverage(self) -> float:
        return 1.0 - self.missing / self.seen if self.seen else 0.0


class DriftMonitor:
    """Tracks rolling distributions and raises alerts, never retrains."""

    def __init__(
        self,
        *,
        psi_threshold: float = 0.25,
        zscore_threshold: float = 4.0,
        min_coverage: float = 0.5,
        min_samples: int = 200,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.zscore_threshold = zscore_threshold
        self.min_coverage = min_coverage
        self.min_samples = min_samples
        self.windows: dict[str, FeatureWindow] = {}

    def observe(self, values: dict[str, float | None]) -> None:
        for name, value in values.items():
            self.windows.setdefault(name, FeatureWindow()).add(value)

    # ------------------------------------------------------------------ #
    def evaluate(self) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        for name, window in sorted(self.windows.items()):
            if window.seen < self.min_samples:
                continue

            if window.coverage < self.min_coverage:
                alerts.append(
                    DriftAlert(
                        name,
                        DriftKind.COVERAGE,
                        window.coverage,
                        self.min_coverage,
                        f"only {window.coverage:.1%} of observations populated",
                    )
                )
                continue

            reference = list(window.reference)
            recent = list(window.recent)
            if len(reference) < self.min_samples or len(recent) < 30:
                continue

            psi = population_stability_index(reference, recent)
            if psi > self.psi_threshold:
                alerts.append(
                    DriftAlert(
                        name, DriftKind.GRADUAL, psi, self.psi_threshold,
                        "distribution has shifted against the reference window",
                    )
                )

            mean_ref = sum(reference) / len(reference)
            variance = sum((v - mean_ref) ** 2 for v in reference) / max(1, len(reference) - 1)
            stdev = sqrt(variance)
            if stdev > 0:
                mean_recent = sum(recent) / len(recent)
                z = abs(mean_recent - mean_ref) / (stdev / sqrt(len(recent)))
                if z > self.zscore_threshold:
                    alerts.append(
                        DriftAlert(
                            name, DriftKind.SUDDEN, z, self.zscore_threshold,
                            f"recent mean {mean_recent:.4g} vs reference {mean_ref:.4g}",
                        )
                    )

        for alert in alerts:
            drift_alerts.inc(kind=alert.kind.value)
            logger.warning("feature.drift", detail=str(alert))
        return alerts

    def snapshot(self) -> dict[str, Any]:
        return {
            name: {
                "seen": w.seen,
                "coverage": round(w.coverage, 4),
                "reference_n": len(w.reference),
                "recent_n": len(w.recent),
            }
            for name, w in sorted(self.windows.items())
        }
