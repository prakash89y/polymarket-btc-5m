"""In-process metrics registry.

Deliberately dependency-free. The bot evaluates a market a few times a minute, so
a lock-protected dict is orders of magnitude more capacity than it needs, and
avoiding a client library keeps the container small and the failure modes
obvious. A Prometheus text rendering is provided so Module 11 can expose it
without anything else changing.

Labels are part of a metric's identity: ``markets_rejected{reason="stale_book"}``
and ``markets_rejected{reason="wide_spread"}`` are separate series. The
rejection-reason breakdown is the fastest answer to "why has the bot stopped
trading", so it is a first-class metric rather than a log grep.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

Labels = tuple[tuple[str, str], ...]


def _freeze(labels: Mapping[str, Any] | None) -> Labels:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _render_labels(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


@dataclass
class _Series:
    """One label combination of one metric."""

    value: float = 0.0
    count: int = 0
    total: float = 0.0
    #: Fixed buckets in milliseconds; parse latency lives on this scale.
    buckets: dict[float, int] = field(default_factory=dict)


DEFAULT_BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1_000, 5_000)


class Metric:
    """Base metric: a family of series keyed by label combination."""

    kind = "untyped"

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._series: dict[Labels, _Series] = {}
        self._lock = threading.Lock()

    def _get(self, labels: Labels) -> _Series:
        series = self._series.get(labels)
        if series is None:
            series = _Series()
            self._series[labels] = series
        return series

    def series(self) -> dict[Labels, _Series]:
        with self._lock:
            return dict(self._series)

    def reset(self) -> None:
        with self._lock:
            self._series.clear()


class Counter(Metric):
    """Monotonically increasing count."""

    kind = "counter"

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        with self._lock:
            self._get(_freeze(labels)).value += amount

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._get(_freeze(labels)).value


class Gauge(Metric):
    """A value that goes up and down (drift, open positions, bankroll)."""

    kind = "gauge"

    def set(self, value: float, **labels: Any) -> None:
        with self._lock:
            self._get(_freeze(labels)).value = value

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._get(_freeze(labels)).value


class Histogram(Metric):
    """Latency distribution with fixed buckets, plus count and total."""

    kind = "histogram"

    def __init__(
        self, name: str, description: str = "", buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS
    ) -> None:
        super().__init__(name, description)
        self.buckets = buckets

    def observe(self, value: float, **labels: Any) -> None:
        with self._lock:
            series = self._get(_freeze(labels))
            series.count += 1
            series.total += value
            for bound in self.buckets:
                if value <= bound:
                    series.buckets[bound] = series.buckets.get(bound, 0) + 1

    def mean(self, **labels: Any) -> float:
        with self._lock:
            series = self._get(_freeze(labels))
            return series.total / series.count if series.count else 0.0

    def count(self, **labels: Any) -> int:
        with self._lock:
            return self._get(_freeze(labels)).count

    @contextmanager
    def time(self, **labels: Any) -> Iterator[None]:
        """Time a block in milliseconds.

        Uses a monotonic clock: wall-clock time can step backwards, and a
        negative latency sample would poison the average silently.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe((time.perf_counter() - start) * 1000.0, **labels)


class MetricsRegistry:
    """Holds every metric in the process."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        return self._register(Counter(name, description))

    def gauge(self, name: str, description: str = "") -> Gauge:
        return self._register(Gauge(name, description))

    def histogram(
        self, name: str, description: str = "", buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS
    ) -> Histogram:
        return self._register(Histogram(name, description, buckets))

    def _register(self, metric: Any) -> Any:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                if type(existing) is not type(metric):
                    raise TypeError(
                        f"Metric {metric.name!r} already registered as {existing.kind}"
                    )
                return existing
            self._metrics[metric.name] = metric
            return metric

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict view, for the dashboard, logs, and tests."""
        out: dict[str, Any] = {}
        with self._lock:
            metrics = dict(self._metrics)
        for name, metric in metrics.items():
            entries: dict[str, Any] = {}
            for labels, series in metric.series().items():
                key = _render_labels(labels) or "_"
                if isinstance(metric, Histogram):
                    entries[key] = {
                        "count": series.count,
                        "total_ms": round(series.total, 3),
                        "mean_ms": round(series.total / series.count, 3) if series.count else 0.0,
                    }
                else:
                    entries[key] = series.value
            out[name] = entries
        return out

    def render_prometheus(self) -> str:
        """Prometheus text exposition, written by hand to avoid a dependency."""
        lines: list[str] = []
        with self._lock:
            metrics = dict(self._metrics)
        for name, metric in sorted(metrics.items()):
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.kind}")
            for labels, series in metric.series().items():
                rendered = _render_labels(labels)
                if isinstance(metric, Histogram):
                    cumulative = 0
                    for bound in metric.buckets:
                        cumulative = series.buckets.get(bound, 0)
                        bucket_labels = _render_labels((*labels, ("le", str(bound))))
                        lines.append(f"{name}_bucket{bucket_labels} {cumulative}")
                    lines.append(f"{name}_sum{rendered} {series.total}")
                    lines.append(f"{name}_count{rendered} {series.count}")
                else:
                    lines.append(f"{name}{rendered} {series.value}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Test helper."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()


METRICS = MetricsRegistry()

# --------------------------------------------------------------------------- #
# Parser / discovery metrics (Module 3)
# --------------------------------------------------------------------------- #
markets_discovered = METRICS.counter(
    "pmbtc_markets_discovered_total", "Markets returned by discovery."
)
markets_accepted = METRICS.counter(
    "pmbtc_markets_accepted_total", "Markets that passed parsing, health, and verification."
)
markets_rejected = METRICS.counter(
    "pmbtc_markets_rejected_total", "Markets rejected, by reason."
)
parse_latency = METRICS.histogram(
    "pmbtc_parse_latency_ms", "Time to parse one Gamma market into a settlement spec."
)
discovery_latency = METRICS.histogram(
    "pmbtc_discovery_latency_ms", "Time for one discovery scan, including network."
)
verification_failures = METRICS.counter(
    "pmbtc_verification_failures_total", "Settlement verification refusals, by status."
)
settlement_mismatches = METRICS.counter(
    "pmbtc_settlement_mismatches_total", "Detected changes in how a series resolves."
)
schema_mismatches = METRICS.counter(
    "pmbtc_schema_mismatches_total", "Gamma payload schema drift events, by kind."
)
http_requests = METRICS.counter("pmbtc_http_requests_total", "Outbound HTTP requests, by outcome.")
clock_drift_ms = METRICS.gauge("pmbtc_clock_drift_ms", "Estimated local clock offset vs reference.")
lifecycle_transitions = METRICS.counter(
    "pmbtc_lifecycle_transitions_total", "Market lifecycle state transitions."
)
