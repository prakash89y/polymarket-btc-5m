"""Dataset statistics.

The report answers the questions you ask before trusting a dataset: how much of
it is there, how much is missing, is it balanced, how late is the data, and is
it all from the settlement source I think it is.

Class balance is the one to read first. In a market this close to fair, a
strong Up/Down imbalance in the sample is far more likely to mean a collection
artefact -- one session over-represented, a bad stretch of downtime -- than a
real edge sitting in plain sight.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord, QualityFlag
from pmbtc.dataset.store import DatasetStore
from pmbtc.utils.timeutils import isoformat


@dataclass
class DatasetStats:
    markets: int = 0
    labelled: int = 0
    unlabelled: int = 0
    void: int = 0
    snapshots: int = 0
    period_start_ms: int | None = None
    period_end_ms: int | None = None
    class_balance: dict[str, int] = field(default_factory=dict)
    settlement_providers: dict[str, int] = field(default_factory=dict)
    snapshots_by_horizon: dict[int, int] = field(default_factory=dict)
    timeline_completeness: float = 0.0
    feature_coverage: dict[str, float] = field(default_factory=dict)
    missing_pct_by_source: dict[str, float] = field(default_factory=dict)
    mean_latency_by_source: dict[str, float] = field(default_factory=dict)
    quality_distribution: dict[str, int] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)
    mean_jitter_ms: float = 0.0

    @property
    def label_rate(self) -> float:
        return self.labelled / self.markets if self.markets else 0.0

    @property
    def up_rate(self) -> float | None:
        up = self.class_balance.get("up", 0)
        down = self.class_balance.get("down", 0)
        total = up + down
        return up / total if total else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "markets": self.markets,
            "labelled": self.labelled,
            "unlabelled": self.unlabelled,
            "void": self.void,
            "label_rate": round(self.label_rate, 4),
            "snapshots": self.snapshots,
            "period": {
                "start": isoformat(self.period_start_ms) if self.period_start_ms else None,
                "end": isoformat(self.period_end_ms) if self.period_end_ms else None,
            },
            "class_balance": self.class_balance,
            "up_rate": round(self.up_rate, 4) if self.up_rate is not None else None,
            "settlement_providers": self.settlement_providers,
            "snapshots_by_horizon": self.snapshots_by_horizon,
            "timeline_completeness": round(self.timeline_completeness, 4),
            "feature_coverage": {k: round(v, 4) for k, v in self.feature_coverage.items()},
            "missing_pct_by_source": {
                k: round(v, 4) for k, v in self.missing_pct_by_source.items()
            },
            "mean_latency_ms_by_source": {
                k: round(v, 1) for k, v in self.mean_latency_by_source.items()
            },
            "quality_distribution": self.quality_distribution,
            "flag_counts": self.flag_counts,
            "mean_snapshot_jitter_ms": round(self.mean_jitter_ms, 1),
        }


def _quality_bucket(score: float) -> str:
    if score >= 0.9:
        return "0.9-1.0"
    if score >= 0.75:
        return "0.75-0.9"
    if score >= 0.5:
        return "0.5-0.75"
    if score > 0.0:
        return "0.0-0.5"
    return "zero"


def compute_stats(
    markets: list[MarketRecord], snapshots: list[FeatureSnapshot], horizons: tuple[int, ...] = ()
) -> DatasetStats:
    stats = DatasetStats()
    stats.markets = len(markets)
    stats.snapshots = len(snapshots)

    for market in markets:
        if market.is_labelled:
            stats.labelled += 1
            key = market.official_outcome.value if market.official_outcome else "void"
            stats.class_balance[key] = stats.class_balance.get(key, 0) + 1
        elif market.resolved_at_ms is not None:
            # Resolved but with no outcome: a 50-50 void. Real, and correctly
            # excluded from classification.
            stats.void += 1
        else:
            stats.unlabelled += 1

        provider = market.settlement_provider.value
        stats.settlement_providers[provider] = stats.settlement_providers.get(provider, 0) + 1

    settlement_times = [m.settlement_time_ms for m in markets]
    if settlement_times:
        stats.period_start_ms = min(settlement_times)
        stats.period_end_ms = max(settlement_times)

    # --- snapshots ---------------------------------------------------- #
    by_horizon: Counter[int] = Counter()
    jitters: list[int] = []
    feature_present: Counter[str] = Counter()
    feature_total: Counter[str] = Counter()
    source_missing: Counter[str] = Counter()
    source_total: Counter[str] = Counter()
    source_latency: dict[str, list[int]] = {}
    flags: Counter[str] = Counter()
    quality_buckets: Counter[str] = Counter()

    for snapshot in snapshots:
        by_horizon[snapshot.horizon_seconds] += 1
        jitters.append(snapshot.scheduling_jitter_ms)
        quality_buckets[_quality_bucket(snapshot.quality_score)] += 1
        for obs in snapshot.observations:
            feature_total[obs.name] += 1
            source_total[obs.data_source] += 1
            if obs.value is None:
                source_missing[obs.data_source] += 1
            else:
                feature_present[obs.name] += 1
                source_latency.setdefault(obs.data_source, []).append(obs.latency_ms)
            for flag in obs.flags:
                if flag is not QualityFlag.OK:
                    flags[flag.value] += 1

    stats.snapshots_by_horizon = dict(sorted(by_horizon.items(), reverse=True))
    stats.mean_jitter_ms = sum(jitters) / len(jitters) if jitters else 0.0
    stats.feature_coverage = {
        name: feature_present[name] / total for name, total in feature_total.items() if total
    }
    stats.missing_pct_by_source = {
        source: source_missing[source] / total for source, total in source_total.items() if total
    }
    stats.mean_latency_by_source = {
        source: sum(values) / len(values) for source, values in source_latency.items() if values
    }
    stats.flag_counts = dict(flags)
    stats.quality_distribution = dict(quality_buckets)

    expected = horizons or tuple(sorted(by_horizon, reverse=True))
    if markets and expected:
        stats.timeline_completeness = len(snapshots) / (len(markets) * len(expected))

    return stats


def stats_for_store(store: DatasetStore, horizons: tuple[int, ...] = ()) -> DatasetStats:
    return compute_stats(store.markets(), store.snapshots(), horizons)


def render_stats(stats: DatasetStats, console: Console | None = None) -> None:
    """Operator-facing report."""
    console = console or Console()

    overview = Table(title="Dataset overview", show_lines=False)
    overview.add_column("metric")
    overview.add_column("value")
    rows: list[tuple[str, str]] = [
        ("markets", str(stats.markets)),
        ("labelled", f"{stats.labelled} ({stats.label_rate:.1%})"),
        ("awaiting label", str(stats.unlabelled)),
        ("void (50-50)", str(stats.void)),
        ("snapshots", str(stats.snapshots)),
        ("timeline completeness", f"{stats.timeline_completeness:.1%}"),
        ("mean snapshot jitter", f"{stats.mean_jitter_ms:.0f} ms"),
    ]
    if stats.period_start_ms and stats.period_end_ms:
        rows.append(
            ("period", f"{isoformat(stats.period_start_ms)} -> {isoformat(stats.period_end_ms)}")
        )
    if stats.up_rate is not None:
        rows.append(("class balance", f"up {stats.up_rate:.1%} / down {1 - stats.up_rate:.1%}"))
    for name, value in rows:
        overview.add_row(name, value)
    console.print(overview)

    if stats.settlement_providers:
        providers = Table(title="Settlement providers", show_lines=False)
        providers.add_column("provider")
        providers.add_column("markets")
        for provider, count in sorted(stats.settlement_providers.items()):
            providers.add_row(provider, str(count))
        console.print(providers)

    if stats.snapshots_by_horizon:
        horizons = Table(title="Snapshots by horizon", show_lines=False)
        horizons.add_column("T-seconds")
        horizons.add_column("count")
        for horizon, count in stats.snapshots_by_horizon.items():
            horizons.add_row(f"T-{horizon}", str(count))
        console.print(horizons)

    if stats.missing_pct_by_source:
        sources = Table(title="Sources", show_lines=False)
        for column in ("source", "missing", "mean latency"):
            sources.add_column(column)
        for source in sorted(stats.missing_pct_by_source):
            latency = stats.mean_latency_by_source.get(source)
            sources.add_row(
                source,
                f"{stats.missing_pct_by_source[source]:.1%}",
                f"{latency:.0f} ms" if latency is not None else "-",
            )
        console.print(sources)

    if stats.quality_distribution:
        quality = Table(title="Snapshot quality distribution", show_lines=False)
        quality.add_column("bucket")
        quality.add_column("snapshots")
        for bucket in ("0.9-1.0", "0.75-0.9", "0.5-0.75", "0.0-0.5", "zero"):
            if bucket in stats.quality_distribution:
                quality.add_row(bucket, str(stats.quality_distribution[bucket]))
        console.print(quality)

    if stats.flag_counts:
        console.print("quality flags: " + ", ".join(
            f"{name}={count}" for name, count in sorted(stats.flag_counts.items())
        ))
