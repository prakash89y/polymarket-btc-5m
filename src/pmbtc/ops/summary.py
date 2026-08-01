"""Daily dataset summary and readiness progress.

Written once a day as JSON (for tooling and history) and rendered as a table
(for a human). Its job is to answer, in ten seconds: is the dataset growing, is
it healthy, and how far is it from being trainable.

Progress is expressed as a *fraction of each threshold* and, where the rate can
be estimated, as a projected completion date. "72% of the way to the labelled-
market threshold, roughly 4 days out" is actionable; "1,447 markets" is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from pmbtc.config import Config
from pmbtc.dataset.stats import DatasetStats, stats_for_store
from pmbtc.dataset.store import DatasetStore
from pmbtc.models.readiness import ReadinessReport, readiness_for_store
from pmbtc.utils.timeutils import DAY_MS, isoformat, utc_now_ms


@dataclass
class DailySummary:
    generated_at_ms: int
    stats: DatasetStats
    readiness: ReadinessReport
    #: Markets and snapshots added in the last 24 hours.
    markets_last_day: int = 0
    snapshots_last_day: int = 0
    labelled_last_day: int = 0
    feed_health: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def progress(self) -> dict[str, float]:
        """Fraction of each readiness threshold met, capped at 1.0."""
        return {
            check.name: min(1.0, check.actual / check.required) if check.required else 1.0
            for check in self.readiness.checks
        }

    def days_remaining(self, name: str) -> float | None:
        """Rough days until a threshold is met at the current daily rate."""
        rates = {
            "labelled_markets": self.labelled_last_day,
            "complete_timelines": self.markets_last_day,
            "min_samples": self.snapshots_last_day,
        }
        rate = rates.get(name)
        if not rate:
            return None
        check = next((c for c in self.readiness.checks if c.name == name), None)
        if check is None or check.passed:
            return 0.0
        return (check.required - check.actual) / rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at_ms": self.generated_at_ms,
            "generated_at": isoformat(self.generated_at_ms),
            "last_24h": {
                "markets": self.markets_last_day,
                "labelled": self.labelled_last_day,
                "snapshots": self.snapshots_last_day,
            },
            "dataset": self.stats.as_dict(),
            "readiness": {
                "ready": self.readiness.ready,
                "checks": [
                    {
                        "name": c.name,
                        "actual": c.actual,
                        "required": c.required,
                        "passed": c.passed,
                        "progress": round(self.progress.get(c.name, 0.0), 4),
                        "days_remaining": self.days_remaining(c.name),
                    }
                    for c in self.readiness.checks
                ],
            },
            "feed_health": self.feed_health,
        }

    def render(self, console: Console | None = None) -> None:
        console = console or Console()

        growth = Table(title="Collection — last 24 hours", show_lines=False)
        growth.add_column("metric")
        growth.add_column("value")
        growth.add_row("markets discovered", str(self.markets_last_day))
        growth.add_row("markets labelled", str(self.labelled_last_day))
        growth.add_row("snapshots captured", str(self.snapshots_last_day))
        growth.add_row("total labelled", str(self.stats.labelled))
        growth.add_row("total snapshots", str(self.stats.snapshots))
        if self.stats.up_rate is not None:
            growth.add_row("class balance", f"up {self.stats.up_rate:.1%}")
        console.print(growth)

        progress = Table(title="Readiness progress", show_lines=False)
        for column in ("threshold", "have", "need", "progress", "eta", "status"):
            progress.add_column(column)
        for check in self.readiness.checks:
            eta = self.days_remaining(check.name)
            progress.add_row(
                check.name,
                f"{check.actual:g}",
                f"{check.required:g}",
                f"{self.progress.get(check.name, 0.0):.1%}",
                "—" if eta is None else (f"{eta:.1f}d" if eta else "met"),
                "[green]PASS[/]" if check.passed else "[yellow]pending[/]",
            )
        console.print(progress)

        if self.stats.quality_distribution:
            quality = Table(title="Data quality distribution", show_lines=False)
            quality.add_column("bucket")
            quality.add_column("snapshots")
            for bucket, count in self.stats.quality_distribution.items():
                quality.add_row(bucket, str(count))
            console.print(quality)

        if self.feed_health:
            feeds = Table(title="Feed health", show_lines=False)
            for column in ("feed", "state", "frames", "p95 latency", "reconnects", "heartbeat"):
                feeds.add_column(column)
            for feed in self.feed_health:
                feeds.add_row(
                    str(feed.get("name", feed.get("slug", "?")))[:28],
                    str(feed.get("state", "?")),
                    str(feed.get("frames", 0)),
                    f"{feed.get('p95_latency_ms', '—')}",
                    str(feed.get("reconnects", 0)),
                    f"{feed.get('heartbeat_age_ms', '—')} ms",
                )
            console.print(feeds)


def build_summary(config: Config, store: DatasetStore) -> DailySummary:
    now = utc_now_ms()
    cutoff = now - DAY_MS
    markets = store.markets()
    snapshots = store.snapshots()

    from pmbtc.ops.heartbeat import read_heartbeat

    heartbeat = read_heartbeat(config)
    return DailySummary(
        generated_at_ms=now,
        stats=stats_for_store(store, tuple(config.dataset.snapshot_horizons_s)),
        readiness=readiness_for_store(config, store),
        markets_last_day=sum(1 for m in markets if m.discovery_time_ms >= cutoff),
        labelled_last_day=sum(
            1 for m in markets if m.resolved_at_ms and m.resolved_at_ms >= cutoff
        ),
        snapshots_last_day=sum(1 for s in snapshots if s.captured_at_ms >= cutoff),
        feed_health=heartbeat.feeds if heartbeat else [],
    )


def write_summary(config: Config, summary: DailySummary) -> Path:
    """Persist one day's summary. Append-only by filename, never overwritten."""
    directory = config.resolved_path(config.app.artifact_dir) / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    day = isoformat(summary.generated_at_ms)[:10]
    path = directory / f"summary-{day}.json"
    path.write_text(json.dumps(summary.as_dict(), indent=2, default=str), encoding="utf-8")
    return path
