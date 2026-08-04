"""Alerting — only for the four conditions worth waking someone for.

Alert fatigue is the failure mode here. A system that pages on every reconnect
teaches its operator to ignore it, and then the one alert that mattered is
ignored too. So this module fires on exactly four conditions:

1. **Collection stopped** — the heartbeat is stale or absent.
2. **Feed health degraded** — a feed is disconnected, its p95 latency has blown
   out, or its heartbeat has aged past budget.
3. **Readiness regressed** — a threshold that was previously met no longer is.
   Progress going *backwards* means something is deleting or corrupting data,
   which is far more serious than progress being slow.
4. **Data quality below threshold** — the share of low-quality snapshots has
   crossed the configured line.

5. **Supervisor restart failing** — the collector will not stay up across
   repeated restarts. Distinct from (1): collection stopped is the symptom, and
   this says the automatic remedy has already been tried and is not working.

Everything else is a log line. Alerts are deduplicated by key so a persistent
condition notifies once and then goes quiet until it clears.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.utils.timeutils import isoformat, utc_now_ms

log = get_logger("pmbtc.ops.alerts")

alerts_fired = METRICS.counter("pmbtc_alerts_total", "Alerts raised, by kind.")

#: Readiness metrics that only ever grow. A fall in one of these means data was
#: lost or corrupted. The others are ratios and are expected to fluctuate.
MONOTONIC_READINESS_METRICS: frozenset[str] = frozenset(
    {"labelled_markets", "complete_timelines", "min_samples"}
)


class AlertKind(StrEnum):
    COLLECTION_STOPPED = "collection_stopped"
    FEED_DEGRADED = "feed_degraded"
    READINESS_REGRESSED = "readiness_regressed"
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    #: The supervisor cannot keep the collector up. Raised by
    #: :mod:`pmbtc.ops.supervisor`, dispatched through this same engine.
    SUPERVISOR_RESTART_FAILING = "supervisor_restart_failing"


class Severity(StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Alert:
    kind: AlertKind
    severity: Severity
    message: str
    detail: dict[str, Any]
    raised_at_ms: int

    @property
    def key(self) -> str:
        """Dedup key: one alert per kind per subject."""
        subject = self.detail.get("feed") or self.detail.get("check") or "-"
        return f"{self.kind.value}:{subject}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
            "raised_at_ms": self.raised_at_ms,
            "raised_at": isoformat(self.raised_at_ms),
        }


class AlertState:
    """Remembers what is already firing, so a standing problem alerts once."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, int] = {}
        self.readiness_high_water: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.active = dict(data.get("active", {}))
        self.readiness_high_water = dict(data.get("readiness_high_water", {}))

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {"active": self.active, "readiness_high_water": self.readiness_high_water},
                indent=2,
            ),
            encoding="utf-8",
        )

    def should_fire(self, alert: Alert) -> bool:
        return alert.key not in self.active

    def mark_fired(self, alert: Alert) -> None:
        self.active[alert.key] = alert.raised_at_ms

    def clear(self, key: str) -> None:
        self.active.pop(key, None)


#: The conditions :func:`evaluate` itself produces, and therefore the only ones
#: it is entitled to retract. Any other key in the alert state belongs to a
#: different producer (today: the supervisor) and must be left alone.
EVALUATED_KINDS = frozenset(
    {
        AlertKind.COLLECTION_STOPPED.value,
        AlertKind.FEED_DEGRADED.value,
        AlertKind.READINESS_REGRESSED.value,
        AlertKind.QUALITY_BELOW_THRESHOLD.value,
    }
)


def evaluate(
    config: Config,
    state: AlertState,
    *,
    heartbeat: Any,
    summary: Any,
) -> list[Alert]:
    """Return the alerts that should fire now."""
    now = utc_now_ms()
    alerts: list[Alert] = []
    fired_keys: set[str] = set()

    # 1. Collection stopped -------------------------------------------- #
    stale_after = config.service.status_interval_seconds * 3 * 1000
    if heartbeat is None:
        alerts.append(
            Alert(
                AlertKind.COLLECTION_STOPPED,
                Severity.CRITICAL,
                "No heartbeat file — the collector has never run or was removed",
                {},
                now,
            )
        )
    elif heartbeat.age_ms(now) > stale_after:
        alerts.append(
            Alert(
                AlertKind.COLLECTION_STOPPED,
                Severity.CRITICAL,
                f"Heartbeat is {heartbeat.age_ms(now) / 1000:.0f}s old "
                f"(budget {stale_after / 1000:.0f}s)",
                {"age_ms": heartbeat.age_ms(now), "pid": heartbeat.pid},
                now,
            )
        )

    # 2. Feed health ---------------------------------------------------- #
    if heartbeat is not None:
        for feed in heartbeat.feeds:
            name = str(feed.get("name") or feed.get("slug") or "?")
            state_name = str(feed.get("state", "unknown"))
            detail = {"feed": name, **feed}
            if state_name not in {"live", "connecting"}:
                alerts.append(
                    Alert(
                        AlertKind.FEED_DEGRADED,
                        Severity.WARNING,
                        f"Feed {name} is {state_name}",
                        detail,
                        now,
                    )
                )
                continue
            p95 = feed.get("p95_latency_ms")
            budget = config.feeds.clob_staleness_budget_ms / 3
            if isinstance(p95, (int, float)) and p95 > budget:
                alerts.append(
                    Alert(
                        AlertKind.FEED_DEGRADED,
                        Severity.WARNING,
                        f"Feed {name} p95 latency {p95:.0f}ms exceeds {budget:.0f}ms",
                        detail,
                        now,
                    )
                )

    # 3. Readiness regression ------------------------------------------- #
    # Only for metrics that *accumulate*. Counts going backwards means data is
    # being lost or corrupted. Ratios like class balance and feature coverage
    # legitimately move both ways as new markets arrive, and alerting on them
    # would be noise — which is how an operator learns to ignore the channel.
    for check in summary.readiness.checks:
        if check.name not in MONOTONIC_READINESS_METRICS:
            continue
        previous = state.readiness_high_water.get(check.name)
        if previous is not None and check.actual < previous * 0.98:
            alerts.append(
                Alert(
                    AlertKind.READINESS_REGRESSED,
                    Severity.CRITICAL,
                    f"Readiness metric {check.name} fell from {previous:g} to {check.actual:g}",
                    {"check": check.name, "previous": previous, "current": check.actual},
                    now,
                )
            )
        state.readiness_high_water[check.name] = max(previous or 0.0, check.actual)

    # 4. Data quality ---------------------------------------------------- #
    distribution = summary.stats.quality_distribution
    total = sum(distribution.values())
    if total:
        low = distribution.get("0.0-0.5", 0) + distribution.get("zero", 0)
        fraction = low / total
        if fraction > config.training.max_low_quality_fraction:
            alerts.append(
                Alert(
                    AlertKind.QUALITY_BELOW_THRESHOLD,
                    Severity.WARNING,
                    f"{fraction:.1%} of snapshots are below the quality floor "
                    f"(limit {config.training.max_low_quality_fraction:.1%})",
                    {"low": low, "total": total, "fraction": round(fraction, 4)},
                    now,
                )
            )

    # Clear only the conditions *this function* evaluates.
    #
    # It is not the sole producer any more: v0.13.0 added
    # `pmbtc.ops.supervisor`, which raises SUPERVISOR_RESTART_FAILING through
    # the same state file. Clearing every key not re-raised here silently
    # erased that alert on the collector's next status tick — so on 2026-08-03
    # a supervisor at 24 consecutive restart failures fired once and then
    # vanished from `pmbtc watch`, leaving the operator with no standing signal
    # while the restart loop continued. A producer may only retract its own
    # conditions.
    fired_keys = {a.key for a in alerts}
    for key in list(state.active):
        if key.split(":", 1)[0] not in EVALUATED_KINDS:
            continue  # owned by another producer; not ours to retract
        if key not in fired_keys:
            # Condition cleared: forget it so a recurrence alerts again.
            state.clear(key)
    return alerts


def dispatch(config: Config, state: AlertState, alerts: list[Alert]) -> list[Alert]:
    """Log and record new alerts. Returns the ones actually fired."""
    fired: list[Alert] = []
    for alert in alerts:
        if not state.should_fire(alert):
            continue
        state.mark_fired(alert)
        fired.append(alert)
        alerts_fired.inc(kind=alert.kind.value, severity=alert.severity.value)
        emit = log.error if alert.severity is Severity.CRITICAL else log.warning
        emit("alert", **alert.as_dict())
    state.save()
    del config
    return fired


def alert_state_path(config: Config) -> Path:
    return config.resolved_path(config.app.data_dir) / "alert_state.json"
