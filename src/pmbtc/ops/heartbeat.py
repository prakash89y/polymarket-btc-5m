"""Liveness heartbeat.

The service writes a small JSON file on every status tick. Anything that wants
to know "is collection actually running" reads that file rather than inspecting
process tables, which is both simpler and honest about the distinction that
matters: a process that is *running* but no longer *collecting* is exactly the
failure worth alerting on, and a stale heartbeat catches it where a PID check
would not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.utils.timeutils import isoformat, utc_now_ms

log = get_logger("pmbtc.ops.heartbeat")


@dataclass(frozen=True, slots=True)
class Heartbeat:
    written_at_ms: int
    pid: int
    sessions: int
    snapshots: int
    labels: int
    errors: int
    archived_frames: int
    clock_status: str
    feeds: list[dict[str, Any]]

    def age_ms(self, now_ms: int | None = None) -> int:
        return (now_ms or utc_now_ms()) - self.written_at_ms

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "written_at_ms": self.written_at_ms,
            "written_at": isoformat(self.written_at_ms),
            "pid": self.pid,
            "sessions": self.sessions,
            "snapshots": self.snapshots,
            "labels": self.labels,
            "errors": self.errors,
            "archived_frames": self.archived_frames,
            "clock_status": self.clock_status,
            "feeds": self.feeds,
        }
        return payload


def heartbeat_path(config: Config) -> Path:
    return config.resolved_path(config.app.data_dir) / "heartbeat.json"


def write_heartbeat(config: Config, payload: dict[str, Any]) -> None:
    """Write atomically: a torn heartbeat would read as a dead service."""
    path = heartbeat_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_heartbeat(config: Config) -> Heartbeat | None:
    path = heartbeat_path(config)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("heartbeat.unreadable", error=str(exc))
        return None
    return Heartbeat(
        written_at_ms=int(data.get("written_at_ms", 0)),
        pid=int(data.get("pid", 0)),
        sessions=int(data.get("sessions", 0)),
        snapshots=int(data.get("snapshots", 0)),
        labels=int(data.get("labels", 0)),
        errors=int(data.get("errors", 0)),
        archived_frames=int(data.get("archived_frames", 0)),
        clock_status=str(data.get("clock_status", "unknown")),
        feeds=list(data.get("feeds", [])),
    )
