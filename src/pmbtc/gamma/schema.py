"""Gamma payload schema contracts, fingerprinting, and drift detection.

The parser reads fields by name out of a JSON document owned by someone else.
That is a standing hazard: if ``endDate`` is renamed, or ``orderMinSize`` starts
arriving as a string, a naive parser keeps running and produces subtly wrong
specifications. This module makes that impossible to miss.

Three severities, because they need three different responses:

``BREAKING``
    A required field vanished, or changed type. The payload can no longer be
    trusted. Parsing raises; trading stops. Silence here would be the worst
    possible outcome, so it is the loudest.
``CHANGED``
    A field we treat as optional changed type, or a known field disappeared.
    Recorded, alerted, and the market is rejected -- but the process lives.
``ADDITIVE``
    New fields appeared. Polymarket adds fields routinely; halting on this
    would be a self-inflicted outage. Recorded and alerted only.

Every distinct field/type shape gets a fingerprint and a monotonically
increasing schema version, persisted, so "when did this change?" is answerable
after the fact rather than guessed at.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pmbtc.exceptions import DataError
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import schema_mismatches
from pmbtc.utils.timeutils import utc_now_ms

log = get_logger("pmbtc.gamma.schema")


class DriftSeverity(StrEnum):
    ADDITIVE = "additive"
    CHANGED = "changed"
    BREAKING = "breaking"


class SchemaViolation(DataError):
    """A payload no longer satisfies the contract the parser was written for."""

    halts_trading = True


@dataclass(frozen=True, slots=True)
class FieldContract:
    """One field the parser depends on."""

    name: str
    #: Accepted JSON types. ``str`` covers Gamma's habit of serialising numbers
    #: and lists as strings (``outcomes`` is a JSON-encoded string, not a list).
    types: tuple[type, ...]
    required: bool = True
    description: str = ""

    def check(self, payload: dict[str, Any]) -> str | None:
        if self.name not in payload:
            return "missing" if self.required else None
        value = payload[self.name]
        if value is None:
            return None if not self.required else "null"
        if not isinstance(value, self.types):
            names = "/".join(t.__name__ for t in self.types)
            return f"expected {names}, got {type(value).__name__}"
        return None


#: The contract for a Gamma *market* object, as verified against live payloads
#: on 2026-07-31. Only fields the parser actually reads appear here: a contract
#: that covers fields nobody uses generates false alarms.
MARKET_CONTRACT: tuple[FieldContract, ...] = (
    FieldContract("conditionId", (str,), True, "on-chain condition identifier"),
    FieldContract("id", (str, int), True, "Gamma market id"),
    FieldContract("slug", (str,), True, "market slug"),
    FieldContract("question", (str,), False, "human-readable title"),
    FieldContract("description", (str,), True, "resolution rules prose"),
    FieldContract("resolutionSource", (str,), True, "structured settlement source"),
    FieldContract("endDate", (str,), True, "settlement instant, ISO-8601 UTC"),
    FieldContract("outcomes", (str, list), True, "JSON-encoded outcome names"),
    FieldContract("outcomePrices", (str, list), False, "JSON-encoded prices"),
    FieldContract("closed", (bool,), False, "market closed flag"),
    FieldContract("active", (bool,), False, "market active flag"),
    FieldContract("acceptingOrders", (bool,), False, "venue accepting orders"),
    FieldContract("enableOrderBook", (bool,), False, "CLOB enabled"),
    FieldContract("orderPriceMinTickSize", (float, int), False, "tick size"),
    FieldContract("orderMinSize", (float, int), False, "minimum order size"),
    FieldContract("clobTokenIds", (str, list), False, "outcome token ids"),
    FieldContract("bestBid", (float, int), False, "best bid"),
    FieldContract("bestAsk", (float, int), False, "best ask"),
    FieldContract("spread", (float, int), False, "quoted spread"),
    FieldContract("lastTradePrice", (float, int), False, "last trade"),
    FieldContract("volume", (str, float, int), False, "traded volume"),
    FieldContract("liquidity", (str, float, int), False, "resting liquidity"),
    FieldContract("feesEnabled", (bool,), False, "fees active on this market"),
    FieldContract("events", (list,), True, "embedded event, carrying the series"),
)

#: Contract for the embedded event object. ``startTime`` is load-bearing: it is
#: the window open, and the parser cross-checks it against the slug epoch.
EVENT_CONTRACT: tuple[FieldContract, ...] = (
    FieldContract("slug", (str,), True, "event slug"),
    FieldContract("title", (str,), False, "event title"),
    FieldContract("startTime", (str,), True, "window open, ISO-8601 UTC"),
    FieldContract("endDate", (str,), False, "window close"),
    FieldContract("resolutionSource", (str,), False, "mirrored settlement source"),
    FieldContract("series", (list,), True, "series membership"),
)

SERIES_CONTRACT: tuple[FieldContract, ...] = (
    FieldContract("slug", (str,), True, "series slug -- the family identity"),
    FieldContract("recurrence", (str,), True, "window cadence, e.g. '5m'"),
)


@dataclass(frozen=True, slots=True)
class SchemaDrift:
    """One detected difference between a payload and the contract."""

    severity: DriftSeverity
    scope: str
    field_name: str
    detail: str

    def __str__(self) -> str:
        return f"{self.severity.value}: {self.scope}.{self.field_name} ({self.detail})"


@dataclass
class SchemaCheckResult:
    fingerprint: str
    version: int
    drifts: tuple[SchemaDrift, ...]
    known: bool

    @property
    def breaking(self) -> tuple[SchemaDrift, ...]:
        return tuple(d for d in self.drifts if d.severity is DriftSeverity.BREAKING)

    @property
    def changed(self) -> tuple[SchemaDrift, ...]:
        return tuple(d for d in self.drifts if d.severity is DriftSeverity.CHANGED)

    @property
    def additive(self) -> tuple[SchemaDrift, ...]:
        return tuple(d for d in self.drifts if d.severity is DriftSeverity.ADDITIVE)

    @property
    def compatible(self) -> bool:
        """Safe to parse: nothing breaking, nothing changed."""
        return not self.breaking and not self.changed


def _fingerprint(payload: dict[str, Any]) -> str:
    """Hash the *shape* of a payload: field names and JSON types, not values."""
    shape = sorted(f"{k}:{type(v).__name__}" for k, v in payload.items())
    event = (payload.get("events") or [{}])[0]
    if isinstance(event, dict):
        shape += sorted(f"events[].{k}:{type(v).__name__}" for k, v in event.items())
        series = (event.get("series") or [{}])[0]
        if isinstance(series, dict):
            shape += sorted(
                f"events[].series[].{k}:{type(v).__name__}" for k, v in series.items()
            )
    return hashlib.sha256("|".join(shape).encode("utf-8")).hexdigest()[:16]


def _check_contract(
    payload: dict[str, Any], contract: tuple[FieldContract, ...], scope: str
) -> list[SchemaDrift]:
    drifts: list[SchemaDrift] = []
    for spec in contract:
        problem = spec.check(payload)
        if problem is None:
            continue
        severity = DriftSeverity.BREAKING if spec.required else DriftSeverity.CHANGED
        drifts.append(SchemaDrift(severity, scope, spec.name, problem))
    return drifts


def _unknown_fields(
    payload: dict[str, Any], contract: tuple[FieldContract, ...], scope: str
) -> list[SchemaDrift]:
    known = {c.name for c in contract}
    return [
        SchemaDrift(DriftSeverity.ADDITIVE, scope, name, "new field")
        for name in sorted(set(payload) - known)
    ]


class SchemaRegistry:
    """Persisted record of every payload shape we have seen.

    Stored as a small JSON document rather than JSONL: it is read on every scan
    and rewritten rarely, and being able to open it and read the whole history
    at once is worth more than append-only semantics here.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, dict[str, Any]] = {}
        self._next_version = 1
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("schema.registry_unreadable", path=str(self.path), error=str(exc))
            return
        self._versions = data.get("versions", {})
        self._next_version = int(data.get("next_version", len(self._versions) + 1))

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {"next_version": self._next_version, "versions": self._versions}, indent=2
            ),
            encoding="utf-8",
        )

    def version_for(self, fingerprint: str) -> tuple[int, bool]:
        """``(version, already_known)`` for a payload shape."""
        entry = self._versions.get(fingerprint)
        if entry is not None:
            entry["last_seen_ms"] = utc_now_ms()
            entry["seen_count"] = int(entry.get("seen_count", 0)) + 1
            self._save()
            return int(entry["version"]), True
        version = self._next_version
        self._versions[fingerprint] = {
            "version": version,
            "first_seen_ms": utc_now_ms(),
            "last_seen_ms": utc_now_ms(),
            "seen_count": 1,
        }
        self._next_version += 1
        self._save()
        return version, False

    @property
    def known_count(self) -> int:
        return len(self._versions)


@dataclass
class SchemaChecker:
    """Validates payloads against the contracts and tracks shape versions."""

    registry: SchemaRegistry
    strict: bool = True
    alert_on_new_fields: bool = True
    _alerted: set[str] = field(default_factory=set)

    def check(self, market: dict[str, Any]) -> SchemaCheckResult:
        drifts: list[SchemaDrift] = []
        drifts += _check_contract(market, MARKET_CONTRACT, "market")

        events = market.get("events")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            event = events[0]
            drifts += _check_contract(event, EVENT_CONTRACT, "events[]")
            series = event.get("series")
            if isinstance(series, list) and series and isinstance(series[0], dict):
                drifts += _check_contract(series[0], SERIES_CONTRACT, "events[].series[]")
                drifts += _unknown_fields(series[0], SERIES_CONTRACT, "events[].series[]")
            else:
                drifts.append(
                    SchemaDrift(
                        DriftSeverity.BREAKING, "events[]", "series", "missing or empty"
                    )
                )
            drifts += _unknown_fields(event, EVENT_CONTRACT, "events[]")
        drifts += _unknown_fields(market, MARKET_CONTRACT, "market")

        fingerprint = _fingerprint(market)
        version, known = self.registry.version_for(fingerprint)
        result = SchemaCheckResult(fingerprint, version, tuple(drifts), known)
        self._report(result)
        return result

    def _report(self, result: SchemaCheckResult) -> None:
        for drift in result.breaking:
            schema_mismatches.inc(kind="breaking")
            log.error("schema.breaking", drift=str(drift), fingerprint=result.fingerprint)
        for drift in result.changed:
            schema_mismatches.inc(kind="changed")
            log.warning("schema.changed", drift=str(drift), fingerprint=result.fingerprint)
        if result.additive and self.alert_on_new_fields and not result.known:
            # Alert once per shape, not once per market: at one scan every 30
            # seconds, per-market alerting would be pure noise.
            key = result.fingerprint
            if key not in self._alerted:
                self._alerted.add(key)
                schema_mismatches.inc(kind="additive")
                log.info(
                    "schema.new_fields",
                    version=result.version,
                    fingerprint=result.fingerprint,
                    fields=[d.field_name for d in result.additive],
                )

    def check_or_raise(self, market: dict[str, Any]) -> SchemaCheckResult:
        """Check, and refuse to proceed on an incompatible payload."""
        result = self.check(market)
        if result.breaking or (self.strict and result.changed):
            raise SchemaViolation(
                "Gamma payload is incompatible with the parser contract",
                context={
                    "fingerprint": result.fingerprint,
                    "schema_version": result.version,
                    "drifts": "; ".join(
                        str(d) for d in (*result.breaking, *result.changed)
                    ),
                },
            )
        return result


def open_schema_checker(config: Any) -> SchemaChecker:
    """Build a checker from configuration."""
    cfg = config.schema_check
    registry = SchemaRegistry(config.resolved_path(cfg.registry_file))
    return SchemaChecker(
        registry=registry, strict=cfg.strict, alert_on_new_fields=cfg.alert_on_new_fields
    )
