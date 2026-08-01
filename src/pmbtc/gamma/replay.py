"""Replay archived Gamma responses through the current parser.

The requirement this satisfies is the strict one: *a parser change must never
alter historical interpretations without explicit review.*

The mechanism is a golden baseline. Every archived market is parsed and reduced
to its settlement fingerprint plus the handful of fields that decide a trade.
That reduction is written to a baseline file and committed. On every later run,
replay recomputes it and diffs. A changed interpretation shows up as a named
difference on a named market, and the only way to accept it is to regenerate the
baseline deliberately -- which is a reviewable diff, not a silent behaviour
change.

The baseline stores the *interpretation*, not the payload, so it stays small and
readable while the raw archive holds the bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from pmbtc.gamma.client import ResponseArchive
from pmbtc.logging_setup import get_logger
from pmbtc.settlement.parser import SettlementParseError, parse_settlement_spec
from pmbtc.settlement.spec import SettlementSpecification

log = get_logger("pmbtc.gamma.replay")

BASELINE_VERSION = 1


def interpretation_of(spec: SettlementSpecification) -> dict[str, Any]:
    """The fields whose meaning must never drift silently.

    Deliberately excludes ``detected_at_ms`` and anything else that legitimately
    varies between runs -- a baseline that fails on a timestamp teaches everyone
    to ignore it.
    """
    return {
        "spec_hash": spec.spec_hash,
        "provider": spec.provider.value,
        "trading_pair": spec.trading_pair,
        "interval_seconds": spec.interval_seconds,
        "window_open_ms": spec.window_open_ms,
        "window_close_ms": spec.window_close_ms,
        "tie_rule": spec.tie_rule.value,
        "reference_rule": spec.reference_rule,
        "price_decimals": spec.price_decimals,
        "confidence": round(spec.confidence, 4),
        "conflicts": sorted(spec.conflicts),
    }


@dataclass(frozen=True, slots=True)
class Difference:
    condition_id: str
    slug: str
    field_name: str
    baseline: Any
    current: Any

    def __str__(self) -> str:
        return (
            f"{self.slug or self.condition_id[:14]}.{self.field_name}: "
            f"{self.baseline!r} -> {self.current!r}"
        )


@dataclass
class ReplayReport:
    parsed: int = 0
    unparseable: int = 0
    new_markets: list[str] = field(default_factory=list)
    missing_markets: list[str] = field(default_factory=list)
    differences: list[Difference] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        """True when no previously-recorded interpretation changed.

        New markets do not break stability -- adding history is expected.
        Markets vanishing from the archive does not either. Only a *changed*
        reading of the same payload does.
        """
        return not self.differences

    def summary(self) -> str:
        if self.stable:
            return (
                f"stable: {self.parsed} payload(s) reinterpreted identically, "
                f"{len(self.new_markets)} new"
            )
        return f"UNSTABLE: {len(self.differences)} interpretation change(s)"


class ReplaySession:
    """Loads archived payloads and replays them through the current parser."""

    def __init__(self, archive_dir: Path) -> None:
        self.archive = ResponseArchive(archive_dir, enabled=False)
        self.archive.root = archive_dir

    # ------------------------------------------------------------------ #
    def iter_markets(self, day: str | None = None) -> list[dict[str, Any]]:
        """Every market object contained in the archive, de-duplicated.

        The archive holds whole responses, which repeat the same market across
        successive scans. Later captures win, so the replay sees each market's
        most recent recorded form.
        """
        by_condition: dict[str, dict[str, Any]] = {}
        for path in self.archive.files(day):
            try:
                record = orjson.loads(path.read_bytes())
            except (OSError, orjson.JSONDecodeError) as exc:
                log.warning("replay.unreadable", path=str(path), error=str(exc))
                continue
            for market in _extract_markets(record.get("payload")):
                condition_id = str(market.get("conditionId") or "")
                if condition_id:
                    by_condition[condition_id] = market
        return list(by_condition.values())

    def interpret(self, day: str | None = None) -> dict[str, dict[str, Any]]:
        """``condition_id -> interpretation`` for the whole archive."""
        out: dict[str, dict[str, Any]] = {}
        for market in self.iter_markets(day):
            try:
                spec = parse_settlement_spec(market)
            except SettlementParseError:
                continue
            entry = interpretation_of(spec)
            entry["slug"] = spec.slug
            out[spec.condition_id] = entry
        return out

    # ------------------------------------------------------------------ #
    def write_baseline(self, path: Path, day: str | None = None) -> int:
        interpretations = self.interpret(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"version": BASELINE_VERSION, "markets": interpretations},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return len(interpretations)

    def compare_to_baseline(self, path: Path, day: str | None = None) -> ReplayReport:
        report = ReplayReport()
        current = self.interpret(day)
        report.parsed = len(current)

        if not path.exists():
            report.new_markets = sorted(current)
            return report

        baseline = json.loads(path.read_text(encoding="utf-8")).get("markets", {})
        for condition_id, entry in current.items():
            previous = baseline.get(condition_id)
            if previous is None:
                report.new_markets.append(condition_id)
                continue
            for key, value in entry.items():
                if key == "slug":
                    continue
                if previous.get(key) != value:
                    report.differences.append(
                        Difference(condition_id, entry.get("slug", ""), key,
                                   previous.get(key), value)
                    )
        report.missing_markets = sorted(set(baseline) - set(current))
        return report


def _extract_markets(payload: Any) -> list[dict[str, Any]]:
    """Pull market objects out of any archived response shape."""
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return _extract_markets(payload["data"])
        if payload.get("conditionId"):
            return [payload]
        if payload.get("markets"):
            return _extract_markets(payload["markets"])
        return []
    if not isinstance(payload, list):
        return []

    markets: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("conditionId"):
            markets.append(item)
        elif item.get("markets"):
            # An event: re-attach it to each of its markets so the parser sees
            # the same shape the live pipeline gives it.
            event_summary = {k: v for k, v in item.items() if k != "markets"}
            for market in item["markets"]:
                if isinstance(market, dict):
                    merged = dict(market)
                    merged.setdefault("events", [event_summary])
                    markets.append(merged)
    return markets
