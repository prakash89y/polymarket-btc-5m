"""Durable record of how every market settled.

Two jobs:

1. **Historical fidelity.** A backtest must replay each market against the
   settlement source that was in force *at the time*, not today's. Polymarket
   has already moved one BTC family from Binance to Chainlink; assuming today's
   provider held historically would silently corrupt every label before the
   switch. So the spec is stored per market and read back by condition id.

2. **Change detection.** Each series carries a settlement fingerprint
   (:attr:`SettlementSpecification.spec_hash`). When a new market in a known
   series hashes differently, the family has changed how it resolves and the
   verifier refuses to trade until the change is reviewed.

Storage is append-only JSONL. It is small (one line per market), trivially
greppable, replicates as a file, and cannot be corrupted by a partial write of a
later record. An index is built in memory at load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import orjson

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.settlement.spec import SettlementSpecification
from pmbtc.utils.timeutils import utc_now_ms

log = get_logger("pmbtc.settlement.store")


@dataclass(frozen=True, slots=True)
class SettlementChange:
    """A detected change in how a series resolves."""

    series_slug: str
    previous_hash: str
    current_hash: str
    previous_provider: str
    current_provider: str
    first_seen_ms: int

    @property
    def is_provider_change(self) -> bool:
        return self.previous_provider != self.current_provider

    def __str__(self) -> str:
        return (
            f"series {self.series_slug}: {self.previous_provider}({self.previous_hash}) "
            f"-> {self.current_provider}({self.current_hash})"
        )


class SettlementSpecStore:
    """Append-only store of settlement specifications, indexed in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._by_condition: dict[str, SettlementSpecification] = {}
        #: series slug -> the fingerprint most recently recorded for it.
        self._series_hash: dict[str, str] = {}
        self._series_history: dict[str, list[tuple[int, str, str]]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_no, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                spec = SettlementSpecification.model_validate(record["spec"])
            except Exception as exc:
                # One bad line must not make the whole history unreadable.
                log.warning("settlement.store.bad_line", line=line_no, error=str(exc))
                continue
            self._index(spec)

    def _index(self, spec: SettlementSpecification) -> None:
        self._by_condition[spec.condition_id] = spec
        key = spec.series_slug or spec.slug
        if key:
            previous = self._series_hash.get(key)
            if previous != spec.spec_hash:
                self._series_history.setdefault(key, []).append(
                    (spec.detected_at_ms, spec.spec_hash, spec.provider.value)
                )
            self._series_hash[key] = spec.spec_hash

    # ------------------------------------------------------------------ #
    def record(self, spec: SettlementSpecification) -> SettlementChange | None:
        """Persist a spec. Returns a :class:`SettlementChange` if it differs.

        Idempotent per market: re-recording an unchanged market rewrites nothing.
        """
        existing = self._by_condition.get(spec.condition_id)
        if existing is not None and existing.spec_hash == spec.spec_hash:
            return None

        key = spec.series_slug or spec.slug
        previous_hash = self._series_hash.get(key) if key else None
        change: SettlementChange | None = None
        if previous_hash and previous_hash != spec.spec_hash:
            previous_spec = self._latest_for_series(key)
            change = SettlementChange(
                series_slug=key,
                previous_hash=previous_hash,
                current_hash=spec.spec_hash,
                previous_provider=previous_spec.provider.value if previous_spec else "unknown",
                current_provider=spec.provider.value,
                first_seen_ms=spec.detected_at_ms or utc_now_ms(),
            )
            log.warning(
                "settlement.changed",
                series=key,
                previous=change.previous_provider,
                current=change.current_provider,
                previous_hash=previous_hash,
                current_hash=spec.spec_hash,
            )

        with self.path.open("a", encoding="utf-8") as fh:
            payload = {
                "recorded_at_ms": utc_now_ms(),
                "condition_id": spec.condition_id,
                "series_slug": spec.series_slug,
                "spec_hash": spec.spec_hash,
                "spec": spec.model_dump(mode="json"),
            }
            fh.write(orjson.dumps(payload, default=str).decode("utf-8") + "\n")
            fh.flush()

        self._index(spec)
        return change

    def _latest_for_series(self, series_slug: str) -> SettlementSpecification | None:
        candidates = [
            s for s in self._by_condition.values() if (s.series_slug or s.slug) == series_slug
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.detected_at_ms)

    # ------------------------------------------------------------------ #
    def get(self, condition_id: str) -> SettlementSpecification | None:
        """The spec recorded for a market -- what a backtest must replay against."""
        return self._by_condition.get(condition_id)

    def known_hash(self, series_slug: str) -> str | None:
        """Fingerprint last recorded for a series, for the verifier's gate."""
        return self._series_hash.get(series_slug)

    def history(self, series_slug: str) -> tuple[tuple[int, str, str], ...]:
        """``(detected_at_ms, spec_hash, provider)`` for each distinct change."""
        return tuple(self._series_history.get(series_slug, ()))

    def providers_seen(self, series_slug: str) -> tuple[str, ...]:
        seen: list[str] = []
        for _, _, provider in self._series_history.get(series_slug, ()):
            if provider not in seen:
                seen.append(provider)
        return tuple(seen)

    def __len__(self) -> int:
        return len(self._by_condition)


def open_spec_store(config: Config) -> SettlementSpecStore:
    """Open the store at its configured location."""
    path = config.resolved_path(config.app.data_dir) / "settlement_specs.jsonl"
    return SettlementSpecStore(path)
