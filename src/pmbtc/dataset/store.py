"""Append-only dataset store.

Snapshots are immutable and horizon-addressed. Writing ``(condition_id,
horizon)`` twice is refused, not merged: a re-run that silently replaced
yesterday's T-60 snapshot with today's reconstruction would destroy exactly the
property that makes this dataset trustworthy.

Market records are the one thing that *does* update, and only in one direction:
an unlabelled record gains its label once the venue resolves. Any attempt to
change an already-assigned label is refused.

JSONL, one file per concern. It is append-only by nature, survives a partial
write, and can be inspected with a text editor at 3 a.m. Parquet is an export
format here, not the source of truth -- see :mod:`pmbtc.dataset.export`.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from pmbtc.config import Config
from pmbtc.dataset.leakage import LeakageGuard
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord
from pmbtc.exceptions import DataError
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.dataset.store")


class ImmutableRecordError(DataError):
    """An attempt to overwrite something the dataset guarantees is immutable."""


class DatasetStore:
    """Durable, append-only home for markets and their feature timelines."""

    def __init__(self, root: Path, guard: LeakageGuard | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.markets_path = root / "markets.jsonl"
        self.snapshots_path = root / "snapshots.jsonl"
        self.guard = guard or LeakageGuard()
        self._markets: dict[str, MarketRecord] = {}
        self._snapshots: dict[tuple[str, int], FeatureSnapshot] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        for path, handler in (
            (self.markets_path, self._index_market),
            (self.snapshots_path, self._index_snapshot),
        ):
            if not path.exists():
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    handler(orjson.loads(line))
                except Exception as exc:
                    # One corrupt line must not make the dataset unreadable.
                    log.warning("dataset.bad_line", path=path.name, line=number, error=str(exc))

    def _index_market(self, payload: dict) -> None:
        record = MarketRecord.model_validate(payload)
        self._markets[record.condition_id] = record

    def _index_snapshot(self, payload: dict) -> None:
        snapshot = FeatureSnapshot.model_validate(payload)
        self._snapshots[snapshot.key] = snapshot

    @staticmethod
    def _append(path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(orjson.dumps(payload, default=str).decode("utf-8") + "\n")
            fh.flush()

    # ------------------------------------------------------------------ #
    # Markets
    # ------------------------------------------------------------------ #
    def upsert_market(self, record: MarketRecord) -> bool:
        """Record or update a market. Returns True when something was written.

        The only legal update is unlabelled -> labelled. Re-labelling is refused
        because a changed label silently invalidates every model trained before
        the change.
        """
        existing = self._markets.get(record.condition_id)
        if existing is not None:
            if existing.is_labelled and record.is_labelled:
                if existing.official_outcome is not record.official_outcome:
                    raise ImmutableRecordError(
                        "Refusing to change an already-assigned label",
                        context={
                            "condition_id": record.condition_id,
                            "existing": existing.official_outcome,
                            "attempted": record.official_outcome,
                        },
                    )
                return False
            if existing.model_dump() == record.model_dump():
                return False

        self._markets[record.condition_id] = record
        self._append(self.markets_path, record.model_dump(mode="json"))
        return True

    def market(self, condition_id: str) -> MarketRecord | None:
        return self._markets.get(condition_id)

    def markets(self) -> list[MarketRecord]:
        return list(self._markets.values())

    def unlabelled_markets(self) -> list[MarketRecord]:
        """Markets past settlement that still need their official outcome."""
        return [m for m in self._markets.values() if not m.is_labelled]

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #
    def has_snapshot(self, condition_id: str, horizon_seconds: int) -> bool:
        return (condition_id, horizon_seconds) in self._snapshots

    def record_snapshot(self, snapshot: FeatureSnapshot) -> None:
        """Store a snapshot, or refuse.

        Two gates, both fatal: the leakage guard, and immutability. Neither
        degrades to a warning -- a dataset that sometimes enforces its
        invariants does not have them.
        """
        if snapshot.key in self._snapshots:
            raise ImmutableRecordError(
                "Snapshot already recorded; snapshots are never overwritten",
                context={
                    "condition_id": snapshot.condition_id,
                    "horizon_seconds": snapshot.horizon_seconds,
                },
            )
        self.guard.enforce(snapshot)
        self._snapshots[snapshot.key] = snapshot
        self._append(self.snapshots_path, snapshot.model_dump(mode="json"))

    def snapshot(self, condition_id: str, horizon_seconds: int) -> FeatureSnapshot | None:
        return self._snapshots.get((condition_id, horizon_seconds))

    def snapshots(self, condition_id: str | None = None) -> list[FeatureSnapshot]:
        if condition_id is None:
            return list(self._snapshots.values())
        return sorted(
            (s for s in self._snapshots.values() if s.condition_id == condition_id),
            key=lambda s: -s.horizon_seconds,
        )

    def timeline(self, condition_id: str) -> dict[int, FeatureSnapshot]:
        """``horizon -> snapshot`` for one market."""
        return {s.horizon_seconds: s for s in self.snapshots(condition_id)}

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, int]:
        return {
            "markets": len(self._markets),
            "labelled": sum(1 for m in self._markets.values() if m.is_labelled),
            "snapshots": len(self._snapshots),
        }

    def __len__(self) -> int:
        return len(self._markets)


def open_dataset_store(config: Config) -> DatasetStore:
    return DatasetStore(
        config.resolved_path(config.dataset.root_dir),
        guard=LeakageGuard(config.dataset.observation_tolerance_ms),
    )
