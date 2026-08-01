"""Market lifecycle tracking.

A 5-minute market is only interesting for a few minutes, but *when* it was in
each state matters permanently: the training set needs to know when a market
became tradeable, when it entered the no-entry zone, and when it settled, so
that a feature computed at decision time can never be contaminated by a later
state.

States are derived from the clock and the payload rather than assumed, and only
*transitions* are persisted -- at one scan every 30 seconds, storing the state
each time would write thousands of identical rows a day and bury the two lines
that matter.

Illegal transitions (settled -> open) are recorded and refused rather than
silently applied: they mean the clock moved backwards or a payload was stale,
both of which are worth knowing about immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import orjson

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import lifecycle_transitions
from pmbtc.utils.timeutils import isoformat

log = get_logger("pmbtc.gamma.lifecycle")


class MarketState(StrEnum):
    """Where a market is in its life."""

    DISCOVERED = "discovered"      # seen, not yet open for our purposes
    OPEN = "open"                  # tradeable window in progress
    NEAR_SETTLEMENT = "near_settlement"  # inside the safety window; no new entries
    SETTLED = "settled"            # past its settlement instant
    ARCHIVED = "archived"          # resolved and closed by the venue
    REJECTED = "rejected"          # failed health or verification; terminal


#: Forward-only. Skipping ahead is legal (a slow scan can miss a state);
#: going backwards is not.
_ORDER: dict[MarketState, int] = {
    MarketState.DISCOVERED: 0,
    MarketState.OPEN: 1,
    MarketState.NEAR_SETTLEMENT: 2,
    MarketState.SETTLED: 3,
    MarketState.ARCHIVED: 4,
    MarketState.REJECTED: 5,
}


@dataclass(frozen=True, slots=True)
class Transition:
    condition_id: str
    slug: str
    previous: MarketState | None
    current: MarketState
    at_ms: int
    reason: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "previous": self.previous.value if self.previous else None,
            "current": self.current.value,
            "at_ms": self.at_ms,
            "at": isoformat(self.at_ms),
            "reason": self.reason,
        }


def classify(
    *,
    settlement_ms: int,
    window_open_ms: int,
    now_ms: int,
    safety_window_s: int,
    closed: bool = False,
    accepting_orders: bool = True,
) -> MarketState:
    """Derive the current state from time and venue flags."""
    if closed:
        return MarketState.ARCHIVED
    if now_ms >= settlement_ms:
        return MarketState.SETTLED
    if now_ms >= settlement_ms - safety_window_s * 1000:
        return MarketState.NEAR_SETTLEMENT
    if now_ms >= window_open_ms and accepting_orders:
        return MarketState.OPEN
    return MarketState.DISCOVERED


class LifecycleTracker:
    """In-memory state per market, with an append-only transition log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, MarketState] = {}
        self._history: dict[str, list[Transition]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
                transition = Transition(
                    condition_id=record["condition_id"],
                    slug=record.get("slug", ""),
                    previous=MarketState(record["previous"]) if record.get("previous") else None,
                    current=MarketState(record["current"]),
                    at_ms=int(record["at_ms"]),
                    reason=record.get("reason", ""),
                )
            except Exception as exc:
                log.warning("lifecycle.bad_line", error=str(exc))
                continue
            self._state[transition.condition_id] = transition.current
            self._history.setdefault(transition.condition_id, []).append(transition)

    # ------------------------------------------------------------------ #
    def observe(
        self, condition_id: str, slug: str, state: MarketState, at_ms: int, reason: str = ""
    ) -> Transition | None:
        """Record a state observation. Returns a transition only on change."""
        previous = self._state.get(condition_id)
        if previous is state:
            return None
        if previous is not None and _ORDER[state] < _ORDER[previous]:
            # Backwards means something is wrong upstream, not that the market
            # went back in time.
            log.error(
                "lifecycle.illegal_transition",
                condition_id=condition_id,
                slug=slug,
                previous=previous.value,
                attempted=state.value,
            )
            return None

        transition = Transition(condition_id, slug, previous, state, at_ms, reason)
        self._state[condition_id] = state
        self._history.setdefault(condition_id, []).append(transition)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(orjson.dumps(transition.as_record()).decode("utf-8") + "\n")
            fh.flush()
        lifecycle_transitions.inc(to=state.value)
        log.info(
            "lifecycle.transition",
            condition_id=condition_id,
            slug=slug,
            previous=previous.value if previous else None,
            current=state.value,
            reason=reason,
        )
        return transition

    def state_of(self, condition_id: str) -> MarketState | None:
        return self._state.get(condition_id)

    def history(self, condition_id: str) -> tuple[Transition, ...]:
        return tuple(self._history.get(condition_id, ()))

    def timestamps(self, condition_id: str) -> dict[str, int]:
        """``state -> first entry timestamp``, for feature engineering."""
        return {t.current.value: t.at_ms for t in self._history.get(condition_id, ())}

    def in_state(self, state: MarketState) -> tuple[str, ...]:
        return tuple(cid for cid, current in self._state.items() if current is state)

    def __len__(self) -> int:
        return len(self._state)


def open_lifecycle_tracker(config: Config) -> LifecycleTracker:
    return LifecycleTracker(
        config.resolved_path(config.app.data_dir) / "market_lifecycle.jsonl"
    )
