"""Leakage prevention.

This is the module that decides whether the whole project is worth anything. A
leaked feature produces a backtest that looks like genius and an account that
loses money, and the failure is invisible in every metric except live PnL.

So leakage is prevented in three independent places:

1. **At write time.** :class:`LeakageGuard` inspects every snapshot before it is
   stored. Any observation whose ``event_time`` postdates the snapshot instant
   is a hard error -- the snapshot is refused, not truncated.

2. **At export time.** :func:`scan_snapshots` re-checks the whole dataset, so a
   snapshot written by an older or buggier version of the collector cannot
   quietly survive into a training set.

3. **In the tests.** The scanner runs over deliberately poisoned fixtures, so
   the detector itself is verified rather than trusted.

The invariant, stated once:

    For a snapshot at instant ``S``, every observation must satisfy
    ``event_time <= S`` **and** ``observation_time <= S + tolerance``.

``event_time <= S`` is the real rule: information created after ``S`` cannot be
known at ``S``. The ``observation_time`` bound catches a subtler case -- a value
whose event predates ``S`` but which we only *received* afterwards. Acting on it
would have been impossible in real time, so it is leakage too, allowed only a
small tolerance for scheduling jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord, Observation
from pmbtc.exceptions import LookaheadError
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.dataset.leakage")


class LeakageKind(StrEnum):
    FUTURE_EVENT = "future_event"
    FUTURE_OBSERVATION = "future_observation"
    POST_SETTLEMENT = "post_settlement"
    SNAPSHOT_AFTER_SETTLEMENT = "snapshot_after_settlement"
    HORIZON_MISMATCH = "horizon_mismatch"
    LABEL_IN_FEATURES = "label_in_features"


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    kind: LeakageKind
    condition_id: str
    horizon_seconds: int
    feature: str
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.kind.value} @ T-{self.horizon_seconds}s "
            f"[{self.condition_id[:14]}] {self.feature}: {self.detail}"
        )


#: Feature names that could only be known after the fact. Any of these appearing
#: as a feature is leakage by definition, whatever its timestamps claim.
FORBIDDEN_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "label",
        "outcome",
        "official_outcome",
        "settlement_probability",
        "settlement_price",
        "yes_final_price",
        "no_final_price",
        "resolved_at_ms",
        "winner",
        "payout",
    }
)


class LeakageGuard:
    """Write-time enforcement of the leakage invariant."""

    def __init__(self, observation_tolerance_ms: int = 500) -> None:
        #: Scheduling jitter allowance for ``observation_time``. Small on
        #: purpose: it exists for millisecond-scale timer slop, not to excuse a
        #: feature that genuinely arrived late.
        self.observation_tolerance_ms = observation_tolerance_ms

    # ------------------------------------------------------------------ #
    def inspect(self, snapshot: FeatureSnapshot) -> list[LeakageFinding]:
        """Every leakage problem in one snapshot."""
        findings: list[LeakageFinding] = []
        instant = snapshot.snapshot_time_ms

        if instant > snapshot.settlement_time_ms:
            findings.append(
                LeakageFinding(
                    LeakageKind.SNAPSHOT_AFTER_SETTLEMENT,
                    snapshot.condition_id,
                    snapshot.horizon_seconds,
                    "-",
                    f"snapshot instant {instant} is after settlement "
                    f"{snapshot.settlement_time_ms}",
                )
            )

        expected = snapshot.settlement_time_ms - snapshot.horizon_seconds * 1000
        if instant != expected:
            findings.append(
                LeakageFinding(
                    LeakageKind.HORIZON_MISMATCH,
                    snapshot.condition_id,
                    snapshot.horizon_seconds,
                    "-",
                    f"instant {instant} does not match horizon (expected {expected})",
                )
            )

        for obs in snapshot.observations:
            findings.extend(self._inspect_observation(snapshot, obs, instant))
        return findings

    def _inspect_observation(
        self, snapshot: FeatureSnapshot, obs: Observation, instant: int
    ) -> list[LeakageFinding]:
        findings: list[LeakageFinding] = []
        if obs.name.lower() in FORBIDDEN_FEATURE_NAMES:
            findings.append(
                LeakageFinding(
                    LeakageKind.LABEL_IN_FEATURES,
                    snapshot.condition_id,
                    snapshot.horizon_seconds,
                    obs.name,
                    "outcome-derived value present as a feature",
                )
            )
        if obs.value is None:
            # A missing value carries no information and cannot leak.
            return findings

        if obs.event_time_ms > instant:
            findings.append(
                LeakageFinding(
                    LeakageKind.FUTURE_EVENT,
                    snapshot.condition_id,
                    snapshot.horizon_seconds,
                    obs.name,
                    f"event at {obs.event_time_ms} is {obs.event_time_ms - instant}ms "
                    "after the snapshot instant",
                )
            )
        if obs.observation_time_ms > instant + self.observation_tolerance_ms:
            findings.append(
                LeakageFinding(
                    LeakageKind.FUTURE_OBSERVATION,
                    snapshot.condition_id,
                    snapshot.horizon_seconds,
                    obs.name,
                    f"received at {obs.observation_time_ms}, "
                    f"{obs.observation_time_ms - instant}ms after the snapshot instant",
                )
            )
        if obs.event_time_ms > snapshot.settlement_time_ms:
            findings.append(
                LeakageFinding(
                    LeakageKind.POST_SETTLEMENT,
                    snapshot.condition_id,
                    snapshot.horizon_seconds,
                    obs.name,
                    "event postdates settlement",
                )
            )
        return findings

    def enforce(self, snapshot: FeatureSnapshot) -> None:
        """Raise unless the snapshot is clean.

        Fatal by design. There is no partial-credit handling: a snapshot that
        contains future information is discarded whole, because deciding which
        half to keep is exactly the judgement call that lets leakage through.
        """
        findings = self.inspect(snapshot)
        if not findings:
            return
        log.error(
            "leakage.blocked",
            condition_id=snapshot.condition_id,
            horizon=snapshot.horizon_seconds,
            findings=[str(f) for f in findings],
        )
        raise LookaheadError(
            "Snapshot rejected: it contains information from after its own instant",
            context={
                "condition_id": snapshot.condition_id,
                "horizon_seconds": snapshot.horizon_seconds,
                "findings": "; ".join(str(f) for f in findings),
            },
        )


# --------------------------------------------------------------------------- #
# Dataset-wide scanning
# --------------------------------------------------------------------------- #
def scan_snapshots(
    snapshots: list[FeatureSnapshot], guard: LeakageGuard | None = None
) -> list[LeakageFinding]:
    """Re-check a whole dataset. Used at export and in the test suite."""
    guard = guard or LeakageGuard()
    findings: list[LeakageFinding] = []
    for snapshot in snapshots:
        findings.extend(guard.inspect(snapshot))
    return findings


def scan_monotonicity(snapshots: list[FeatureSnapshot]) -> list[LeakageFinding]:
    """Check that earlier snapshots do not contain later information.

    The requirement in its strongest form: a T-300 snapshot must not contain
    anything the T-60 snapshot learned. Because every observation carries its own
    event time, this reduces to checking that each snapshot's newest event is no
    newer than its own instant -- and, across horizons, that the newest event
    time is non-decreasing as the horizon shrinks.
    """
    findings: list[LeakageFinding] = []
    by_market: dict[str, list[FeatureSnapshot]] = {}
    for snapshot in snapshots:
        by_market.setdefault(snapshot.condition_id, []).append(snapshot)

    for condition_id, group in by_market.items():
        # Descending horizon == chronological order.
        ordered = sorted(group, key=lambda s: -s.horizon_seconds)
        previous_newest = None
        previous_horizon = None
        for snapshot in ordered:
            events = [o.event_time_ms for o in snapshot.observations if o.value is not None]
            if not events:
                continue
            newest = max(events)
            if previous_newest is not None and newest < previous_newest:
                findings.append(
                    LeakageFinding(
                        LeakageKind.FUTURE_EVENT,
                        condition_id,
                        snapshot.horizon_seconds,
                        "-",
                        f"newest event {newest} is older than the T-{previous_horizon}s "
                        f"snapshot's {previous_newest}; snapshots are out of order",
                    )
                )
            previous_newest = newest
            previous_horizon = snapshot.horizon_seconds
    return findings


def assert_no_label_leakage(
    snapshots: list[FeatureSnapshot], markets: dict[str, MarketRecord]
) -> list[LeakageFinding]:
    """Confirm no snapshot postdates the resolution of its own market."""
    findings: list[LeakageFinding] = []
    for snapshot in snapshots:
        market = markets.get(snapshot.condition_id)
        if market is None or market.resolved_at_ms is None:
            continue
        for obs in snapshot.observations:
            if obs.value is not None and obs.observation_time_ms >= market.resolved_at_ms:
                findings.append(
                    LeakageFinding(
                        LeakageKind.POST_SETTLEMENT,
                        snapshot.condition_id,
                        snapshot.horizon_seconds,
                        obs.name,
                        "observed at or after the market resolved",
                    )
                )
    return findings
