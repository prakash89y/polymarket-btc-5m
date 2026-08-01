"""Centralised clock service.

Why this is not optional
------------------------
The instrument is 300 seconds long. Every quantity the model consumes -- time
remaining, ``sqrt(1 - progress)`` volatility scaling, the entry and safety
windows -- is measured from the window boundary. A local clock that is one
second fast will submit orders one second later than intended relative to the
venue, and will mislabel the final seconds of every training row.

Measured on the development machine while building this module: the local clock
was **~1.3 seconds behind** Binance server time. That is 0.4% of the instrument,
silently.

Design
------
1. **Multiple independent references, median offset.** A single reference that
   is itself wrong is indistinguishable from local drift. The median of several
   sources is robust to one bad one.

2. **Round-trip correction, and uncertainty is tracked.** Each sample estimates
   the offset at the midpoint of the request, and carries ``rtt/2`` of
   irreducible uncertainty. Samples with a high RTT are discarded rather than
   averaged in -- a 1200 ms round trip cannot pin a clock to 100 ms.

3. **Fail closed.** If the service has never synced, its last sync is stale, or
   drift exceeds the configured limit, :meth:`ClockService.can_submit_order`
   returns false with a reason. An unknown clock is treated exactly like a bad
   clock.

4. **Never used to fabricate precision.** ``now_ms`` returns corrected time and
   ``uncertainty_ms`` says how much to trust it. Callers that need a hard
   boundary decision use :meth:`can_submit_order`, which already accounts for it.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import clock_drift_ms
from pmbtc.utils.timeutils import utc_now_ms

log = get_logger("pmbtc.clock")


class ClockStatus(StrEnum):
    HEALTHY = "healthy"
    NEVER_SYNCED = "never_synced"
    STALE = "stale"
    DRIFTED = "drifted"
    UNCERTAIN = "uncertain"

    @property
    def is_healthy(self) -> bool:
        return self is ClockStatus.HEALTHY


@dataclass(frozen=True, slots=True)
class ClockSample:
    """One offset measurement against one reference."""

    source: str
    #: reference_time - local_time, at the midpoint of the request.
    offset_ms: float
    rtt_ms: float
    taken_at_ms: int

    @property
    def uncertainty_ms(self) -> float:
        return self.rtt_ms / 2.0


@dataclass(frozen=True, slots=True)
class SubmissionDecision:
    """Whether an order may be submitted for a window, and why not."""

    allowed: bool
    reason: str
    seconds_to_settlement: float
    clock_status: ClockStatus

    def __bool__(self) -> bool:
        return self.allowed


class ClockService:
    """Corrected time, drift detection, and the pre-settlement safety gate."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg = config.clock
        self._samples: list[ClockSample] = []
        self._offset_ms: float = 0.0
        self._uncertainty_ms: float = 0.0
        self._last_sync_ms: int | None = None

    # ------------------------------------------------------------------ #
    # Time
    # ------------------------------------------------------------------ #
    def now_ms(self) -> int:
        """Best estimate of true UTC time, in epoch millis."""
        return int(utc_now_ms() + self._offset_ms)

    @property
    def offset_ms(self) -> float:
        """Correction applied to the local clock. Positive means we are behind."""
        return self._offset_ms

    @property
    def uncertainty_ms(self) -> float:
        return self._uncertainty_ms

    @property
    def last_sync_ms(self) -> int | None:
        return self._last_sync_ms

    @property
    def samples(self) -> tuple[ClockSample, ...]:
        return tuple(self._samples)

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #
    def add_sample(self, sample: ClockSample) -> None:
        """Record a measurement. Kept separate from I/O so it is testable."""
        if sample.rtt_ms > self.cfg.max_sample_rtt_ms:
            log.debug(
                "clock.sample_discarded",
                source=sample.source,
                rtt_ms=round(sample.rtt_ms, 1),
                limit_ms=self.cfg.max_sample_rtt_ms,
            )
            return
        # One sample per source: the freshest wins, so a slow source cannot
        # dominate the median with stale repeats.
        self._samples = [s for s in self._samples if s.source != sample.source]
        self._samples.append(sample)
        self._recompute()

    def _recompute(self) -> None:
        if not self._samples:
            return
        offsets = [s.offset_ms for s in self._samples]
        self._offset_ms = statistics.median(offsets)
        # Uncertainty is the best (smallest) half-RTT available, widened by how
        # much the sources disagree with each other.
        best = min(s.uncertainty_ms for s in self._samples)
        spread = (max(offsets) - min(offsets)) / 2 if len(offsets) > 1 else 0.0
        self._uncertainty_ms = best + spread
        self._last_sync_ms = max(s.taken_at_ms for s in self._samples)
        clock_drift_ms.set(self._offset_ms)

    async def sync(self, fetchers: dict[str, Any] | None = None) -> ClockStatus:
        """Take a fresh sample from each configured reference.

        ``fetchers`` maps a source name to an awaitable returning reference time
        in epoch millis. Injected rather than imported so this module has no
        network dependency and tests need no mocking framework.
        """
        if not fetchers:
            return self.status()
        for name, fetch in fetchers.items():
            if name not in self.cfg.sources:
                continue
            try:
                before = time.time() * 1000.0
                reference_ms = await fetch()
                after = time.time() * 1000.0
            except Exception as exc:
                log.warning("clock.sync_failed", source=name, error=str(exc))
                continue
            if not reference_ms:
                continue
            midpoint = (before + after) / 2.0
            self.add_sample(
                ClockSample(
                    source=name,
                    offset_ms=float(reference_ms) - midpoint,
                    rtt_ms=after - before,
                    taken_at_ms=int(after),
                )
            )
        status = self.status()
        log.info(
            "clock.synced",
            status=status.value,
            offset_ms=round(self._offset_ms, 1),
            uncertainty_ms=round(self._uncertainty_ms, 1),
            sources=[s.source for s in self._samples],
        )
        return status

    async def run_forever(self, fetchers: dict[str, Any]) -> None:  # pragma: no cover - loop
        """Background resync task."""
        while True:
            await self.sync(fetchers)
            await asyncio.sleep(self.cfg.sync_interval_seconds)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def status(self) -> ClockStatus:
        if self._last_sync_ms is None or len(self._samples) < self.cfg.min_samples:
            return ClockStatus.NEVER_SYNCED
        age_s = (utc_now_ms() - self._last_sync_ms) / 1000.0
        if age_s > self.cfg.stale_sync_seconds:
            return ClockStatus.STALE
        if abs(self._offset_ms) > self.cfg.max_drift_ms:
            return ClockStatus.DRIFTED
        if self._uncertainty_ms > self.cfg.max_uncertainty_ms:
            return ClockStatus.UNCERTAIN
        return ClockStatus.HEALTHY

    # ------------------------------------------------------------------ #
    # Window arithmetic
    # ------------------------------------------------------------------ #
    def ms_to_settlement(self, settlement_ms: int) -> int:
        return settlement_ms - self.now_ms()

    def seconds_to_settlement(self, settlement_ms: int) -> float:
        return self.ms_to_settlement(settlement_ms) / 1000.0

    def seconds_to_lock(self, settlement_ms: int) -> float:
        """Seconds until the safety window closes -- the real deadline.

        The venue accepts orders until settlement, but we stop earlier: an order
        that fills in the last seconds cannot be managed and races the
        settlement feed.
        """
        return self.seconds_to_settlement(settlement_ms) - self.cfg.order_safety_window_seconds

    def can_submit_order(self, settlement_ms: int) -> SubmissionDecision:
        """The single gate every order must pass.

        Uses the *pessimistic* edge of the clock uncertainty: if we might already
        be inside the safety window, we treat ourselves as inside it.
        """
        status = self.status()
        remaining = self.seconds_to_settlement(settlement_ms)
        if not status.is_healthy:
            return SubmissionDecision(
                allowed=False,
                reason=f"clock is {status.value}"
                + (
                    f" (offset {self._offset_ms:.0f}ms, limit {self.cfg.max_drift_ms}ms)"
                    if status is ClockStatus.DRIFTED
                    else ""
                ),
                seconds_to_settlement=remaining,
                clock_status=status,
            )
        if remaining <= 0:
            return SubmissionDecision(
                False, "window already settled", remaining, status
            )
        worst_case = remaining - self._uncertainty_ms / 1000.0
        if worst_case <= self.cfg.order_safety_window_seconds:
            return SubmissionDecision(
                allowed=False,
                reason=(
                    f"inside the {self.cfg.order_safety_window_seconds}s safety window "
                    f"({worst_case:.1f}s remaining, worst case)"
                ),
                seconds_to_settlement=remaining,
                clock_status=status,
            )
        return SubmissionDecision(True, "ok", remaining, status)

    def describe(self) -> dict[str, Any]:
        """Operator-facing summary, used by ``pmbtc doctor`` and the dashboard."""
        return {
            "status": self.status().value,
            "offset_ms": round(self._offset_ms, 1),
            "uncertainty_ms": round(self._uncertainty_ms, 1),
            "last_sync_ms": self._last_sync_ms,
            "sources": [
                {"source": s.source, "offset_ms": round(s.offset_ms, 1),
                 "rtt_ms": round(s.rtt_ms, 1)}
                for s in self._samples
            ],
            "safety_window_s": self.cfg.order_safety_window_seconds,
        }
