"""Clock service: drift, uncertainty, and the pre-settlement gate.

The gate must fail closed. Every "unknown clock" path is tested as carefully as
the drifted one, because an unsynced clock and a wrong clock are equally unsafe
and only one of them is obvious.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pmbtc.clock import ClockSample, ClockService, ClockStatus
from pmbtc.config import load_config
from pmbtc.utils.timeutils import utc_now_ms

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def clock():  # type: ignore[no-untyped-def]
    return ClockService(load_config(SHIPPED_CONFIG))


def sample(source: str = "binance", offset_ms: float = 0.0, rtt_ms: float = 100.0) -> ClockSample:
    return ClockSample(source, offset_ms, rtt_ms, utc_now_ms())


class TestStatus:
    def test_starts_unsynced(self, clock: ClockService) -> None:
        assert clock.status() is ClockStatus.NEVER_SYNCED

    def test_healthy_after_a_good_sample(self, clock: ClockService) -> None:
        clock.add_sample(sample(offset_ms=50, rtt_ms=100))
        assert clock.status() is ClockStatus.HEALTHY

    def test_drift_beyond_limit(self, clock: ClockService) -> None:
        clock.add_sample(sample(offset_ms=5_000, rtt_ms=100))
        assert clock.status() is ClockStatus.DRIFTED

    def test_negative_drift_also_counts(self, clock: ClockService) -> None:
        clock.add_sample(sample(offset_ms=-5_000, rtt_ms=100))
        assert clock.status() is ClockStatus.DRIFTED

    def test_slow_samples_are_discarded_not_averaged(self, clock: ClockService) -> None:
        # A 3-second round trip cannot pin a clock; using it would manufacture
        # false precision.
        clock.add_sample(sample(rtt_ms=3_000))
        assert clock.samples == ()
        assert clock.status() is ClockStatus.NEVER_SYNCED

    def test_stale_sync(self, clock: ClockService) -> None:
        old = ClockSample("binance", 10.0, 100.0, utc_now_ms() - 10 * 60_000)
        clock.add_sample(old)
        assert clock.status() is ClockStatus.STALE

    def test_disagreeing_sources_widen_uncertainty(self, clock: ClockService) -> None:
        clock.add_sample(sample("binance", offset_ms=0, rtt_ms=100))
        clock.add_sample(sample("clob", offset_ms=1_600, rtt_ms=100))
        # Sources that disagree by 1.6s cannot jointly pin the clock to better
        # than ±800ms, which exceeds the 750ms limit and blocks trading.
        assert clock.uncertainty_ms == pytest.approx(850, abs=1)
        assert clock.status() is ClockStatus.UNCERTAIN


class TestOffset:
    def test_median_of_sources(self, clock: ClockService) -> None:
        for source, offset in (("binance", 100.0), ("clob", 120.0), ("other", 900.0)):
            clock.add_sample(sample(source, offset_ms=offset, rtt_ms=50))
        assert clock.offset_ms == 120.0  # robust to the outlier

    def test_one_sample_per_source_freshest_wins(self, clock: ClockService) -> None:
        clock.add_sample(sample("binance", offset_ms=100))
        clock.add_sample(sample("binance", offset_ms=200))
        assert len(clock.samples) == 1
        assert clock.offset_ms == 200

    def test_now_applies_the_correction(self, clock: ClockService) -> None:
        clock.add_sample(sample(offset_ms=1_000, rtt_ms=100))
        assert clock.now_ms() - utc_now_ms() == pytest.approx(1_000, abs=50)


class TestSubmissionGate:
    def _settlement(self, clock: ClockService, seconds_ahead: float) -> int:
        return clock.now_ms() + int(seconds_ahead * 1000)

    def test_allows_a_comfortable_window(self, clock: ClockService) -> None:
        clock.add_sample(sample(rtt_ms=50))
        decision = clock.can_submit_order(self._settlement(clock, 120))
        assert decision.allowed is True
        assert bool(decision) is True

    def test_blocks_inside_the_safety_window(self, clock: ClockService) -> None:
        clock.add_sample(sample(rtt_ms=50))
        decision = clock.can_submit_order(self._settlement(clock, 10))
        assert decision.allowed is False
        assert "safety window" in decision.reason

    def test_blocks_after_settlement(self, clock: ClockService) -> None:
        clock.add_sample(sample(rtt_ms=50))
        decision = clock.can_submit_order(self._settlement(clock, -5))
        assert decision.allowed is False
        assert "settled" in decision.reason

    def test_uncertainty_is_treated_pessimistically(self, clock: ClockService) -> None:
        # 22s remaining, 20s safety window, but ±3s of clock uncertainty: we
        # might already be inside it, so we behave as if we are.
        clock.add_sample(sample(rtt_ms=1_800))
        decision = clock.can_submit_order(self._settlement(clock, 22))
        assert decision.allowed is False

    def test_fails_closed_when_never_synced(self, clock: ClockService) -> None:
        decision = clock.can_submit_order(utc_now_ms() + 200_000)
        assert decision.allowed is False
        assert decision.clock_status is ClockStatus.NEVER_SYNCED

    def test_fails_closed_when_drifted(self, clock: ClockService) -> None:
        clock.add_sample(sample(offset_ms=9_000, rtt_ms=50))
        decision = clock.can_submit_order(clock.now_ms() + 200_000)
        assert decision.allowed is False
        assert "drifted" in decision.reason

    def test_seconds_to_lock_is_earlier_than_settlement(self, clock: ClockService) -> None:
        clock.add_sample(sample(rtt_ms=50))
        settlement = self._settlement(clock, 100)
        assert clock.seconds_to_lock(settlement) == pytest.approx(
            clock.seconds_to_settlement(settlement) - 20, abs=0.5
        )


class TestSyncPlumbing:
    async def test_sync_records_samples(self, clock: ClockService) -> None:
        async def fake_binance() -> int:
            return utc_now_ms() + 250

        status = await clock.sync({"binance": fake_binance})
        assert status is ClockStatus.HEALTHY
        assert clock.offset_ms == pytest.approx(250, abs=100)

    async def test_a_failing_source_does_not_break_sync(self, clock: ClockService) -> None:
        async def broken() -> int:
            raise RuntimeError("network down")

        async def good() -> int:
            return utc_now_ms()

        await clock.sync({"clob": broken, "binance": good})
        assert [s.source for s in clock.samples] == ["binance"]

    async def test_unconfigured_sources_are_ignored(self, clock: ClockService) -> None:
        async def rogue() -> int:
            return utc_now_ms() + 60_000

        await clock.sync({"not_in_config": rogue})
        assert clock.samples == ()

    def test_describe_is_operator_readable(self, clock: ClockService) -> None:
        clock.add_sample(sample(rtt_ms=50))
        described = clock.describe()
        assert described["status"] == "healthy"
        assert described["safety_window_s"] == 20
        assert described["sources"][0]["source"] == "binance"
