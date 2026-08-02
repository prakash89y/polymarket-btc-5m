"""Regression tests for the production-pipeline RCA (Module 8.5 operations).

Each class pins one proven root cause. The numbers quoted in the docstrings are
measurements from the 13.5-hour production run that exposed them, not
illustrations — if a test here fails, that specific production failure has
returned.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from pmbtc.clock import ClockSample, ClockService, ClockStatus
from pmbtc.config import Config
from pmbtc.live.feed import FeedState, WebSocketFeed


@pytest.fixture
def config() -> Config:
    return Config()


# --------------------------------------------------------------------------- #
# Root causes 1 & 2: the resync loop is supervised and keeps running
# --------------------------------------------------------------------------- #
class TestClockResync:
    """Production symptom: `clock.synced` appeared exactly once in 13.5 hours."""

    @staticmethod
    def _fast_interval(monkeypatch) -> None:
        """Collapse the resync interval without touching the config floor.

        ``clock.sync_interval_seconds`` is validated at >= 5s, and the property
        under test is that the loop *repeats*, not how long it waits.
        """
        import pmbtc.clock as clockmod

        class _Fast:
            CancelledError = asyncio.CancelledError

            @staticmethod
            async def sleep(_seconds: float) -> None:
                await asyncio.sleep(0.01)

        monkeypatch.setattr(clockmod, "asyncio", _Fast)

    @pytest.mark.asyncio
    async def test_run_forever_syncs_repeatedly(self, monkeypatch) -> None:
        self._fast_interval(monkeypatch)
        clock = ClockService(Config())
        calls = 0

        async def fetch() -> int:
            nonlocal calls
            calls += 1
            from pmbtc.utils.timeutils import utc_now_ms

            return utc_now_ms() + 5

        task = asyncio.create_task(clock.run_forever({"clob": fetch}))
        await asyncio.sleep(0.2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert calls >= 3, f"resync loop ran {calls} time(s); it must keep syncing"

    @pytest.mark.asyncio
    async def test_a_failing_sync_does_not_kill_the_loop(self, monkeypatch) -> None:
        """A loop that dies on one bad network minute recreates the bug."""
        self._fast_interval(monkeypatch)
        clock = ClockService(Config())
        calls = 0

        async def flaky() -> int:
            nonlocal calls
            calls += 1
            raise ConnectionError("network blip")

        task = asyncio.create_task(clock.run_forever({"clob": flaky}))
        await asyncio.sleep(0.2)
        alive = not task.done()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert alive, "resync loop terminated after a failed sync"
        assert calls >= 3

    @pytest.mark.asyncio
    async def test_discovery_stack_supervises_the_loop(self, monkeypatch) -> None:
        """Root cause 1 exactly: the loop existed but nothing started it."""
        import pmbtc.gamma.runtime as runtime

        started: list[str] = []

        async def fake_run_forever(self: ClockService, fetchers: dict[str, Any]) -> None:
            started.append("running")
            await asyncio.sleep(3600)

        monkeypatch.setattr(ClockService, "run_forever", fake_run_forever)
        monkeypatch.setattr(
            runtime.DiscoveryStack, "sync_clock", lambda self: asyncio.sleep(0)
        )

        class _Client:
            async def start(self) -> None: ...
            async def aclose(self) -> None: ...

        monkeypatch.setattr(runtime, "GammaClient", lambda config: _Client())
        async with runtime.discovery_stack(Config()):
            await asyncio.sleep(0.05)
        assert started, "discovery_stack did not supervise the clock resync loop"


class TestClockSampleQuality:
    """Root cause 3: a drifted/uncertain sample became permanent.

    Production: one startup sample with 927 ms uncertainty — which the service
    itself scored DRIFTED and UNCERTAIN — was applied for 13.5 hours, while the
    measured true offset was 297 ms against the 1,636 ms being used.
    """

    def _sample(self, source: str, offset: float, rtt: float, at: int) -> ClockSample:
        return ClockSample(source=source, offset_ms=offset, rtt_ms=rtt, taken_at_ms=at)

    def test_a_worse_sample_does_not_replace_a_good_one(self, config: Config) -> None:
        from pmbtc.utils.timeutils import utc_now_ms

        clock = ClockService(config)
        now = utc_now_ms()
        clock.add_sample(self._sample("clob", 100.0, 80.0, now))
        assert clock.offset_ms == pytest.approx(100.0)
        clock.add_sample(self._sample("clob", 1636.0, 1800.0, now))
        assert clock.offset_ms == pytest.approx(100.0), "a worse sample was adopted"

    def test_a_better_sample_does_replace(self, config: Config) -> None:
        from pmbtc.utils.timeutils import utc_now_ms

        clock = ClockService(config)
        now = utc_now_ms()
        clock.add_sample(self._sample("clob", 1636.0, 1400.0, now))
        clock.add_sample(self._sample("clob", 297.0, 100.0, now))
        assert clock.offset_ms == pytest.approx(297.0)

    def test_a_stale_incumbent_never_blocks_a_fresh_sample(self, config: Config) -> None:
        """Precision does not outrank freshness once a reading has aged out —
        that preference is exactly what froze the offset."""
        from pmbtc.utils.timeutils import utc_now_ms

        clock = ClockService(config)
        old = utc_now_ms() - (config.clock.stale_sync_seconds + 60) * 1000
        clock.add_sample(self._sample("clob", 1636.0, 10.0, old))
        clock.add_sample(self._sample("clob", 297.0, 900.0, utc_now_ms()))
        assert clock.offset_ms == pytest.approx(297.0)

    def test_an_unusable_incumbent_is_always_replaceable(self, config: Config) -> None:
        from pmbtc.utils.timeutils import utc_now_ms

        clock = ClockService(config)
        now = utc_now_ms()
        # 1854 ms RTT -> 927 ms uncertainty, over max_uncertainty_ms (750).
        clock.add_sample(self._sample("clob", 1636.0, 1854.0, now))
        assert clock.status() is not ClockStatus.HEALTHY
        clock.add_sample(self._sample("clob", 297.0, 1900.0, now))
        assert clock.offset_ms == pytest.approx(297.0)

    def test_high_rtt_samples_are_still_discarded(self, config: Config) -> None:
        """The pre-existing gate is untouched."""
        from pmbtc.utils.timeutils import utc_now_ms

        clock = ClockService(config)
        clock.add_sample(
            self._sample("clob", 50.0, config.clock.max_sample_rtt_ms + 1, utc_now_ms())
        )
        assert clock.samples == ()


# --------------------------------------------------------------------------- #
# Root causes 4 & 5: transport liveness is not data freshness
# --------------------------------------------------------------------------- #
class _ScriptedSocket:
    """A socket that yields frames, then goes silent, then optionally fails."""

    def __init__(self, frames: list[str], then: Exception | None = None) -> None:
        self._frames = list(frames)
        self._then = then

    async def recv(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        if self._then is not None:
            raise self._then
        await asyncio.sleep(3600)  # silence, forever
        raise AssertionError("unreachable")

    async def close(self) -> None: ...


class _ProbeFeed(WebSocketFeed):
    def __init__(self, budget_ms: int = 60) -> None:
        super().__init__("probe", "wss://example.invalid", staleness_budget_ms=budget_ms)
        self.handled: list[Any] = []

    async def subscribe(self, ws: Any) -> None: ...

    async def handle(self, payload: Any, received_ms: int) -> None:
        self.handled.append(payload)


class TestTransportVersusFreshness:
    """Root cause 4/5.

    Production: 0 of 1,353,586 archived CLOB inter-frame gaps exceeded the 15 s
    budget, yet 232 stale events tore down healthy sockets — because 17% of
    markets are thin enough to say nothing for 15 s.
    """

    @pytest.mark.asyncio
    async def test_silence_does_not_end_the_connection(self) -> None:
        feed = _ProbeFeed(budget_ms=40)
        ws = _ScriptedSocket(['{"a":1}'])
        task = asyncio.create_task(feed._consume(ws))
        await asyncio.sleep(0.25)
        assert not task.done(), "a quiet market ended the connection"
        assert feed.health.quiet_periods >= 1
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_silence_still_marks_the_data_stale(self) -> None:
        """The safety gate is unchanged: quiet data remains unusable."""
        feed = _ProbeFeed(budget_ms=40)
        ws = _ScriptedSocket(['{"a":1}'])
        task = asyncio.create_task(feed._consume(ws))
        await asyncio.sleep(0.25)
        assert feed.health.state is FeedState.STALE
        assert not feed.health.state.is_usable
        assert not feed.is_fresh()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_the_connection_survives_silence_and_data_resumes(self) -> None:
        """The whole point: a quiet gap must not cost us the connection."""
        feed = _ProbeFeed(budget_ms=40)

        class _Resuming(_ScriptedSocket):
            def __init__(self) -> None:
                super().__init__([])
                self.calls = 0

            async def recv(self) -> str:
                self.calls += 1
                if self.calls == 1:
                    return '{"a":1}'
                if self.calls == 3:
                    return '{"a":2}'
                await asyncio.sleep(3600)  # cancelled by the budget timeout
                raise AssertionError("unreachable")

        task = asyncio.create_task(feed._consume(_Resuming()))
        await asyncio.sleep(0.25)
        assert not task.done(), "the connection did not survive the quiet period"
        assert feed.handled == [{"a": 1}, {"a": 2}], "data did not resume"
        assert feed.health.quiet_periods >= 1
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_an_arriving_frame_clears_the_stale_state(self) -> None:
        """Deterministic: a generous budget means no timeout can race this."""
        feed = _ProbeFeed(budget_ms=5_000)
        feed.health.state = FeedState.STALE
        task = asyncio.create_task(feed._consume(_ScriptedSocket(['{"a":1}'])))
        await asyncio.sleep(0.05)
        assert feed.health.state is FeedState.LIVE
        assert feed.is_fresh()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_transport_failure_still_ends_the_connection(self) -> None:
        """Reconnects must still happen — on transport failure only."""
        feed = _ProbeFeed(budget_ms=1_000)
        ws = _ScriptedSocket(['{"a":1}'], then=ConnectionResetError("socket died"))
        with pytest.raises(ConnectionResetError):
            await feed._consume(ws)

    @pytest.mark.asyncio
    async def test_a_quiet_period_is_counted_once_not_per_timeout(self) -> None:
        feed = _ProbeFeed(budget_ms=30)
        task = asyncio.create_task(feed._consume(_ScriptedSocket(['{"a":1}'])))
        await asyncio.sleep(0.4)
        assert feed.health.quiet_periods == 1, "one silence must not inflate the count"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def test_staleness_budgets_are_unchanged(self, config: Config) -> None:
        """Explicitly pinned: the fix changed behaviour, not thresholds."""
        assert config.feeds.clob_staleness_budget_ms == 15_000
        assert config.feeds.binance_staleness_budget_ms == 10_000
        assert config.dataset.min_snapshot_quality == 0.5
        assert config.dataset.min_coverage == 0.6

    def test_clob_feed_enables_keepalive(self, config: Config) -> None:
        """With silence no longer proving liveness, a ping is the only evidence
        the transport is alive."""
        from pmbtc.live.clob import PolymarketMarketFeed

        feed = PolymarketMarketFeed(url="wss://x.invalid/market", up_token_id="1",
                                    down_token_id="2", slug="s")
        assert feed.ping_interval_s is not None and feed.ping_interval_s > 0
        assert feed.ping_timeout_s is not None and feed.ping_timeout_s > 0
