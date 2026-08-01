"""Resilient WebSocket feed base class.

A live feed that dies quietly is worse than no feed: the bot keeps trading on a
book that stopped updating. So this class treats *silence as failure*. If no
frame arrives within the staleness budget the connection is considered dead and
torn down, even though the socket is still nominally open — which is exactly how
a stalled feed presents itself.

Reconnection is bounded-exponential with jitter, and every state change is
observable, because "why did the bot stop trading at 03:00" must be answerable
from the logs alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import websockets

from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.utils.timeutils import utc_now_ms

log = get_logger("pmbtc.live.feed")

feed_frames = METRICS.counter("pmbtc_feed_frames_total", "WebSocket frames received, by feed.")
feed_reconnects = METRICS.counter("pmbtc_feed_reconnects_total", "Feed reconnections, by reason.")
feed_state = METRICS.gauge("pmbtc_feed_connected", "1 when a feed is connected and fresh.")
feed_staleness = METRICS.gauge("pmbtc_feed_staleness_ms", "Age of the newest frame, by feed.")


class FeedState(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LIVE = "live"
    STALE = "stale"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

    @property
    def is_usable(self) -> bool:
        return self is FeedState.LIVE


@dataclass
class FeedHealth:
    """Everything needed to decide whether to trust this feed right now.

    Latency here means *transport* latency: the gap between the venue's own
    event timestamp and our receipt of it. Only frames that carry a usable
    timestamp contribute, so a venue that omits one does not silently report
    zero latency.

    Percentiles matter more than the mean. A feed with 20 ms average latency and
    a 900 ms p99 is a feed that will be stale at exactly the wrong moment, and
    the mean alone hides that completely.
    """

    name: str
    state: FeedState = FeedState.IDLE
    connected_at_ms: int = 0
    #: When this feed *first* came up, so uptime survives reconnects.
    started_at_ms: int = 0
    last_frame_ms: int = 0
    frames: int = 0
    reconnects: int = 0
    last_error: str = ""
    subscriptions: tuple[str, ...] = field(default_factory=tuple)

    # --- integrity counters ------------------------------------------- #
    #: Frames we could not decode or route.
    dropped: int = 0
    #: Frames whose venue sequence/hash we had already seen.
    duplicates: int = 0
    #: Frames whose event timestamp predates one already processed.
    out_of_order: int = 0
    #: Total connected milliseconds, accumulated across sessions.
    connected_ms: int = 0

    #: Bounded latency sample; a full history would grow without limit on a
    #: feed delivering 150 frames per second.
    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=2_000))
    _last_event_ms: int = 0
    _seen_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=512))
    _hash_set: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ #
    def observe_latency(self, event_ms: int, received_ms: int) -> None:
        """Record transport latency for one frame, and order violations."""
        if event_ms <= 0:
            return
        self._latencies.append(float(received_ms - event_ms))
        if self._last_event_ms and event_ms < self._last_event_ms:
            self.out_of_order += 1
        else:
            self._last_event_ms = event_ms

    def observe_hash(self, frame_hash: str) -> bool:
        """Track a venue-supplied frame id. Returns True if it is a duplicate."""
        if not frame_hash:
            return False
        if frame_hash in self._hash_set:
            self.duplicates += 1
            return True
        if len(self._seen_hashes) == self._seen_hashes.maxlen:
            self._hash_set.discard(self._seen_hashes[0])
        self._seen_hashes.append(frame_hash)
        self._hash_set.add(frame_hash)
        return False

    # ------------------------------------------------------------------ #
    def age_ms(self, now_ms: int | None = None) -> int:
        """Heartbeat age: milliseconds since the last frame of any kind."""
        if not self.last_frame_ms:
            return -1
        return (now_ms or utc_now_ms()) - self.last_frame_ms

    def uptime_ms(self, now_ms: int | None = None) -> int:
        """Milliseconds connected, including the session in progress."""
        total = self.connected_ms
        if self.state.is_usable and self.connected_at_ms:
            total += (now_ms or utc_now_ms()) - self.connected_at_ms
        return total

    def availability(self, now_ms: int | None = None) -> float:
        """Connected time as a fraction of time since the feed first started."""
        now = now_ms or utc_now_ms()
        if not self.started_at_ms:
            return 0.0
        span = now - self.started_at_ms
        return min(1.0, self.uptime_ms(now) / span) if span > 0 else 0.0

    def _percentile(self, fraction: float) -> float | None:
        if not self._latencies:
            return None
        ordered = sorted(self._latencies)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    @property
    def mean_latency_ms(self) -> float | None:
        return sum(self._latencies) / len(self._latencies) if self._latencies else None

    @property
    def p50_latency_ms(self) -> float | None:
        return self._percentile(0.50)

    @property
    def p95_latency_ms(self) -> float | None:
        return self._percentile(0.95)

    @property
    def p99_latency_ms(self) -> float | None:
        return self._percentile(0.99)

    def as_dict(self, now_ms: int | None = None) -> dict[str, Any]:
        now = now_ms or utc_now_ms()
        return {
            "name": self.name,
            "state": self.state.value,
            "frames": self.frames,
            "reconnects": self.reconnects,
            "uptime_ms": self.uptime_ms(now),
            "availability": round(self.availability(now), 5),
            "heartbeat_age_ms": self.age_ms(now),
            "mean_latency_ms": _round(self.mean_latency_ms),
            "p50_latency_ms": _round(self.p50_latency_ms),
            "p95_latency_ms": _round(self.p95_latency_ms),
            "p99_latency_ms": _round(self.p99_latency_ms),
            "dropped": self.dropped,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "subscriptions": len(self.subscriptions),
            "last_error": self.last_error,
        }


def _round(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None


class WebSocketFeed(ABC):
    """Base class for a self-healing streaming feed."""

    def __init__(
        self,
        name: str,
        url: str,
        *,
        staleness_budget_ms: int = 15_000,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
        ping_interval_s: float | None = 20.0,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        self.name = name
        self.url = url
        #: Source of receive timestamps. Defaults to the raw local clock, but
        #: the service injects the *corrected* clock — without it, this machine's
        #: measured ~700ms offset shows up as negative transport latency, since
        #: venues stamp events with their own (correct) clocks.
        self.now_fn = now_fn or utc_now_ms
        self.staleness_budget_ms = staleness_budget_ms
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.ping_interval_s = ping_interval_s
        self.health = FeedHealth(name=name)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._ws: Any = None

    # ------------------------------------------------------------------ #
    @abstractmethod
    async def subscribe(self, ws: Any) -> None:
        """Send whatever the venue needs to start streaming."""

    @abstractmethod
    async def handle(self, payload: Any, received_ms: int) -> None:
        """Process one decoded frame."""

    def on_disconnect(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Hook for subclasses to invalidate state a reconnect cannot preserve.

        Intentionally a concrete no-op rather than abstract: a feed holding no
        incremental state has nothing to invalidate and should not be forced to
        write an empty override. Any subclass holding an incrementally
        maintained book *must* clear it here — after a gap such a book is not
        merely stale, it is wrong, and a wrong book is far more dangerous than
        an absent one.
        """

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name=f"feed:{self.name}")

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                self._task.cancel()
                await self._task
            self._task = None
        self.health.state = FeedState.STOPPED
        feed_state.set(0, feed=self.name)

    async def __aenter__(self) -> WebSocketFeed:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            self.health.state = FeedState.CONNECTING
            try:
                async with websockets.connect(
                    self.url,
                    open_timeout=20,
                    ping_interval=self.ping_interval_s,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    await self.subscribe(ws)
                    self.health.state = FeedState.LIVE
                    self.health.connected_at_ms = self.now_fn()
                    if not self.health.started_at_ms:
                        self.health.started_at_ms = self.health.connected_at_ms
                    attempt = 0
                    feed_state.set(1, feed=self.name)
                    log.info("feed.connected", feed=self.name, url=self.url)
                    await self._consume(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.health.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("feed.error", feed=self.name, error=self.health.last_error)

            self._ws = None
            feed_state.set(0, feed=self.name)
            # Bank this session's connected time so uptime survives reconnects.
            if self.health.connected_at_ms:
                self.health.connected_ms += self.now_fn() - self.health.connected_at_ms
                self.health.connected_at_ms = 0
            self.on_disconnect()
            if self._stop.is_set():
                break

            self.health.state = FeedState.RECONNECTING
            self.health.reconnects += 1
            feed_reconnects.inc(feed=self.name)
            # Full jitter: several feeds dropped by the same network blip must
            # not all reconnect on the same tick.
            delay = min(self.backoff_base_s * (2**attempt), self.backoff_max_s)
            attempt += 1
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=random.uniform(0, delay))

    async def _consume(self, ws: Any) -> None:
        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=self.staleness_budget_ms / 1000.0
                )
            except TimeoutError:
                # Silence is failure. The socket may be fine; the data is not.
                self.health.state = FeedState.STALE
                self.health.last_error = (
                    f"no frame for {self.staleness_budget_ms}ms; treating as dead"
                )
                log.warning(
                    "feed.stale", feed=self.name, budget_ms=self.staleness_budget_ms
                )
                feed_reconnects.inc(feed=self.name, reason="stale")
                return

            received = self.now_fn()
            self.health.frames += 1
            self.health.last_frame_ms = received
            feed_frames.inc(feed=self.name)
            feed_staleness.set(0, feed=self.name)

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if not raw or raw.strip() in {"PONG", "PING"}:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self.health.dropped += 1
                log.debug("feed.non_json", feed=self.name, sample=raw[:120])
                continue
            try:
                await self.handle(payload, received)
            except Exception as exc:
                # A handler bug must not kill the connection; the next frame
                # may well be fine, and dropping the feed makes it worse. It is
                # still a dropped message and counted as one.
                self.health.dropped += 1
                log.warning("feed.handler_error", feed=self.name, error=str(exc))

    # ------------------------------------------------------------------ #
    def is_fresh(self, now_ms: int | None = None) -> bool:
        now = now_ms or self.now_fn()
        age = self.health.age_ms(now)
        fresh = self.health.state.is_usable and 0 <= age <= self.staleness_budget_ms
        feed_staleness.set(float(age), feed=self.name)
        return fresh


class ReplayFeed(WebSocketFeed):
    """Feed that replays archived frames instead of connecting.

    Exists so the whole live pipeline — book maintenance, tape, feature
    providers — can be exercised in tests and in historical replay without a
    network, using the exact frames a real session recorded.
    """

    def __init__(self, name: str, frames: list[tuple[int, Any]]) -> None:
        super().__init__(name, url="replay://", staleness_budget_ms=10**9)
        self.frames = frames

    async def subscribe(self, ws: Any) -> None:  # pragma: no cover - unused
        return None

    async def handle(self, payload: Any, received_ms: int) -> None:  # pragma: no cover
        return None

    async def replay(self, handler: Any) -> int:
        """Feed every archived frame through ``handler``. Returns frames sent."""
        for received_ms, payload in self.frames:
            self.health.frames += 1
            self.health.last_frame_ms = received_ms
            await handler(payload, received_ms)
        self.health.state = FeedState.LIVE
        return len(self.frames)


def monotonic_ms() -> int:
    """Monotonic milliseconds, for measuring durations that must not go back."""
    return int(time.monotonic() * 1000)
