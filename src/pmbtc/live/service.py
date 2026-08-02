"""Continuous collection service.

Runs indefinitely, building the dataset while later modules are written. Its job
each window is:

1. discover and verify upcoming markets (Modules 2-3),
2. bring up a live CLOB feed per market shortly *before* it opens, so the book
   is warm and the tape has history when the first snapshot fires,
3. keep one shared Binance feed running for the reference series,
4. hand the live feeds to the Module 4 collector as feature providers,
5. tear down feeds once a window has settled and its cooldown has passed,
6. backfill official labels on a slower cadence.

Two properties matter more than throughput. It must **not lose data on a
transient error** — the loop restarts rather than exits, bounded by a
consecutive-error limit so a persistent fault still surfaces. And it must **not
leak sockets**: every feed is owned by this class and closed on teardown, since
a process that accumulates one socket per 5-minute window would exhaust its file
descriptors within a day.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from pmbtc.clock import ClockService
from pmbtc.config import Config
from pmbtc.dataset.collector import HistoricalCollector
from pmbtc.dataset.store import DatasetStore
from pmbtc.features.provider import EngineeredFeatureProvider
from pmbtc.gamma.client import GammaClient
from pmbtc.gamma.discovery import MarketCandidate, MarketDiscovery
from pmbtc.live.archive import TickArchive
from pmbtc.live.binance import BinanceMarketFeed
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.utils.timeutils import isoformat

log = get_logger("pmbtc.live.service")

service_windows = METRICS.counter(
    "pmbtc_service_windows_total", "Market windows streamed end to end."
)
service_errors = METRICS.counter("pmbtc_service_errors_total", "Service loop errors.")
active_feeds = METRICS.gauge("pmbtc_active_feeds", "Live market feeds currently open.")


@dataclass
class MarketSession:
    """One market's live streaming session."""

    candidate: MarketCandidate
    feed: PolymarketMarketFeed
    #: The canonical engineered-feature provider for this market. One engine
    #: per market, so cross-pass state (velocity, quote stability) belongs to
    #: that market and cannot bleed into the next one.
    provider: EngineeredFeatureProvider
    opened_at_ms: int
    window_open_price: float | None = None

    @property
    def condition_id(self) -> str:
        return self.candidate.spec.condition_id

    @property
    def settlement_ms(self) -> int:
        return self.candidate.spec.window_close_ms


@dataclass
class ServiceStats:
    windows_streamed: int = 0
    snapshots: int = 0
    labels: int = 0
    errors: int = 0
    feeds_opened: int = 0
    by_state: dict[str, int] = field(default_factory=dict)


class CollectionService:
    """Long-running dataset collection with live feeds."""

    def __init__(
        self,
        config: Config,
        client: GammaClient,
        clock: ClockService,
        discovery: MarketDiscovery,
        store: DatasetStore,
    ) -> None:
        self.config = config
        self.client = client
        self.clock = clock
        self.discovery = discovery
        self.store = store
        self.archive = TickArchive(
            config.resolved_path(config.feeds.archive_dir),
            enabled=config.feeds.archive_ticks,
            buffer_frames=config.feeds.archive_buffer_frames,
        )
        self.reference_feed: BinanceMarketFeed | None = None
        self.sessions: dict[str, MarketSession] = {}
        self.stats = ServiceStats()
        #: Last time the heartbeat file was written successfully.
        self._last_heartbeat_ok_ms = 0
        self.collector = HistoricalCollector(
            config, client, clock, discovery, store, providers=[]
        )
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self.config.feeds.binance_enabled:
            self.reference_feed = BinanceMarketFeed(
                self.config.venues.binance_spot.ws_base_url,
                self.config.venues.binance_spot.symbol,
                staleness_budget_ms=self.config.feeds.binance_staleness_budget_ms,
                archive=self.archive,
                now_fn=self.clock.now_ms,
            )
            await self.reference_feed.start()
            log.info("service.reference_feed_started", symbol=self.reference_feed.symbol)

    async def stop(self) -> None:
        self._stop.set()
        for session in list(self.sessions.values()):
            await self._close_session(session)
        if self.reference_feed is not None:
            await self.reference_feed.stop()
        self.archive.close()
        active_feeds.set(0)

    async def __aenter__(self) -> CollectionService:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------ #
    def _providers_for(self, session: MarketSession) -> list[Any]:
        """The canonical pipeline, and nothing else.

        Exactly one provider: the Module 6 engine. It is the only thing in the
        system permitted to turn market data into a feature, so live collection
        and archive replay cannot diverge. Gamma is absent by design.
        """
        provider = session.provider
        provider.window_open_price = session.window_open_price
        provider.tick_size = _tick_size_of(session.candidate)
        return [provider]

    async def _open_session(self, candidate: MarketCandidate) -> MarketSession | None:
        market = candidate.market
        raw_tokens = market.get("clobTokenIds")
        try:
            token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        except (TypeError, json.JSONDecodeError):
            token_ids = None
        if not token_ids or len(token_ids) < 2:
            log.warning("service.no_token_ids", slug=candidate.spec.slug)
            return None

        outcomes = candidate.spec.outcomes
        # Token order follows the outcomes array; do not assume Up is first.
        up_index = 0
        if outcomes and outcomes[0].strip().lower() != "up":
            up_index = 1

        feed = PolymarketMarketFeed(
            url=f"{self.config.polymarket.ws_base_url.rstrip('/')}/market",
            up_token_id=str(token_ids[up_index]),
            down_token_id=str(token_ids[1 - up_index]),
            condition_id=candidate.spec.condition_id,
            slug=candidate.spec.slug,
            staleness_budget_ms=self.config.feeds.clob_staleness_budget_ms,
            archive=self.archive,
            now_fn=self.clock.now_ms,
        )
        await feed.start()
        session = MarketSession(
            candidate=candidate,
            feed=feed,
            provider=EngineeredFeatureProvider(
                self.collector.scorer,
                clob=feed,
                reference=self.reference_feed,
            ),
            opened_at_ms=self.clock.now_ms(),
        )
        self.sessions[session.condition_id] = session
        self.stats.feeds_opened += 1
        active_feeds.set(len(self.sessions))
        log.info(
            "service.session_opened",
            slug=candidate.spec.slug,
            settles=isoformat(session.settlement_ms),
            open_sessions=len(self.sessions),
        )
        return session

    async def _close_session(self, session: MarketSession) -> None:
        await session.feed.stop()
        self.sessions.pop(session.condition_id, None)
        active_feeds.set(len(self.sessions))
        self.stats.windows_streamed += 1
        service_windows.inc()
        log.info(
            "service.session_closed",
            slug=session.candidate.spec.slug,
            frames=session.feed.health.frames,
            reconnects=session.feed.health.reconnects,
        )

    # ------------------------------------------------------------------ #
    async def _sync_sessions(self) -> None:
        """Open feeds for imminent windows, close them after settlement."""
        now = self.clock.now_ms()
        feeds_cfg = self.config.feeds

        # Closed at settlement, not settlement + cooldown. All eleven horizons
        # (T-300 .. T-1) are captured before the window closes, so the extra
        # cooldown held a subscription open on a market that had stopped
        # existing - guaranteed silence, and one more feed competing for the
        # concurrency slot a genuinely active market needed.
        for session in list(self.sessions.values()):
            if now >= session.settlement_ms:
                await self._close_session(session)

        if not feeds_cfg.clob_enabled:
            return

        result = await self.discovery.scan()
        for candidate in sorted(result.candidates, key=lambda c: c.settlement_ms):
            condition_id = candidate.spec.condition_id
            if condition_id in self.sessions:
                continue
            if len(self.sessions) >= feeds_cfg.max_concurrent_markets:
                break
            opens_in = candidate.spec.window_open_ms - now
            if opens_in > feeds_cfg.warmup_seconds * 1000:
                continue
            if now > candidate.settlement_ms:
                continue
            self.collector.register(candidate)
            await self._open_session(candidate)

    def _capture_open_price(self, session: MarketSession) -> None:
        """Record the reference price at the window open, once.

        Displacement from the open is the dominant feature, and it must be
        measured from the price that actually started the window rather than
        from whatever bar happens to be handy later.
        """
        if session.window_open_price is not None or self.reference_feed is None:
            return
        now = self.clock.now_ms()
        window_open = session.candidate.spec.window_open_ms
        if now < window_open:
            return
        price = self.reference_feed.last_price
        if price is not None:
            session.window_open_price = price
            log.info(
                "service.window_open_price",
                slug=session.candidate.spec.slug,
                price=price,
            )

    async def _capture_due_snapshots(self) -> int:
        captured = 0
        now = self.clock.now_ms()
        for session in list(self.sessions.values()):
            self._capture_open_price(session)
            record = self.store.market(session.condition_id)
            if record is None:
                continue
            self.collector.providers = self._providers_for(session)
            for horizon in self.collector.due_horizons(record, now):
                if await self.collector.capture(record, horizon) is not None:
                    captured += 1
            self.collector.sweep_missed(record, now)
        self.stats.snapshots += captured
        return captured


    # ------------------------------------------------------------------ #
    async def run(self, duration_seconds: float | None = None) -> ServiceStats:
        """Run until stopped, or for a fixed duration."""
        deadline = (
            self.clock.now_ms() + int(duration_seconds * 1000)
            if duration_seconds is not None
            else None
        )
        last_scan = 0
        last_label = 0
        last_status = 0
        last_heartbeat = 0
        consecutive_errors = 0

        while not self._stop.is_set():
            if deadline is not None and self.clock.now_ms() >= deadline:
                break
            try:
                now = self.clock.now_ms()
                if now - last_scan >= self.config.gamma.discovery_poll_seconds * 1000:
                    await self._sync_sessions()
                    last_scan = now
                await self._capture_due_snapshots()

                now = self.clock.now_ms()
                if now - last_label >= self.config.service.label_backfill_seconds * 1000:
                    self.stats.labels += await self.collector.resolve_pending_labels()
                    last_label = now
                if now - last_heartbeat >= (
                    self.config.service.heartbeat_interval_seconds * 1000
                ):
                    self._write_heartbeat(now)
                    last_heartbeat = now
                if now - last_status >= self.config.service.status_interval_seconds * 1000:
                    self.log_status()
                    last_status = now

                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_errors += 1
                self.stats.errors += 1
                service_errors.inc()
                log.error(
                    "service.loop_error",
                    error=f"{type(exc).__name__}: {exc}",
                    consecutive=consecutive_errors,
                )
                if (
                    not self.config.service.restart_on_error
                    or consecutive_errors >= self.config.service.max_consecutive_errors
                ):
                    log.error("service.giving_up", consecutive=consecutive_errors)
                    raise
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.config.service.restart_backoff_s
                    )

            # 250 ms keeps the T-2 / T-1 snapshots (one second apart) reachable
            # without spinning.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=0.25)

        self.stats.labels += await self.collector.resolve_pending_labels()
        return self.stats

    def log_status(self) -> None:
        now = self.clock.now_ms()
        feeds = [s.feed.snapshot_state(now) for s in self.sessions.values()]
        self._write_heartbeat(now)
        self._check_heartbeat_self(now)
        self._evaluate_alerts()
        log.info(
            "service.status",
            sessions=len(self.sessions),
            snapshots=self.stats.snapshots,
            labels=self.stats.labels,
            errors=self.stats.errors,
            archived_frames=self.archive.frames_written,
            clock=self.clock.status().value,
            reference=(
                self.reference_feed.snapshot_state(now) if self.reference_feed else None
            ),
            feeds=feeds,
        )

    def _evaluate_alerts(self) -> None:
        """Run the existing alert engine against our own heartbeat.

        Root cause 7: ``ops.alerts.evaluate``/``dispatch`` were reachable only
        from the manual ``pmbtc watch`` command, so a collector that died left a
        262-minute-stale heartbeat and nothing said a word. The logic is reused
        verbatim - this method only supplies the inputs and the schedule, and
        adds no second implementation of any alert condition.

        Alerting must never be able to stop collection, so every failure here is
        swallowed with a warning.
        """
        from pmbtc.ops import (
            AlertState,
            alert_state_path,
            build_summary,
            dispatch,
            evaluate,
            read_heartbeat,
        )

        try:
            state = AlertState(alert_state_path(self.config))
            heartbeat = read_heartbeat(self.config)
            summary = build_summary(self.config, self.store)
            alerts = evaluate(
                self.config, state, heartbeat=heartbeat, summary=summary
            )
            fired = dispatch(self.config, state, alerts)
            if fired:
                log.warning(
                    "service.alerts_fired",
                    count=len(fired),
                    kinds=[a.kind.value for a in fired],
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "service.alert_evaluation_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _write_heartbeat(self, now_ms: int) -> None:
        """Publish liveness. Detailed feed health goes here, not just a ping.

        A process that is running but no longer collecting is the failure worth
        alerting on, so the heartbeat carries enough to judge that.
        """
        from pmbtc.ops.heartbeat import write_heartbeat

        feeds = [s.feed.health.as_dict(now_ms) for s in self.sessions.values()]
        if self.reference_feed is not None:
            feeds.append(self.reference_feed.health.as_dict(now_ms))
        try:
            write_heartbeat(
                self.config,
                {
                    "written_at_ms": now_ms,
                    "pid": os.getpid(),
                    "sessions": len(self.sessions),
                    "snapshots": self.stats.snapshots,
                    "labels": self.stats.labels,
                    "errors": self.stats.errors,
                    "archived_frames": self.archive.frames_written,
                    "clock_status": self.clock.status().value,
                    "feeds": feeds,
                },
            )
            self._last_heartbeat_ok_ms = now_ms
        except OSError as exc:
            # A heartbeat failure must never take down collection.
            log.warning("service.heartbeat_failed", error=str(exc))

    def _check_heartbeat_self(self, now_ms: int) -> None:
        """Notice when our own liveness signal stops being written.

        The heartbeat is how everything else decides collection is alive, so a
        silently failing write is the one blind spot the file itself cannot
        report. If the last successful write is older than the alert engine's
        own staleness rule, say so in the log the operator is already reading.
        """
        last = self._last_heartbeat_ok_ms
        if not last:
            return
        stale_after = self.config.service.status_interval_seconds * 3 * 1000
        age = now_ms - last
        if age > stale_after:
            log.error(
                "service.heartbeat_stale",
                age_ms=age,
                limit_ms=stale_after,
                detail="heartbeat writes are failing; external monitors are blind",
            )

    def status(self) -> dict[str, Any]:
        now = self.clock.now_ms()
        return {
            "sessions": [s.feed.snapshot_state(now) for s in self.sessions.values()],
            "reference": self.reference_feed.snapshot_state(now) if self.reference_feed else None,
            "stats": {
                "windows_streamed": self.stats.windows_streamed,
                "snapshots": self.stats.snapshots,
                "labels": self.stats.labels,
                "errors": self.stats.errors,
                "archived_frames": self.archive.frames_written,
            },
            "clock": self.clock.describe(),
        }


def _tick_size_of(candidate: MarketCandidate) -> float | None:
    """Per-market tick size, read from the verified spec.

    Feeds `ms_spread_ticks`, which is meaningless without it — the same 0.01
    spread is one tick on a 0.01 grid and ten on a 0.001 grid.
    """
    return candidate.spec.tick_size
