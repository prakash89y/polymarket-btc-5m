"""The historical collector.

Runs a precise countdown per market. For each discovered market it computes the
instants T-300, T-240, ... T-1 relative to that market's own settlement time,
sleeps until each one, and captures a snapshot.

Two scheduling rules protect integrity, and both prefer a *missing* snapshot to
a *wrong* one:

1. **A late snapshot is skipped, never backdated.** If the loop wakes 900 ms
   after T-60, the observations it could gather are from after the instant they
   would claim to represent. Recording that is leakage. The miss is logged and
   counted instead.

2. **A snapshot is captured once.** Immutability is enforced by the store, but
   the collector also checks first, so a restart mid-window resumes cleanly
   rather than dying on a duplicate.

Labels are never produced here. Settlement resolution is a separate pass that
runs after the venue publishes an outcome, and it only ever fills in a blank.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pmbtc.clock import ClockService
from pmbtc.config import Config
from pmbtc.dataset.labels import resolve_label
from pmbtc.dataset.providers import (
    BinanceReferenceProvider,
    FeatureProvider,
    PolymarketQuoteProvider,
    SnapshotContext,
    TimeFeatureProvider,
)
from pmbtc.dataset.quality import QualityScorer
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord, Observation
from pmbtc.dataset.store import DatasetStore, ImmutableRecordError
from pmbtc.exceptions import LookaheadError
from pmbtc.gamma.client import GammaClient
from pmbtc.gamma.discovery import MarketCandidate, MarketDiscovery
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.utils.timeutils import isoformat

log = get_logger("pmbtc.dataset.collector")

snapshots_captured = METRICS.counter(
    "pmbtc_snapshots_captured_total", "Feature snapshots written to the dataset."
)
snapshots_missed = METRICS.counter(
    "pmbtc_snapshots_missed_total", "Snapshot instants missed, by reason."
)
labels_resolved = METRICS.counter(
    "pmbtc_labels_resolved_total", "Markets labelled from the official outcome."
)
snapshot_jitter = METRICS.histogram(
    "pmbtc_snapshot_jitter_ms", "Scheduling jitter between nominal and actual capture."
)


@dataclass
class CollectionStats:
    markets_tracked: int = 0
    snapshots_written: int = 0
    snapshots_missed: int = 0
    labels_written: int = 0
    leakage_blocked: int = 0
    misses_by_reason: dict[str, int] = field(default_factory=dict)

    def miss(self, reason: str) -> None:
        self.snapshots_missed += 1
        self.misses_by_reason[reason] = self.misses_by_reason.get(reason, 0) + 1
        snapshots_missed.inc(reason=reason)


class HistoricalCollector:
    """Builds the dataset one market at a time."""

    def __init__(
        self,
        config: Config,
        client: GammaClient,
        clock: ClockService,
        discovery: MarketDiscovery,
        store: DatasetStore,
        providers: list[FeatureProvider] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.clock = clock
        self.discovery = discovery
        self.store = store
        self.scorer = QualityScorer()
        self.providers = providers if providers is not None else self._default_providers()
        self.stats = CollectionStats()
        self._tracked: dict[str, MarketCandidate] = {}
        #: Horizons already reported as missed, so each gap is logged once.
        #: Per-instance: a class attribute here would leak state between
        #: collectors and silently suppress real misses.
        self._known_misses: set[tuple[str, int]] = set()

    def _default_providers(self) -> list[FeatureProvider]:
        return [
            TimeFeatureProvider(self.scorer),
            PolymarketQuoteProvider(self.scorer),
            BinanceReferenceProvider(
                self.scorer,
                self.config.venues.binance_spot.rest_base_url,
                self.config.venues.binance_spot.symbol,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Market registration
    # ------------------------------------------------------------------ #
    def register(self, candidate: MarketCandidate) -> MarketRecord:
        """Record a discovered market's metadata (never its label)."""
        spec = candidate.spec
        event = (candidate.market.get("events") or [{}])[0]
        series = (event.get("series") or [{}])[0] if isinstance(event, dict) else {}
        now = self.clock.now_ms()

        existing = self.store.market(spec.condition_id)
        record = MarketRecord(
            condition_id=spec.condition_id,
            market_id=spec.market_id,
            event_id=str(event.get("id") or ""),
            series_id=str(series.get("id") or ""),
            series_slug=spec.series_slug,
            slug=spec.slug,
            question=spec.question,
            # Discovery time is set once: re-discovering a market later must not
            # move it, or the dataset loses when we first could have acted.
            discovery_time_ms=existing.discovery_time_ms if existing else now,
            open_time_ms=spec.window_open_ms,
            lock_time_ms=spec.window_close_ms
            - self.config.clock.order_safety_window_seconds * 1000,
            settlement_time_ms=spec.window_close_ms,
            settlement_provider=spec.provider,
            settlement_spec_hash=spec.spec_hash,
            settlement_parser_version=spec.parser_version,
            trading_pair=spec.trading_pair,
            tie_rule=spec.tie_rule.value,
            # Label fields deliberately left blank.
            official_outcome=existing.official_outcome if existing else None,
            settlement_probability=existing.settlement_probability if existing else None,
            yes_final_price=existing.yes_final_price if existing else None,
            no_final_price=existing.no_final_price if existing else None,
            resolved_at_ms=existing.resolved_at_ms if existing else None,
            resolution_source=existing.resolution_source if existing else "",
        )
        self.store.upsert_market(record)
        self._tracked[spec.condition_id] = candidate
        self.stats.markets_tracked = len(self._tracked)
        return record

    # ------------------------------------------------------------------ #
    # Snapshot scheduling
    # ------------------------------------------------------------------ #
    def due_horizons(self, record: MarketRecord, now_ms: int) -> list[int]:
        """Horizons whose instant has arrived and can still be captured cleanly."""
        tolerance = self.config.dataset.max_snapshot_jitter_ms
        due: list[int] = []
        for horizon in self.config.dataset.snapshot_horizons_s:
            instant = record.settlement_time_ms - horizon * 1000
            if now_ms < instant:
                continue
            if self.store.has_snapshot(record.condition_id, horizon):
                continue
            if now_ms - instant > tolerance:
                continue
            due.append(horizon)
        return due

    def next_instant_ms(self, record: MarketRecord, now_ms: int) -> int | None:
        """The next uncaptured snapshot instant, for precise sleeping."""
        upcoming = [
            record.settlement_time_ms - h * 1000
            for h in self.config.dataset.snapshot_horizons_s
            if not self.store.has_snapshot(record.condition_id, h)
            and record.settlement_time_ms - h * 1000 > now_ms
        ]
        return min(upcoming) if upcoming else None

    def sweep_missed(self, record: MarketRecord, now_ms: int) -> None:
        """Count horizons that went by uncaptured, so gaps are visible."""
        tolerance = self.config.dataset.max_snapshot_jitter_ms
        for horizon in self.config.dataset.snapshot_horizons_s:
            instant = record.settlement_time_ms - horizon * 1000
            if now_ms - instant <= tolerance:
                continue
            if self.store.has_snapshot(record.condition_id, horizon):
                continue
            if (record.condition_id, horizon) in self._known_misses:
                continue
            self._known_misses.add((record.condition_id, horizon))
            self.stats.miss("late")
            log.warning(
                "collector.snapshot_missed",
                slug=record.slug,
                horizon=horizon,
                late_by_ms=now_ms - instant,
            )

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #
    async def capture(self, record: MarketRecord, horizon_seconds: int) -> FeatureSnapshot | None:
        """Gather every provider's view of one instant and store it."""
        candidate = self._tracked.get(record.condition_id)
        market = candidate.market if candidate else {}
        instant = record.settlement_time_ms - horizon_seconds * 1000
        now = self.clock.now_ms()

        context = SnapshotContext(
            condition_id=record.condition_id,
            slug=record.slug,
            snapshot_time_ms=instant,
            settlement_time_ms=record.settlement_time_ms,
            window_open_ms=record.open_time_ms,
            horizon_seconds=horizon_seconds,
            now_ms=now,
            market=market,
        )

        observations: list[Observation] = []
        for provider in self.providers:
            try:
                observations.extend(await provider.collect(context))
            except Exception as exc:
                # One failing provider must not cost the whole snapshot; its
                # features are recorded as missing so coverage reflects reality.
                log.warning(
                    "collector.provider_failed",
                    provider=provider.name,
                    error=str(exc),
                    slug=record.slug,
                )
                observations.append(
                    self.scorer.missing(provider.name, provider.source, instant)
                )

        snapshot = FeatureSnapshot(
            condition_id=record.condition_id,
            slug=record.slug,
            horizon_seconds=horizon_seconds,
            snapshot_time_ms=instant,
            captured_at_ms=now,
            settlement_time_ms=record.settlement_time_ms,
            observations=tuple(observations),
            clock_offset_ms=self.clock.offset_ms,
            clock_status=self.clock.status().value,
        )

        try:
            self.store.record_snapshot(snapshot)
        except LookaheadError:
            self.stats.leakage_blocked += 1
            self.stats.miss("leakage")
            return None
        except ImmutableRecordError:
            self.stats.miss("duplicate")
            return None

        self.stats.snapshots_written += 1
        snapshots_captured.inc(horizon=str(horizon_seconds))
        snapshot_jitter.observe(float(snapshot.scheduling_jitter_ms))
        log.info(
            "collector.snapshot",
            slug=record.slug,
            horizon=horizon_seconds,
            jitter_ms=snapshot.scheduling_jitter_ms,
            coverage=round(snapshot.coverage, 3),
            quality=round(snapshot.quality_score, 3),
        )
        return snapshot

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #
    async def resolve_pending_labels(self) -> int:
        """Fill in official outcomes for settled markets. Never overwrites."""
        now = self.clock.now_ms()
        written = 0
        for record in self.store.unlabelled_markets():
            if now < record.settlement_time_ms:
                continue
            if now - record.settlement_time_ms > (
                self.config.dataset.resolution_max_wait_s * 1000
            ):
                continue
            try:
                # Ask for the closed set explicitly: a settled market is not in
                # the default (open) result set, and that is the only kind we
                # can label.
                market = await self.client.market_by_condition_id(
                    record.condition_id, closed=True
                )
            except Exception as exc:
                log.warning("collector.resolution_fetch_failed", error=str(exc))
                continue
            if market is None:
                continue

            resolution = resolve_label(market)
            if not resolution.resolved:
                log.debug(
                    "collector.resolution_pending",
                    slug=record.slug,
                    reason=resolution.reason,
                )
                continue

            labelled = record.model_copy(
                update={
                    "official_outcome": resolution.outcome,
                    "settlement_probability": resolution.settlement_probability,
                    "yes_final_price": resolution.yes_final_price,
                    "no_final_price": resolution.no_final_price,
                    "resolved_at_ms": now,
                    "resolution_source": "polymarket_official",
                }
            )
            if self.store.upsert_market(labelled):
                written += 1
                outcome = resolution.outcome.value if resolution.outcome else "void"
                labels_resolved.inc(outcome=outcome)
                log.info(
                    "collector.labelled",
                    slug=record.slug,
                    outcome=outcome,
                    yes=resolution.yes_final_price,
                    no=resolution.no_final_price,
                )
        self.stats.labels_written += written
        return written

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    async def tick(self) -> int:
        """One pass: discover, capture anything due, resolve settled markets."""
        result = await self.discovery.scan()
        for candidate in result.candidates:
            self.register(candidate)

        captured = 0
        now = self.clock.now_ms()
        for record in self.store.markets():
            if record.settlement_time_ms < now - 3_600_000:
                continue  # long past; nothing to capture
            self.sweep_missed(record, now)
            for horizon in self.due_horizons(record, now):
                if await self.capture(record, horizon) is not None:
                    captured += 1
        await self.resolve_pending_labels()
        return captured

    async def run(self, duration_seconds: float) -> CollectionStats:
        """Collect for a fixed wall-clock duration.

        Sleeps until the next snapshot instant rather than polling on a fixed
        interval: T-2 and T-1 are one second apart and a 5-second poll would
        miss both every time.
        """
        deadline = self.clock.now_ms() + int(duration_seconds * 1000)
        last_scan = 0
        while self.clock.now_ms() < deadline:
            now = self.clock.now_ms()
            if now - last_scan >= self.config.gamma.discovery_poll_seconds * 1000:
                result = await self.discovery.scan()
                for candidate in result.candidates:
                    self.register(candidate)
                await self.resolve_pending_labels()
                last_scan = now

            now = self.clock.now_ms()
            for record in self.store.markets():
                self.sweep_missed(record, now)
                for horizon in self.due_horizons(record, now):
                    await self.capture(record, horizon)

            await asyncio.sleep(min(0.25, max(0.0, (deadline - self.clock.now_ms()) / 1000)))

        await self.resolve_pending_labels()
        log.info(
            "collector.finished",
            markets=self.stats.markets_tracked,
            snapshots=self.stats.snapshots_written,
            missed=self.stats.snapshots_missed,
            labels=self.stats.labels_written,
        )
        return self.stats

    def describe_market(self, condition_id: str) -> str:
        record = self.store.market(condition_id)
        if record is None:
            return f"{condition_id[:14]}: unknown"
        timeline = self.store.timeline(condition_id)
        outcome = record.official_outcome.value if record.official_outcome else "pending"
        return (
            f"{record.slug} settles {isoformat(record.settlement_time_ms)} "
            f"snapshots={sorted(timeline, reverse=True)} outcome={outcome}"
        )
