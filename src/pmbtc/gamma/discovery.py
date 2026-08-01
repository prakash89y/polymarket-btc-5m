"""Dynamic market discovery and the full acceptance pipeline.

No slug is ever hard-coded. Discovery asks Gamma for the *series* --
``btc-up-or-down-5m`` -- and takes whatever markets it returns. If Polymarket
changes its naming convention tomorrow, discovery is unaffected, because it
never constructed a name in the first place. The observed slug shape is learned
and persisted purely so the emergency fallback stays current.

Every discovered market runs the same gauntlet, in this order:

1. **Schema check** -- fail safe on an incompatible payload before reading it.
2. **Parse** -- into a canonical settlement specification (Module 2).
3. **Liquidity snapshot** -- captured for accepted *and* rejected markets.
4. **Health** -- identifiers, timestamps, cadence, venue state, book.
5. **Settlement verification** -- the nine gates, against recorded history.
6. **Lifecycle** -- derive and persist the state transition.

A market is accepted only if every stage passes. Each stage has its own
rejection reason, and the reasons are a fixed vocabulary so the metric is a
usable breakdown rather than free text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pmbtc.clock import ClockService
from pmbtc.config import Config
from pmbtc.gamma.client import GammaClient
from pmbtc.gamma.health import HealthResult, MarketHealthChecker
from pmbtc.gamma.lifecycle import LifecycleTracker, MarketState, classify
from pmbtc.gamma.liquidity import (
    LiquiditySnapshot,
    LiquidityStore,
    snapshot_from_market,
)
from pmbtc.gamma.schema import SchemaChecker, SchemaCheckResult, SchemaViolation
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import (
    discovery_latency,
    markets_accepted,
    markets_discovered,
    markets_rejected,
    parse_latency,
    settlement_mismatches,
    verification_failures,
)
from pmbtc.settlement import (
    SettlementSpecification,
    SettlementSpecStore,
    SettlementVerifier,
    VerificationResult,
    parse_settlement_spec,
)
from pmbtc.settlement.parser import SettlementParseError
from pmbtc.utils.timeutils import isoformat, window_open

log = get_logger("pmbtc.gamma.discovery")


@dataclass(frozen=True, slots=True)
class MarketCandidate:
    """A market that passed every gate and may be traded."""

    market: dict[str, Any]
    spec: SettlementSpecification
    snapshot: LiquiditySnapshot
    verification: VerificationResult
    state: MarketState
    schema_version: int

    @property
    def condition_id(self) -> str:
        return self.spec.condition_id

    @property
    def settlement_ms(self) -> int:
        return self.spec.window_close_ms


@dataclass(frozen=True, slots=True)
class RejectedMarket:
    slug: str
    condition_id: str
    stage: str
    reason: str
    detail: str
    snapshot: LiquiditySnapshot | None = None


@dataclass
class DiscoveryResult:
    candidates: list[MarketCandidate] = field(default_factory=list)
    rejected: list[RejectedMarket] = field(default_factory=list)
    scanned: int = 0
    source: str = "series"
    scan_ms: float = 0.0

    @property
    def accepted_count(self) -> int:
        return len(self.candidates)

    @property
    def rejection_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.rejected:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts

    def next_tradeable(self) -> MarketCandidate | None:
        """The soonest-settling accepted market -- the one to work on now."""
        open_ones = [c for c in self.candidates if c.state is MarketState.OPEN]
        pool = open_ones or self.candidates
        return min(pool, key=lambda c: c.settlement_ms) if pool else None


class MarketDiscovery:
    """Series-driven discovery with the full acceptance pipeline."""

    def __init__(
        self,
        config: Config,
        client: GammaClient,
        clock: ClockService,
        *,
        spec_store: SettlementSpecStore,
        lifecycle: LifecycleTracker,
        liquidity: LiquidityStore,
        schema: SchemaChecker,
    ) -> None:
        self.config = config
        self.client = client
        self.clock = clock
        self.spec_store = spec_store
        self.lifecycle = lifecycle
        self.liquidity = liquidity
        self.schema = schema
        self.health = MarketHealthChecker(config)
        self._hint_path = config.resolved_path(config.app.data_dir) / "discovery_hints.json"
        # Own the directory rather than assuming a caller ran ensure_directories.
        self._hint_path.parent.mkdir(parents=True, exist_ok=True)
        self._hints = self._load_hints()

    # ------------------------------------------------------------------ #
    # Naming-convention hints
    # ------------------------------------------------------------------ #
    def _load_hints(self) -> dict[str, Any]:
        if not self._hint_path.exists():
            return {}
        try:
            return dict(json.loads(self._hint_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {}

    def _learn_slug_shape(self, slugs: list[str]) -> None:
        """Remember the slug prefix currently in use.

        Only ever used by the fallback path. Discovery proper does not care what
        markets are called, which is the entire point.
        """
        prefixes = {slug.rsplit("-", 1)[0] + "-" for slug in slugs if "-" in slug}
        if len(prefixes) != 1:
            return
        prefix = prefixes.pop()
        if self._hints.get("slug_prefix") == prefix:
            return
        previous = self._hints.get("slug_prefix")
        self._hints["slug_prefix"] = prefix
        self._hint_path.write_text(json.dumps(self._hints, indent=2), encoding="utf-8")
        log.info("discovery.slug_shape_learned", prefix=prefix, previous=previous)

    @property
    def slug_prefix(self) -> str:
        return str(
            self._hints.get("slug_prefix")
            or (self.config.polymarket.market_slug_patterns or ("btc-updown-5m-",))[0]
        )

    # ------------------------------------------------------------------ #
    # Scanning
    # ------------------------------------------------------------------ #
    async def scan(self) -> DiscoveryResult:
        """One full discovery pass."""
        result = DiscoveryResult()
        with discovery_latency.time():
            markets, source = await self._fetch_markets()
            result.source = source
            result.scanned = len(markets)
            markets_discovered.inc(len(markets), source=source)

            seen_conditions: set[str] = set()
            seen_slugs: set[str] = set()
            for market in markets[: self.config.gamma.max_markets_per_scan]:
                self._process(market, result, seen_conditions, seen_slugs)

        log.info(
            "discovery.scan",
            source=result.source,
            scanned=result.scanned,
            accepted=result.accepted_count,
            rejected=len(result.rejected),
            reasons=result.rejection_reasons,
        )
        return result

    async def _fetch_markets(self) -> tuple[list[dict[str, Any]], str]:
        """Series query first; slug construction only as a last resort."""
        now_ms = self.clock.now_ms()
        events = await self.client.events_by_series(
            self.config.polymarket.series_slug,
            end_date_min=isoformat(now_ms),
            limit=self.config.gamma.page_limit,
        )
        markets = _markets_from_events(events)
        if markets:
            self._learn_slug_shape([str(m.get("slug") or "") for m in markets])
            return markets[: self.config.gamma.lookahead_windows + 1], "series"

        if not self.config.gamma.allow_slug_fallback:
            log.warning("discovery.empty", series=self.config.polymarket.series_slug)
            return [], "series"

        # Fallback: the series query returned nothing. Construct the next few
        # window slugs from the corrected clock and try them directly. This is
        # a degraded mode and is logged as such -- if it ever becomes the normal
        # path, the series slug has changed and needs a human.
        log.warning(
            "discovery.fallback_to_slugs",
            series=self.config.polymarket.series_slug,
            prefix=self.slug_prefix,
        )
        window_ms = self.config.window_ms
        base = window_open(now_ms, window_ms)
        found: list[dict[str, Any]] = []
        for index in range(self.config.gamma.lookahead_windows + 1):
            epoch = (base + index * window_ms) // 1000
            for candidate in await self.client.markets_by_slug(f"{self.slug_prefix}{epoch}"):
                found.append(candidate)
        return found, "slug_fallback"

    # ------------------------------------------------------------------ #
    def _process(
        self,
        market: dict[str, Any],
        result: DiscoveryResult,
        seen_conditions: set[str],
        seen_slugs: set[str],
    ) -> None:
        slug = str(market.get("slug") or "")
        condition_id = str(market.get("conditionId") or "")

        def reject(stage: str, reason: str, detail: str, snap: LiquiditySnapshot | None) -> None:
            markets_rejected.inc(reason=reason)
            result.rejected.append(
                RejectedMarket(slug, condition_id, stage, reason, detail, snap)
            )
            log.info("discovery.rejected", slug=slug, stage=stage, reason=reason, detail=detail)
            if condition_id:
                self.lifecycle.observe(
                    condition_id, slug, MarketState.REJECTED, self.clock.now_ms(), reason
                )

        # 1. Schema -- before any field is read for meaning.
        try:
            schema_result: SchemaCheckResult = self.schema.check_or_raise(market)
        except SchemaViolation as exc:
            reject("schema", "schema_incompatible", str(exc), None)
            return

        # 2. Parse.
        try:
            with parse_latency.time():
                spec = parse_settlement_spec(market, now_ms=self.clock.now_ms())
        except SettlementParseError as exc:
            reject("parse", "unparseable", str(exc), None)
            return

        # 3. Liquidity -- captured even for markets about to be rejected.
        now_ms = self.clock.now_ms()
        snapshot = snapshot_from_market(market, captured_at_ms=now_ms)
        self.liquidity.write(snapshot)

        # 4. Health.
        health: HealthResult = self.health.check(
            market,
            spec,
            snapshot,
            now_ms=now_ms,
            seen_condition_ids=seen_conditions,
            seen_slugs=seen_slugs,
        )
        seen_conditions.add(spec.condition_id)
        seen_slugs.add(spec.slug)
        if not health.healthy:
            issue = health.primary_issue
            reject("health", issue.value if issue else "unhealthy", str(health), snapshot)
            return

        # 5. Settlement verification, against what this series did before.
        change = self.spec_store.record(spec)
        if change is not None:
            settlement_mismatches.inc(series=change.series_slug)
            log.error("discovery.settlement_changed", change=str(change))
        known = self.spec_store.known_hash(spec.series_slug)
        verification = SettlementVerifier(
            self.config, known_hash=change.previous_hash if change else known
        ).verify(spec)
        if not verification.trading_enabled:
            verification_failures.inc(status=verification.status.value)
            reject(
                "settlement",
                verification.status.value,
                "; ".join(verification.reasons),
                snapshot,
            )
            return

        # 6. Lifecycle.
        state = classify(
            settlement_ms=spec.window_close_ms,
            window_open_ms=spec.window_open_ms,
            now_ms=now_ms,
            safety_window_s=self.config.clock.order_safety_window_seconds,
            closed=bool(market.get("closed")),
            accepting_orders=market.get("acceptingOrders") is not False,
        )
        self.lifecycle.observe(spec.condition_id, spec.slug, state, now_ms, "discovery")

        markets_accepted.inc()
        result.candidates.append(
            MarketCandidate(
                market=market,
                spec=spec,
                snapshot=snapshot,
                verification=verification,
                state=state,
                schema_version=schema_result.version,
            )
        )


def _markets_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten events into markets, re-attaching the event for the parser.

    Gamma nests markets inside events on this endpoint, but the settlement
    parser expects the market-shaped object with ``events[0]`` embedded -- the
    same shape ``/markets`` returns. Normalising here means one parser, not two.
    """
    markets: list[dict[str, Any]] = []
    for event in events:
        event_summary = {k: v for k, v in event.items() if k != "markets"}
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            merged = dict(market)
            merged.setdefault("events", [event_summary])
            markets.append(merged)
    return markets
