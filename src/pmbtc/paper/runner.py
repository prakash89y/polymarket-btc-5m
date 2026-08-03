"""The live paper-trading loop.

Wires the Module 5 feeds and the Module 3 discovery stack to the Module 9
engine. Deliberately thin: everything it touches already exists, and the only
new behaviour is *when* to evaluate, *when* to settle, and how to read the tape
afterwards.

It runs in its own process rather than inside :class:`CollectionService`. The
collector was stabilised over v0.12.0 and v0.13.0 and now runs under
supervision; adding a trading loop to it would put the dataset — the asset the
whole project is accumulating — behind a change that has nothing to do with
collecting it. Paper trading is read-only with respect to the dataset and can
fail without costing a single snapshot.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from pmbtc.config import Config
from pmbtc.constants import Outcome
from pmbtc.dataset.labels import resolve_label
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.logging_setup import get_logger
from pmbtc.paper.engine import LiveQuoteSource, PaperStats, PaperTradingEngine
from pmbtc.paper.execution import TapePrint, measure_execution

log = get_logger("pmbtc.paper.runner")


@dataclass
class _Session:
    condition_id: str
    slug: str
    settlement_time_ms: int
    feed: PolymarketMarketFeed
    tokens: tuple[str, str]
    decided: bool = False


@dataclass
class PaperRunner:
    """Discovers live markets, evaluates them, and settles them."""

    config: Config
    stack: Any
    engine: PaperTradingEngine
    quotes: LiveQuoteSource
    sessions: dict[str, _Session] = field(default_factory=dict)
    stats: PaperStats = field(default_factory=PaperStats)

    # ------------------------------------------------------------------ #
    async def run(self, duration_seconds: float | None = None) -> PaperStats:
        clock = self.stack.clock
        deadline = (
            clock.now_ms() + int(duration_seconds * 1000)
            if duration_seconds is not None
            else None
        )
        last_scan = 0
        last_settle = 0
        try:
            while True:
                now = clock.now_ms()
                if deadline is not None and now >= deadline:
                    break
                try:
                    if now - last_scan >= self.config.gamma.discovery_poll_seconds * 1000:
                        await self._sync_sessions()
                        last_scan = now
                    self._evaluate_due()
                    if now - last_settle >= self.config.service.label_backfill_seconds * 1000:
                        await self._settle_pending()
                        last_settle = now
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.stats.errors += 1
                    log.error(
                        "paper.loop_error", error=f"{type(exc).__name__}: {exc}"
                    )
                await asyncio.sleep(0.5)
        finally:
            await self._close_all()
            # One last settlement sweep so a run that ends mid-window still
            # resolves whatever Polymarket has already published.
            with contextlib.suppress(Exception):
                await self._settle_pending()
        self.stats.evaluated = self.engine.stats.evaluated
        self.stats.traded = self.engine.stats.traded
        self.stats.skipped = self.engine.stats.skipped
        self.stats.settled = self.engine.stats.settled
        return self.stats

    # ------------------------------------------------------------------ #
    async def _sync_sessions(self) -> None:
        now = self.stack.clock.now_ms()
        for session in list(self.sessions.values()):
            if now >= session.settlement_time_ms:
                await self._close(session)

        result = await self.stack.discovery.scan()
        for candidate in sorted(result.candidates, key=lambda c: c.settlement_ms):
            spec = candidate.spec
            if spec.condition_id in self.sessions:
                continue
            if len(self.sessions) >= self.config.feeds.max_concurrent_markets:
                break
            if now >= candidate.settlement_ms:
                continue
            # Subscribe within the warmup window so the book is populated by the
            # time the decision horizon arrives.
            if spec.window_open_ms - now > self.config.feeds.warmup_seconds * 1000:
                continue
            await self._open(candidate)

    async def _open(self, candidate: Any) -> None:
        spec = candidate.spec
        raw = candidate.market.get("clobTokenIds")
        try:
            tokens = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            tokens = None
        if not tokens or len(tokens) < 2:
            return
        outcomes = spec.outcomes
        up_index = 0 if not outcomes or outcomes[0].strip().lower() == "up" else 1

        feed = PolymarketMarketFeed(
            url=self.config.polymarket.ws_base_url.rstrip("/") + "/market",
            up_token_id=str(tokens[up_index]),
            down_token_id=str(tokens[1 - up_index]),
            condition_id=spec.condition_id,
            slug=spec.slug,
            staleness_budget_ms=self.config.feeds.clob_staleness_budget_ms,
            archive=None,  # the collector already archives; do not double-write
            now_fn=self.stack.clock.now_ms,
        )
        await feed.start()
        session = _Session(
            condition_id=spec.condition_id,
            slug=spec.slug,
            settlement_time_ms=candidate.settlement_ms,
            feed=feed,
            tokens=(str(tokens[up_index]), str(tokens[1 - up_index])),
        )
        self.sessions[spec.condition_id] = session
        self.quotes.register(spec.condition_id, feed)
        log.info("paper.session_opened", slug=spec.slug, open_sessions=len(self.sessions))

    async def _close(self, session: _Session) -> None:
        await session.feed.stop()
        self.quotes.unregister(session.condition_id)
        self.sessions.pop(session.condition_id, None)

    async def _close_all(self) -> None:
        for session in list(self.sessions.values()):
            with contextlib.suppress(Exception):
                await self._close(session)

    # ------------------------------------------------------------------ #
    def _evaluate_due(self) -> None:
        for session in list(self.sessions.values()):
            if session.decided:
                continue
            record = self.engine.evaluate(
                condition_id=session.condition_id,
                slug=session.slug,
                settlement_time_ms=session.settlement_time_ms,
            )
            if record is not None:
                session.decided = True

    # ------------------------------------------------------------------ #
    async def _settle_pending(self) -> None:
        """Resolve decisions whose market Polymarket has now published.

        The outcome is Polymarket's own resolution and nothing else — the same
        rule Module 4 applies to training labels. A paper P&L computed from a
        reconstruction would be scoring our reconstruction, not our trading.
        """
        pending = self.engine.journal.pending()
        if not pending:
            return
        for decision in pending:
            if self.engine.now_ms() < decision.settlement_time_ms:
                continue
            market = await self._fetch_market(decision.condition_id)
            if market is None:
                continue
            resolution = resolve_label(market)
            if not resolution.resolved:
                continue
            quality = self._measure(decision)
            self.engine.settle(decision, resolution.outcome, quality=quality)

    async def _fetch_market(self, condition_id: str) -> dict[str, Any] | None:
        try:
            # closed=true is mandatory: Gamma excludes settled markets by
            # default and would silently return nothing for every one of them.
            return await self.stack.client.market_by_condition_id(
                condition_id, closed=True
            )
        except Exception as exc:
            log.warning("paper.settle_fetch_failed", error=str(exc))
            return None

    def _measure(self, decision: Any) -> dict[str, Any] | None:
        """Score the simulated order against the tape that followed it."""
        session = self.sessions.get(decision.condition_id)
        if session is None or not decision.traded:
            return None
        tape = session.feed.up_tape
        prints = [
            TapePrint(price=t.price, size=t.size, timestamp_ms=t.timestamp_ms)
            for t in list(tape.trades)
        ]
        outcome = Outcome(decision.outcome) if decision.outcome else None
        quality = measure_execution(
            outcome=outcome,
            expected_price=decision.expected_price,
            decided_at_ms=decision.decided_at_ms,
            prints=prints,
        )
        return quality.as_dict()
