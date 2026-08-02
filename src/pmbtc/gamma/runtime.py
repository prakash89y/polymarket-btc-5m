"""Assembly of the discovery stack.

One place that knows how the pieces fit together, so the CLI, the paper engine,
and the live engine all get an identically wired system rather than three
slightly different ones.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from pmbtc.clock import ClockService
from pmbtc.config import Config
from pmbtc.gamma.client import GammaClient
from pmbtc.gamma.discovery import MarketDiscovery
from pmbtc.gamma.lifecycle import LifecycleTracker, open_lifecycle_tracker
from pmbtc.gamma.liquidity import LiquidityStore, open_liquidity_store
from pmbtc.gamma.schema import SchemaChecker, open_schema_checker
from pmbtc.logging_setup import get_logger
from pmbtc.settlement import SettlementSpecStore, open_spec_store

log = get_logger("pmbtc.gamma.runtime")


@dataclass
class DiscoveryStack:
    config: Config
    client: GammaClient
    clock: ClockService
    discovery: MarketDiscovery
    spec_store: SettlementSpecStore
    lifecycle: LifecycleTracker
    liquidity: LiquidityStore
    schema: SchemaChecker

    async def sync_clock(self) -> None:
        """Sync against every configured reference."""
        await self.clock.sync(clock_fetchers(self.config, self.client))


def clock_fetchers(config: Config, client: GammaClient) -> dict[str, Any]:
    """Reference-clock callables, keyed by source name.

    ``clob`` returns whole seconds and ``binance`` milliseconds; the clock
    service takes the median, so the coarse source acts as a sanity check on the
    precise one rather than degrading it.
    """

    async def _clob() -> int:
        return await client.server_time_ms()

    async def _binance() -> int:
        url = config.venues.binance_spot.rest_base_url.rstrip("/") + "/api/v3/time"
        async with httpx.AsyncClient(
            timeout=config.http.read_timeout_s,
            headers={"User-Agent": config.http.user_agent},
        ) as http:
            response = await http.get(url)
            response.raise_for_status()
            return int(response.json()["serverTime"])

    available = {"clob": _clob, "binance": _binance}
    return {name: fetch for name, fetch in available.items() if name in config.clock.sources}


@asynccontextmanager
async def discovery_stack(config: Config) -> AsyncIterator[DiscoveryStack]:
    """Build, sync, and tear down the whole discovery stack."""
    config.ensure_directories()
    client = GammaClient(config)
    await client.start()
    clock = ClockService(config)
    spec_store = open_spec_store(config)
    lifecycle = open_lifecycle_tracker(config)
    liquidity = open_liquidity_store(config)
    schema = open_schema_checker(config)
    discovery = MarketDiscovery(
        config,
        client,
        clock,
        spec_store=spec_store,
        lifecycle=lifecycle,
        liquidity=liquidity,
        schema=schema,
    )
    stack = DiscoveryStack(
        config=config,
        client=client,
        clock=clock,
        discovery=discovery,
        spec_store=spec_store,
        lifecycle=lifecycle,
        liquidity=liquidity,
        schema=schema,
    )
    # The resync loop is supervised for the lifetime of the stack. Without it
    # the clock is synced once at startup and never again: a 13.5-hour
    # production run logged one `clock.synced` line, leaving `_last_sync_ms`
    # frozen and the status STALE for all but the first five minutes, with a
    # 1,636 ms correction still being applied against a true offset of 297 ms.
    resync: asyncio.Task[None] | None = None
    try:
        await stack.sync_clock()
        resync = asyncio.create_task(
            clock.run_forever(clock_fetchers(config, client)), name="clock:resync"
        )
        log.info(
            "runtime.clock_resync_started",
            interval_s=config.clock.sync_interval_seconds,
            status=clock.status().value,
        )
        yield stack
    finally:
        if resync is not None:
            resync.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await resync
        await client.aclose()
