"""Polymarket CLOB market feed — the primary microstructure source.

Protocol verified against the live socket on 2026-07-31:

``wss://ws-subscriptions-clob.polymarket.com/ws/market``, subscribe with
``{"assets_ids": [...], "type": "market"}``. Three event types arrive:

``book``
    Full L2 snapshot for one asset: ``bids``/``asks`` as ``{price, size}``
    strings, plus ``timestamp`` (ms, string) and ``hash``.
``price_change``
    Batched incremental updates: ``price_changes[]`` each with ``asset_id``,
    ``price``, ``size``, ``side``, and the resulting ``best_bid``/``best_ask``.
``last_trade_price``
    An execution: ``asset_id``, ``price``, ``size``, ``side``, ``timestamp``.

Measured throughput on one 5-minute market: **3,891 frames in 25 seconds**.
Against Gamma's 29-72 second quote staleness, this is the difference between
having microstructure and not.

On reconnect the book is cleared, not kept. An incrementally-maintained book
that missed updates is not stale — it is wrong, and a wrong book prices trades
confidently at the wrong level.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pmbtc.live.archive import TickArchive
from pmbtc.live.book import BookPair
from pmbtc.live.feed import WebSocketFeed
from pmbtc.live.tape import Trade, TradeTape
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.live.clob")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _levels(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        price = _as_float(entry.get("price"))
        size = _as_float(entry.get("size"))
        if price is not None and size is not None:
            out.append((price, size))
    return out


class PolymarketMarketFeed(WebSocketFeed):
    """Streams one market's two outcome books and its trade tape."""

    def __init__(
        self,
        url: str,
        up_token_id: str,
        down_token_id: str,
        *,
        condition_id: str = "",
        slug: str = "",
        staleness_budget_ms: int = 15_000,
        archive: TickArchive | None = None,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(
            name=f"clob:{slug or condition_id[:10]}",
            url=url,
            staleness_budget_ms=staleness_budget_ms,
            # Client-side keepalive is what distinguishes "this market is
            # quiet" from "this socket is dead". While staleness was the
            # reconnect trigger, data arrival doubled as a liveness signal and
            # an extra ping was pure noise on a socket delivering >100
            # frames/second. Now that a quiet feed deliberately stays
            # connected, a ping is the *only* remaining evidence the transport
            # is alive, so it is no longer optional.
            ping_interval_s=20.0,
            now_fn=now_fn,
        )
        self.condition_id = condition_id
        self.slug = slug
        self.books = BookPair()
        self.books.up.asset_id = up_token_id
        self.books.down.asset_id = down_token_id
        self.up_tape = TradeTape()
        self.down_tape = TradeTape()
        self.archive = archive
        self.health.subscriptions = (up_token_id, down_token_id)

    # ------------------------------------------------------------------ #
    async def subscribe(self, ws: Any) -> None:
        await ws.send(
            json.dumps(
                {
                    "assets_ids": [self.books.up.asset_id, self.books.down.asset_id],
                    "type": "market",
                }
            )
        )

    def on_disconnect(self) -> None:
        """Discard the books: after a gap they are wrong, not merely old."""
        self.books.up.clear()
        self.books.down.clear()
        log.info("clob.books_cleared", slug=self.slug, reason="disconnect")

    # ------------------------------------------------------------------ #
    async def handle(self, payload: Any, received_ms: int) -> None:
        if self.archive is not None:
            self.archive.write(self.name, payload, received_ms)
        events = payload if isinstance(payload, list) else [payload]
        for event in events:
            if not isinstance(event, dict):
                self.health.dropped += 1
                continue
            # Transport latency and ordering, from the venue's own timestamp.
            event_ms = _as_int(event.get("timestamp"))
            if event_ms:
                self.health.observe_latency(event_ms, received_ms)
            # The venue's book hash doubles as a frame id: the same hash twice
            # means a redelivered frame, not a new state.
            if self.health.observe_hash(str(event.get("hash", ""))):
                continue

            kind = event.get("event_type")
            if kind == "book":
                self._on_book(event)
            elif kind == "price_change":
                self._on_price_change(event)
            elif kind == "last_trade_price":
                self._on_trade(event)
            else:
                self.health.dropped += 1

    def _on_book(self, event: dict[str, Any]) -> None:
        book = self.books.book_for(str(event.get("asset_id", "")))
        if book is None:
            return
        book.replace(
            bids=_levels(event.get("bids")),
            asks=_levels(event.get("asks")),
            timestamp_ms=_as_int(event.get("timestamp")),
            book_hash=str(event.get("hash", "")),
        )

    def _on_price_change(self, event: dict[str, Any]) -> None:
        timestamp = _as_int(event.get("timestamp"))
        for change in event.get("price_changes") or []:
            if not isinstance(change, dict):
                continue
            book = self.books.book_for(str(change.get("asset_id", "")))
            if book is None:
                continue
            price = _as_float(change.get("price"))
            size = _as_float(change.get("size"))
            side = str(change.get("side", ""))
            if price is None or size is None or not side:
                continue
            book.apply_change(price, size, side, timestamp)

    def _on_trade(self, event: dict[str, Any]) -> None:
        asset_id = str(event.get("asset_id", ""))
        price = _as_float(event.get("price"))
        size = _as_float(event.get("size"))
        if price is None or size is None:
            return
        # "BUY" means the aggressor lifted the offer.
        sign = 1 if str(event.get("side", "")).upper() == "BUY" else -1
        trade = Trade(price, size, sign, _as_int(event.get("timestamp")))
        if asset_id == self.books.up.asset_id:
            self.up_tape.add(trade)
        elif asset_id == self.books.down.asset_id:
            self.down_tape.add(trade)

    # ------------------------------------------------------------------ #
    @property
    def implied_up(self) -> float | None:
        return self.books.implied_up

    def snapshot_state(self, now_ms: int) -> dict[str, Any]:
        """Compact view for logging and the dashboard."""
        return {
            "slug": self.slug,
            "state": self.health.state.value,
            "frames": self.health.frames,
            "implied_up": self.books.implied_up,
            "spread_up": self.books.up.spread,
            "book_age_ms": self.books.up.age_ms(now_ms),
            "trades_up": self.up_tape.total_count,
            "trades_down": self.down_tape.total_count,
        }
