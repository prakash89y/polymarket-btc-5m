"""Binance BTC feed — the reference-price microstructure source.

Streams ``bookTicker`` (top of book, every change) and ``aggTrade`` (executions)
over one combined socket. This replaces the 1-minute REST bars used in Module 4,
which were up to 59 seconds stale at T-1 and correctly flagged out-of-budget.

This is **not** the settlement series — the 5-minute family settles on Chainlink
BTC/USD, and Binance quotes BTC/USDT. It is a feature source and an audit
reference, and every feature derived from it is named ``ref_*`` so no future
reader mistakes it for the thing that decides the payout.

``aggTrade`` sets ``m`` (is-buyer-maker) rather than a side: ``m=true`` means the
resting order was the buyer, so the aggressor was a seller. Getting that
backwards inverts CVD, which is the most useful flow feature here, so it is
converted once, in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pmbtc.live.archive import TickArchive
from pmbtc.live.feed import WebSocketFeed
from pmbtc.live.tape import Trade, TradeTape
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.live.binance")


class BinanceMarketFeed(WebSocketFeed):
    """Top-of-book and trade tape for one Binance symbol."""

    def __init__(
        self,
        ws_base_url: str,
        symbol: str = "BTCUSDT",
        *,
        staleness_budget_ms: int = 10_000,
        archive: TickArchive | None = None,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        stream = symbol.lower()
        url = f"{ws_base_url.rstrip('/')}/stream?streams={stream}@bookTicker/{stream}@aggTrade"
        super().__init__(
            name=f"binance:{stream}",
            url=url,
            staleness_budget_ms=staleness_budget_ms,
            now_fn=now_fn,
        )
        self.symbol = symbol.upper()
        self.tape = TradeTape()
        self.archive = archive
        self.best_bid: float | None = None
        self.best_ask: float | None = None
        self.best_bid_size: float = 0.0
        self.best_ask_size: float = 0.0
        self.quote_time_ms: int = 0
        self.health.subscriptions = (f"{stream}@bookTicker", f"{stream}@aggTrade")

    async def subscribe(self, ws: Any) -> None:
        # Streams are named in the URL; the combined endpoint needs no message.
        return None

    def on_disconnect(self) -> None:
        """Drop the quote. The tape is a rolling window and stays valid."""
        self.best_bid = self.best_ask = None
        self.best_bid_size = self.best_ask_size = 0.0
        self.quote_time_ms = 0

    # ------------------------------------------------------------------ #
    async def handle(self, payload: Any, received_ms: int) -> None:
        if self.archive is not None:
            self.archive.write(self.name, payload, received_ms)
        if not isinstance(payload, dict):
            self.health.dropped += 1
            return
        data = payload.get("data") if "data" in payload else payload
        if not isinstance(data, dict):
            self.health.dropped += 1
            return

        # aggTrade carries an event time; bookTicker on the combined stream does
        # not, so only the former contributes a latency sample rather than
        # pretending the quote arrived instantly.
        event_ms = int(data.get("E") or data.get("T") or 0)
        if event_ms:
            self.health.observe_latency(event_ms, received_ms)

        if "b" in data and "a" in data and "e" not in data:
            self._on_book_ticker(data, received_ms)
        elif data.get("e") == "aggTrade":
            # `a` is the aggregate trade id: a repeat is a redelivery.
            self.health.observe_hash(f"agg:{data.get('a')}" if data.get("a") else "")
            self._on_agg_trade(data)
        else:
            self.health.dropped += 1

    def _on_book_ticker(self, data: dict[str, Any], received_ms: int) -> None:
        self.best_bid = _as_float(data.get("b"))
        self.best_ask = _as_float(data.get("a"))
        self.best_bid_size = _as_float(data.get("B")) or 0.0
        self.best_ask_size = _as_float(data.get("A")) or 0.0
        # bookTicker on the combined stream carries no event time; the receive
        # timestamp is the honest answer for when we knew it.
        self.quote_time_ms = received_ms

    def _on_agg_trade(self, data: dict[str, Any]) -> None:
        price = _as_float(data.get("p"))
        size = _as_float(data.get("q"))
        if price is None or size is None:
            return
        # m=True -> buyer was the maker -> the aggressor sold.
        sign = -1 if data.get("m") else 1
        timestamp = int(data.get("T") or data.get("E") or 0)
        self.tape.add(Trade(price, size, sign, timestamp))

    # ------------------------------------------------------------------ #
    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0 or self.best_bid is None or self.best_ask is None:
            return None
        return ((self.best_ask - self.best_bid) / mid) * 10_000

    @property
    def quote_imbalance(self) -> float | None:
        total = self.best_bid_size + self.best_ask_size
        if total <= 0:
            return None
        return (self.best_bid_size - self.best_ask_size) / total

    @property
    def last_price(self) -> float | None:
        """Freshest price: the trade tape if it has one, else the quote mid."""
        return self.tape.last_price if self.tape.last_price is not None else self.mid

    def snapshot_state(self, now_ms: int) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "state": self.health.state.value,
            "frames": self.health.frames,
            "mid": self.mid,
            "quote_age_ms": now_ms - self.quote_time_ms if self.quote_time_ms else -1,
            "trades": self.tape.total_count,
        }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
