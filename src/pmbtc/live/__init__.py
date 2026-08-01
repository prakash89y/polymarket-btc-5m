"""Live market data — the primary microstructure source.

    WebSocketFeed            self-healing socket; silence counts as failure
      -> PolymarketMarketFeed  CLOB book + trade tape (primary)
      -> BinanceMarketFeed     reference price + flow
      -> TickArchive           raw frames, so live features stay reproducible
      -> Live*Provider         observations for the Module 4 collector
      -> CollectionService     continuous, long-running collection

Gamma is not a feed. Measured at 29-72 seconds stale, it supplies discovery,
metadata, settlement information, and historical context only.
"""

from __future__ import annotations

from pmbtc.live.archive import TickArchive, open_tick_archive
from pmbtc.live.binance import BinanceMarketFeed
from pmbtc.live.book import BookPair, Level, OrderBook
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.live.feed import FeedHealth, FeedState, ReplayFeed, WebSocketFeed
from pmbtc.live.providers import LivePolymarketProvider, LiveReferenceProvider
from pmbtc.live.service import CollectionService, MarketSession, ServiceStats
from pmbtc.live.tape import Trade, TradeTape

__all__ = [
    "BinanceMarketFeed",
    "BookPair",
    "CollectionService",
    "FeedHealth",
    "FeedState",
    "Level",
    "LivePolymarketProvider",
    "LiveReferenceProvider",
    "MarketSession",
    "OrderBook",
    "PolymarketMarketFeed",
    "ReplayFeed",
    "ServiceStats",
    "TickArchive",
    "Trade",
    "TradeTape",
    "WebSocketFeed",
    "open_tick_archive",
]
