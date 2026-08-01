"""Level-2 order book state and microstructure metrics.

Maintained incrementally from the CLOB stream: a ``book`` event replaces the
whole side, a ``price_change`` event updates individual levels. A level whose
size goes to zero is removed rather than kept at zero, so ``len(bids)`` is a
meaningful depth count.

Everything here is deliberately pure and synchronous — no I/O, no clock. The
feed hands it events and asks it questions, which makes every metric testable
against a hand-built book rather than against a live socket.

Note on Polymarket's book shape, verified from a live snapshot: prices are
strings, levels arrive **ascending by price on both sides**, and the two outcome
tokens are complementary (``P(Up) ≈ 1 - P(Down)``). Nothing here assumes the
arrival order — sides are sorted explicitly, because an ordering assumption that
silently flips would invert every imbalance metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pmbtc.constants import PRICE_MAX, PRICE_MIN


@dataclass(frozen=True, slots=True)
class Level:
    price: float
    size: float

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class OrderBook:
    """One outcome token's book."""

    asset_id: str = ""
    #: price -> size, for each side.
    _bids: dict[float, float] = field(default_factory=dict)
    _asks: dict[float, float] = field(default_factory=dict)
    last_update_ms: int = 0
    #: Venue-supplied book hash, used to detect that we missed an update.
    book_hash: str = ""
    updates: int = 0

    # ------------------------------------------------------------------ #
    def replace(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        timestamp_ms: int,
        book_hash: str = "",
    ) -> None:
        """Apply a full snapshot."""
        self._bids = {p: s for p, s in bids if s > 0}
        self._asks = {p: s for p, s in asks if s > 0}
        self.last_update_ms = timestamp_ms
        self.book_hash = book_hash
        self.updates += 1

    def apply_change(self, price: float, size: float, side: str, timestamp_ms: int) -> None:
        """Apply one incremental level update.

        ``size == 0`` means the level is gone. Keeping a zero level would make
        depth counts lie and put a phantom price at the touch.
        """
        book = self._bids if side.upper() == "BUY" else self._asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size
        self.last_update_ms = max(self.last_update_ms, timestamp_ms)
        self.updates += 1

    # ------------------------------------------------------------------ #
    @property
    def bids(self) -> list[Level]:
        """Best first (descending price)."""
        return [Level(p, self._bids[p]) for p in sorted(self._bids, reverse=True)]

    @property
    def asks(self) -> list[Level]:
        """Best first (ascending price)."""
        return [Level(p, self._asks[p]) for p in sorted(self._asks)]

    @property
    def best_bid(self) -> float | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self._asks) if self._asks else None

    @property
    def best_bid_size(self) -> float:
        best = self.best_bid
        return self._bids.get(best, 0.0) if best is not None else 0.0

    @property
    def best_ask_size(self) -> float:
        best = self.best_ask
        return self._asks.get(best, 0.0) if best is not None else 0.0

    @property
    def is_two_sided(self) -> bool:
        return bool(self._bids) and bool(self._asks)

    @property
    def spread(self) -> float | None:
        if not self.is_two_sided:
            return None
        assert self.best_ask is not None and self.best_bid is not None
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> float | None:
        """Arithmetic mid, clamped away from 0 and 1.

        Clamped because this feeds Kelly sizing and log-loss, both of which
        diverge at the boundaries — and a binary book genuinely does trade at
        0.99/1.00 in the last seconds of a decided market.
        """
        if not self.is_two_sided:
            return None
        assert self.best_ask is not None and self.best_bid is not None
        return min(max((self.best_bid + self.best_ask) / 2.0, PRICE_MIN), PRICE_MAX)

    @property
    def microprice(self) -> float | None:
        """Size-weighted touch price.

        Leans towards the side with less size, because that is the side about
        to be consumed. On a wide binary book this is a materially better
        estimate of fair value than the mid, and it is the single cheapest
        microstructure edge available.
        """
        if not self.is_two_sided:
            return None
        bid_size, ask_size = self.best_bid_size, self.best_ask_size
        total = bid_size + ask_size
        if total <= 0:
            return self.mid
        assert self.best_ask is not None and self.best_bid is not None
        raw = (self.best_bid * ask_size + self.best_ask * bid_size) / total
        return min(max(raw, PRICE_MIN), PRICE_MAX)

    # ------------------------------------------------------------------ #
    def depth(self, levels: int, side: str) -> float:
        """Total size within ``levels`` of the touch."""
        book = self.bids if side.upper() == "BUY" else self.asks
        return sum(level.size for level in book[:levels])

    def notional_depth(self, levels: int, side: str) -> float:
        book = self.bids if side.upper() == "BUY" else self.asks
        return sum(level.notional for level in book[:levels])

    def imbalance(self, levels: int = 1) -> float | None:
        """Order-book imbalance in ``[-1, 1]``; positive means bid-heavy.

        The canonical microstructure predictor: more size resting on the bid
        than the ask is weak evidence the next move is up.
        """
        bid = self.depth(levels, "BUY")
        ask = self.depth(levels, "SELL")
        total = bid + ask
        if total <= 0:
            return None
        return (bid - ask) / total

    def price_for_size(self, size: float, side: str) -> float | None:
        """Average fill price to take ``size`` from one side.

        ``side`` is the side being *lifted*: BUY walks the asks. Returns None
        when the book cannot fill the whole size, which is the honest answer —
        a partial fill on a 5-minute binary is not a position anyone wants.
        """
        book = self.asks if side.upper() == "BUY" else self.bids
        remaining = size
        cost = 0.0
        for level in book:
            take = min(remaining, level.size)
            cost += take * level.price
            remaining -= take
            if remaining <= 0:
                return cost / size
        return None

    def slippage_for_size(self, size: float, side: str) -> float | None:
        """Average fill price minus the touch — the cost of demanding liquidity."""
        fill = self.price_for_size(size, side)
        if fill is None:
            return None
        touch = self.best_ask if side.upper() == "BUY" else self.best_bid
        if touch is None:
            return None
        return abs(fill - touch)

    def age_ms(self, now_ms: int) -> int:
        return now_ms - self.last_update_ms if self.last_update_ms else -1

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self.last_update_ms = 0
        self.book_hash = ""


@dataclass
class BookPair:
    """The two complementary outcome books of one market.

    Polymarket quotes Up and Down as separate tokens whose prices should sum to
    1. They do not always, and the gap is a genuine (if usually tiny) arbitrage
    signal — as well as a data-quality check, since a large gap means one side's
    book is stale.
    """

    up: OrderBook = field(default_factory=OrderBook)
    down: OrderBook = field(default_factory=OrderBook)

    def book_for(self, asset_id: str) -> OrderBook | None:
        if asset_id == self.up.asset_id:
            return self.up
        if asset_id == self.down.asset_id:
            return self.down
        return None

    @property
    def implied_up(self) -> float | None:
        """Best estimate of P(Up), reconciling both books.

        Averages the Up book's mid with the complement of the Down book's mid.
        Two independent readings of the same quantity are better than one, and
        their disagreement is itself recorded.
        """
        up_mid = self.up.mid
        down_mid = self.down.mid
        if up_mid is not None and down_mid is not None:
            return min(max((up_mid + (1.0 - down_mid)) / 2.0, PRICE_MIN), PRICE_MAX)
        if up_mid is not None:
            return up_mid
        if down_mid is not None:
            return min(max(1.0 - down_mid, PRICE_MIN), PRICE_MAX)
        return None

    @property
    def complement_gap(self) -> float | None:
        """``up_mid + down_mid - 1``. Near zero on a healthy pair."""
        up_mid = self.up.mid
        down_mid = self.down.mid
        if up_mid is None or down_mid is None:
            return None
        return up_mid + down_mid - 1.0

    @property
    def is_healthy(self) -> bool:
        return self.up.is_two_sided and self.down.is_two_sided
