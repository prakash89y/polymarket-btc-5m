"""Trade tape: cumulative volume delta, flow, and realised volatility.

A rolling window of executed trades, from which the flow features are derived.
Trades are kept in a deque trimmed by age, not by count: "CVD over the last 30
seconds" must mean the same thing in a quiet market and a busy one.

CVD (cumulative volume delta) is signed executed volume — buys minus sells.
Unlike book imbalance, which shows resting intent that can be cancelled, CVD
shows what actually traded, and on a short horizon it is the more reliable of
the two.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from math import sqrt


@dataclass(frozen=True, slots=True)
class Trade:
    price: float
    size: float
    #: +1 for a buyer-initiated trade, -1 for seller-initiated.
    sign: int
    timestamp_ms: int

    @property
    def notional(self) -> float:
        return self.price * self.size

    @property
    def signed_size(self) -> float:
        return self.size * self.sign


@dataclass
class TradeTape:
    """Rolling window of trades with flow and volatility metrics."""

    #: Longest window any metric asks for; the deque is trimmed to this.
    max_window_ms: int = 900_000
    trades: deque[Trade] = field(default_factory=deque)
    total_count: int = 0
    last_price: float | None = None
    last_trade_ms: int = 0

    # ------------------------------------------------------------------ #
    def add(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.total_count += 1
        self.last_price = trade.price
        self.last_trade_ms = max(self.last_trade_ms, trade.timestamp_ms)
        self._trim(trade.timestamp_ms)

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - self.max_window_ms
        while self.trades and self.trades[0].timestamp_ms < cutoff:
            self.trades.popleft()

    def window(self, window_ms: int, now_ms: int) -> list[Trade]:
        """Trades in ``(now - window, now]``.

        Bounded above by ``now`` as well as below: when this is called at a
        snapshot instant, trades that happened afterwards must not be counted,
        even though they are sitting in the deque.
        """
        cutoff = now_ms - window_ms
        return [t for t in self.trades if cutoff < t.timestamp_ms <= now_ms]

    # ------------------------------------------------------------------ #
    def cvd(self, window_ms: int, now_ms: int) -> float:
        """Signed executed size: buys minus sells."""
        return sum(t.signed_size for t in self.window(window_ms, now_ms))

    def volume(self, window_ms: int, now_ms: int) -> float:
        return sum(t.size for t in self.window(window_ms, now_ms))

    def notional(self, window_ms: int, now_ms: int) -> float:
        return sum(t.notional for t in self.window(window_ms, now_ms))

    def trade_count(self, window_ms: int, now_ms: int) -> int:
        return len(self.window(window_ms, now_ms))

    def volume_delta_ratio(self, window_ms: int, now_ms: int) -> float | None:
        """CVD normalised by total volume, in ``[-1, 1]``.

        Scale-free, so it is comparable across quiet and busy windows — which
        the raw CVD is not.
        """
        trades = self.window(window_ms, now_ms)
        total = sum(t.size for t in trades)
        if total <= 0:
            return None
        return sum(t.signed_size for t in trades) / total

    def vwap(self, window_ms: int, now_ms: int) -> float | None:
        trades = self.window(window_ms, now_ms)
        total = sum(t.size for t in trades)
        if total <= 0:
            return None
        return sum(t.notional for t in trades) / total

    def price_change(self, window_ms: int, now_ms: int) -> float | None:
        trades = self.window(window_ms, now_ms)
        if len(trades) < 2:
            return None
        return trades[-1].price - trades[0].price

    def realized_vol(self, window_ms: int, now_ms: int) -> float | None:
        """Standard deviation of trade-to-trade log-ish returns, in bps.

        Simple returns rather than log returns: prices here can be probabilities
        near zero, where a log return is unstable and occasionally undefined.
        """
        trades = self.window(window_ms, now_ms)
        if len(trades) < 3:
            return None
        returns = [
            (b.price - a.price) / a.price for a, b in pairwise(trades) if a.price > 0
        ]
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return sqrt(variance) * 10_000

    def trade_intensity(self, window_ms: int, now_ms: int) -> float:
        """Trades per second — the cheapest proxy for "is anything happening"."""
        if window_ms <= 0:
            return 0.0
        return self.trade_count(window_ms, now_ms) / (window_ms / 1000.0)

    def age_ms(self, now_ms: int) -> int:
        return now_ms - self.last_trade_ms if self.last_trade_ms else -1

    def clear(self) -> None:
        self.trades.clear()
        self.last_price = None
        self.last_trade_ms = 0
