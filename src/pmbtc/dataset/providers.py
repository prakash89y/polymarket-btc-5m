"""Feature providers.

A provider turns "what was true at instant S" into a list of
:class:`~pmbtc.dataset.schema.Observation`. Each one owns its own provenance:
it must state, per value, when the underlying event happened and when we learned
of it. The collector never invents those timestamps on a provider's behalf,
because a provider that cannot say when its data is from is a provider whose
data cannot be trusted.

Module 4 ships the three that need no new infrastructure. The full
microstructure set -- order book imbalance, CVD, trade flow, funding, OI --
arrives with the live collector in Module 5 and plugs in here without the
collector changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from pmbtc.constants import PRICE_MAX, PRICE_MIN
from pmbtc.dataset.quality import QualityScorer
from pmbtc.dataset.schema import Observation
from pmbtc.features.registry import FeatureTier, ReproducibilityPolicy, register
from pmbtc.logging_setup import get_logger
from pmbtc.utils.timeutils import (
    is_weekend,
    minute_of_day,
    session_of,
    to_ms,
)

log = get_logger("pmbtc.dataset.providers")


@dataclass(frozen=True, slots=True)
class SnapshotContext:
    """Everything a provider is allowed to know at snapshot time.

    Note what is absent: the outcome, the settled prices, anything from after
    ``snapshot_time_ms``. Providers cannot leak what they are never given.
    """

    condition_id: str
    slug: str
    snapshot_time_ms: int
    settlement_time_ms: int
    window_open_ms: int
    horizon_seconds: int
    now_ms: int
    market: dict[str, Any]

    @property
    def seconds_to_settlement(self) -> float:
        return (self.settlement_time_ms - self.snapshot_time_ms) / 1000.0

    @property
    def window_progress(self) -> float:
        span = self.settlement_time_ms - self.window_open_ms
        if span <= 0:
            return 0.0
        return (self.snapshot_time_ms - self.window_open_ms) / span


class FeatureProvider(Protocol):
    """Contract every feature source implements."""

    name: str
    source: str

    async def collect(self, context: SnapshotContext) -> list[Observation]: ...


# --------------------------------------------------------------------------- #
# Declarations. No feature may be recorded without one -- including the
# polled/REST features, which predate the registry and were caught unregistered
# by the Module 5 validation pass.
# --------------------------------------------------------------------------- #
_REST = ReproducibilityPolicy.REST_REPLAYABLE
_DERIVED = ReproducibilityPolicy.DERIVED_FROM_ARCHIVE

for _name, _description in (
    ("seconds_to_settlement", "Seconds remaining in the window."),
    ("window_progress", "Fraction of the window elapsed."),
    ("sqrt_time_remaining", "sqrt(1 - progress): the scale of remaining uncertainty."),
    ("minute_of_day", "Minutes since 00:00 UTC."),
    ("is_weekend", "Saturday or Sunday in UTC."),
    ("session_asia", "Asia session indicator."),
    ("session_london", "London session indicator."),
    ("session_overlap", "London/NY overlap indicator."),
    ("session_new_york", "New York session indicator."),
    ("session_late_us", "Late-US session indicator."),
):
    register(
        _name, FeatureTier.DERIVED, "derived", _description, _DERIVED,
        group="short_term", freshness_budget_ms=1_000,
    )

for _name, _description in (
    ("pm_mid", "Gamma cached mid. DELAYED tier: measured 29-72s stale."),
    ("pm_best_bid", "Gamma cached best bid."),
    ("pm_best_ask", "Gamma cached best ask."),
    ("pm_spread", "Gamma cached spread."),
    ("pm_reported_spread", "Spread as Gamma reports it."),
    ("pm_last_trade", "Gamma cached last trade price."),
    ("pm_liquidity", "Gamma reported resting liquidity."),
    ("pm_volume", "Gamma reported traded volume."),
    ("pm_conviction", "Distance of the Gamma mid from 0.5."),
):
    register(
        _name, FeatureTier.DELAYED, "polymarket_gamma", _description, _REST,
        group="regime", freshness_budget_ms=60_000,
    )

for _name, _description in (
    ("ref_price", "BTC close from the most recent completed 1m bar."),
    ("ref_open_price", "BTC price at the window open, from 1m bars."),
    ("ref_change_bps", "Move since the window open, in bps (1m granularity)."),
    ("ref_realized_vol_bps", "Realised volatility from 1m bars, in bps."),
    ("ref_distance_in_vol", "Displacement from open divided by realised vol."),
    ("ref_up_bars_ratio", "Fraction of recent 1m bars that closed up."),
):
    register(
        _name, FeatureTier.NEAR_REAL_TIME, "binance_spot", _description, _REST,
        group="short_term", freshness_budget_ms=60_000,
    )


# --------------------------------------------------------------------------- #
class TimeFeatureProvider:
    """Deterministic features of the clock itself.

    These are exact by construction -- the event time *is* the snapshot instant --
    so they are the one source with no latency and no staleness.
    """

    name = "time"
    source = "derived"

    def __init__(self, scorer: QualityScorer) -> None:
        self.scorer = scorer

    async def collect(self, context: SnapshotContext) -> list[Observation]:
        instant = context.snapshot_time_ms
        session = session_of(instant)
        values: dict[str, float] = {
            "seconds_to_settlement": context.seconds_to_settlement,
            "window_progress": context.window_progress,
            # sqrt of remaining time: the natural scale for how much
            # uncertainty is left in the window.
            "sqrt_time_remaining": max(0.0, 1.0 - context.window_progress) ** 0.5,
            "minute_of_day": float(minute_of_day(instant)),
            "is_weekend": 1.0 if is_weekend(instant) else 0.0,
            "session_asia": 1.0 if session.value == "asia" else 0.0,
            "session_london": 1.0 if session.value == "london" else 0.0,
            "session_overlap": 1.0 if session.value == "london_ny_overlap" else 0.0,
            "session_new_york": 1.0 if session.value == "new_york" else 0.0,
            "session_late_us": 1.0 if session.value == "late_us" else 0.0,
        }
        return [
            self.scorer.build(
                name=name,
                value=value,
                data_source=self.source,
                event_time_ms=instant,
                observation_time_ms=instant,
                snapshot_time_ms=instant,
            )
            for name, value in values.items()
        ]


# --------------------------------------------------------------------------- #
class PolymarketQuoteProvider:
    """The market's own implied probability and book state.

    The single most important non-price feature: whatever the model believes,
    the edge is measured against *this*. ``updatedAt`` is used as the event time
    because Gamma's quote fields are cached, and pretending they are live would
    understate their latency.
    """

    name = "polymarket_quote"
    source = "polymarket_gamma"

    def __init__(self, scorer: QualityScorer) -> None:
        self.scorer = scorer

    async def collect(self, context: SnapshotContext) -> list[Observation]:
        market = context.market
        event_time = _event_time_of(market, context.snapshot_time_ms)
        observed = min(context.now_ms, context.snapshot_time_ms)

        best_bid = _as_float(market.get("bestBid"))
        best_ask = _as_float(market.get("bestAsk"))
        mid: float | None = None
        spread: float | None = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid = min(max((best_bid + best_ask) / 2.0, PRICE_MIN), PRICE_MAX)

        values: dict[str, float | None] = {
            "pm_mid": mid,
            "pm_best_bid": best_bid,
            "pm_best_ask": best_ask,
            "pm_spread": spread,
            "pm_reported_spread": _as_float(market.get("spread")),
            "pm_last_trade": _as_float(market.get("lastTradePrice")),
            "pm_liquidity": _as_float(market.get("liquidity")),
            "pm_volume": _as_float(market.get("volume")),
            # Distance of the market's probability from a coin flip: the shape
            # of this over the countdown is the market's own conviction curve.
            "pm_conviction": abs(mid - 0.5) if mid is not None else None,
        }
        return [
            self.scorer.build(
                name=name,
                value=value,
                data_source=self.source,
                event_time_ms=event_time,
                observation_time_ms=observed,
                snapshot_time_ms=context.snapshot_time_ms,
            )
            for name, value in values.items()
        ]


# --------------------------------------------------------------------------- #
class BinanceReferenceProvider:
    """Reference BTC price and displacement from the window open.

    Distance from the open, normalised by realised volatility, is the dominant
    feature in this market -- it is most of what a fair-value model needs. This
    is the minimum viable version: 1-minute closes from Binance spot, which is
    freely replayable.

    Two honest caveats, both recorded in the data rather than the docs:

    * Binance BTC/USDT is **not** the settlement series (that is Chainlink
      BTC/USD). It is a feature and an audit reference, never a label.
    * 1-minute bars are coarse for a 300-second instrument. Module 5 replaces
      this with the live trade tape; the schema does not change.
    """

    name = "binance_reference"
    source = "binance_spot"

    def __init__(self, scorer: QualityScorer, base_url: str, symbol: str = "BTCUSDT") -> None:
        self.scorer = scorer
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def collect(self, context: SnapshotContext) -> list[Observation]:
        names = (
            "ref_price",
            "ref_open_price",
            "ref_change_bps",
            "ref_realized_vol_bps",
            "ref_distance_in_vol",
            "ref_up_bars_ratio",
        )
        try:
            bars = await self._fetch_bars(context)
        except Exception as exc:
            log.warning("provider.binance_failed", error=str(exc), slug=context.slug)
            return [
                self.scorer.missing(name, self.source, context.snapshot_time_ms)
                for name in names
            ]

        # Only bars that closed at or before the snapshot instant may be used.
        # This is the leakage rule applied at the source, not left to the guard.
        usable = [b for b in bars if b["close_time_ms"] <= context.snapshot_time_ms]
        if not usable:
            return [
                self.scorer.missing(name, self.source, context.snapshot_time_ms)
                for name in names
            ]

        latest = usable[-1]
        price = latest["close"]
        event_time = latest["close_time_ms"]

        opens = [b for b in usable if b["open_time_ms"] >= context.window_open_ms]
        open_price = opens[0]["open"] if opens else usable[0]["open"]

        change_bps = ((price - open_price) / open_price) * 10_000 if open_price else None

        returns = [
            (b["close"] - b["open"]) / b["open"] for b in usable[-30:] if b["open"]
        ]
        vol_bps: float | None = None
        if len(returns) >= 5:
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
            vol_bps = (variance**0.5) * 10_000

        distance_in_vol = (
            change_bps / vol_bps if change_bps is not None and vol_bps else None
        )
        up_ratio = (
            sum(1 for r in returns if r > 0) / len(returns) if returns else None
        )

        values: dict[str, float | None] = {
            "ref_price": price,
            "ref_open_price": open_price,
            "ref_change_bps": change_bps,
            "ref_realized_vol_bps": vol_bps,
            "ref_distance_in_vol": distance_in_vol,
            "ref_up_bars_ratio": up_ratio,
        }
        return [
            self.scorer.build(
                name=name,
                value=value,
                data_source=self.source,
                event_time_ms=event_time,
                observation_time_ms=min(context.now_ms, context.snapshot_time_ms),
                snapshot_time_ms=context.snapshot_time_ms,
            )
            for name, value in values.items()
        ]

    async def _fetch_bars(self, context: SnapshotContext) -> list[dict[str, Any]]:
        """1-minute klines covering the window and enough history for vol."""
        http = await self._http()
        start = context.window_open_ms - 45 * 60_000
        response = await http.get(
            f"{self.base_url}/api/v3/klines",
            params={
                "symbol": self.symbol,
                "interval": "1m",
                "startTime": start,
                "endTime": context.snapshot_time_ms,
                "limit": 200,
            },
        )
        response.raise_for_status()
        bars: list[dict[str, Any]] = []
        for row in response.json():
            bars.append(
                {
                    "open_time_ms": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    # Binance publishes an inclusive close time; +1ms makes it
                    # the exclusive boundary the rest of the codebase uses.
                    "close_time_ms": int(row[6]) + 1,
                }
            )
        return bars


# --------------------------------------------------------------------------- #
def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_time_of(market: dict[str, Any], default_ms: int) -> int:
    raw = market.get("updatedAt")
    if isinstance(raw, str) and raw.strip():
        try:
            return to_ms(raw)
        except Exception:
            return default_ms
    return default_ms
