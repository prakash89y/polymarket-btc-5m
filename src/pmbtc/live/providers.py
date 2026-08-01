"""Live feature providers.

These plug into the Module 4 collector unchanged: same ``FeatureProvider``
protocol, same provenance obligations, same leakage guard. What changes is the
tier — these are ``REAL_TIME`` features read from a socket that is milliseconds
old, rather than ``DELAYED`` ones read from a cache that is a minute old.

Every feature here is registered in :mod:`pmbtc.features.registry` with its tier
and reproducibility policy. Nothing is recorded that is not declared, and each
one is ``ARCHIVED_STREAM``: it is reproducible precisely because the raw frames
are archived as they arrive. Turn archiving off and these features become
untrainable by policy — which is the honest outcome, not a technicality.

The snapshot instant is the hard boundary. The book is read as-of *now*, but the
tape is always queried with ``now_ms=snapshot_time``, so a trade that lands
between the timer firing and the provider running cannot enter a snapshot that
claims to predate it.
"""

from __future__ import annotations

from pmbtc.dataset.providers import SnapshotContext
from pmbtc.dataset.quality import QualityScorer
from pmbtc.dataset.schema import Observation, QualityFlag
from pmbtc.features.registry import FeatureTier, ReproducibilityPolicy, register
from pmbtc.live.binance import BinanceMarketFeed
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.live.providers")

_STREAM = ReproducibilityPolicy.ARCHIVED_STREAM


def _register_book_features() -> None:
    """Declare every Polymarket microstructure feature."""
    register("pm_book_mid", FeatureTier.REAL_TIME, "polymarket_clob",
             "Mid of the Up book from the live CLOB.", _STREAM, freshness_budget_ms=2_000)
    register("pm_microprice", FeatureTier.REAL_TIME, "polymarket_clob",
             "Size-weighted touch price; leans to the side about to be consumed.",
             _STREAM, freshness_budget_ms=2_000)
    register("pm_implied_up", FeatureTier.REAL_TIME, "polymarket_clob",
             "P(Up) reconciled across both complementary outcome books.",
             _STREAM, freshness_budget_ms=2_000)
    register("pm_complement_gap", FeatureTier.REAL_TIME, "polymarket_clob",
             "up_mid + down_mid - 1; non-zero means arb or a stale side.",
             _STREAM, freshness_budget_ms=2_000)
    register("pm_book_spread", FeatureTier.REAL_TIME, "polymarket_clob",
             "Live touch spread on the Up book.", _STREAM, freshness_budget_ms=2_000)
    for levels in (1, 5, 10, 20):
        register(f"pm_imbalance_{levels}", FeatureTier.REAL_TIME, "polymarket_clob",
                 f"Order-book imbalance over {levels} level(s).", _STREAM,
                 freshness_budget_ms=2_000)
    register("pm_depth_bid_usdc", FeatureTier.REAL_TIME, "polymarket_clob",
             "Notional resting on the bid within 10 levels.", _STREAM,
             freshness_budget_ms=2_000)
    register("pm_depth_ask_usdc", FeatureTier.REAL_TIME, "polymarket_clob",
             "Notional resting on the ask within 10 levels.", _STREAM,
             freshness_budget_ms=2_000)
    register("pm_slippage_100", FeatureTier.REAL_TIME, "polymarket_clob",
             "Average adverse fill taking 100 shares from the ask.", _STREAM,
             freshness_budget_ms=2_000)
    for window in (10, 30, 60, 300):
        register(f"pm_cvd_{window}s", FeatureTier.REAL_TIME, "polymarket_clob",
                 f"Signed executed size over {window}s on the Up token.", _STREAM,
                 freshness_budget_ms=window * 1000)
        register(f"pm_volume_delta_ratio_{window}s", FeatureTier.REAL_TIME,
                 "polymarket_clob", f"CVD normalised by volume over {window}s.",
                 _STREAM, freshness_budget_ms=window * 1000)
    register("pm_trade_intensity_60s", FeatureTier.REAL_TIME, "polymarket_clob",
             "Trades per second on the Up token over 60s.", _STREAM,
             freshness_budget_ms=60_000)
    register("pm_book_age_ms", FeatureTier.DERIVED, "polymarket_clob",
             "Age of the newest book update; a data-quality feature.", _STREAM,
             freshness_budget_ms=5_000)


def _register_reference_features() -> None:
    """Declare every Binance reference-price feature."""
    register("ref_live_price", FeatureTier.REAL_TIME, "binance_spot",
             "Freshest BTC/USDT trade price. Not the settlement series.",
             _STREAM, freshness_budget_ms=2_000)
    register("ref_live_spread_bps", FeatureTier.REAL_TIME, "binance_spot",
             "Top-of-book spread in bps.", _STREAM, freshness_budget_ms=2_000)
    register("ref_quote_imbalance", FeatureTier.REAL_TIME, "binance_spot",
             "Top-of-book size imbalance.", _STREAM, freshness_budget_ms=2_000)
    for window in (10, 30, 60, 300):
        register(f"ref_cvd_{window}s", FeatureTier.REAL_TIME, "binance_spot",
                 f"Signed executed BTC volume over {window}s.", _STREAM,
                 freshness_budget_ms=window * 1000)
        register(f"ref_realized_vol_{window}s_bps", FeatureTier.REAL_TIME, "binance_spot",
                 f"Realised volatility over {window}s, in bps.", _STREAM,
                 freshness_budget_ms=window * 1000)
    register("ref_vwap_60s", FeatureTier.REAL_TIME, "binance_spot",
             "60-second VWAP.", _STREAM, freshness_budget_ms=60_000)
    register("ref_vwap_deviation_bps", FeatureTier.REAL_TIME, "binance_spot",
             "Distance of price from 60s VWAP, in bps.", _STREAM,
             freshness_budget_ms=60_000)
    register("ref_change_from_open_bps", FeatureTier.REAL_TIME, "binance_spot",
             "Move since the window opened, in bps. The dominant feature.",
             _STREAM, freshness_budget_ms=2_000)
    # Distinct from the REST provider's `ref_distance_in_vol`, which is computed
    # from 1-minute bars. Same concept, different freshness — and merging them
    # under one name would silently mix a 2-second feature with a 60-second one.
    register("ref_live_distance_in_vol", FeatureTier.REAL_TIME, "binance_spot",
             "Displacement from open over realised vol, from the live tape.",
             _STREAM, freshness_budget_ms=2_000)
    register("ref_trade_intensity_60s", FeatureTier.REAL_TIME, "binance_spot",
             "Trades per second over 60s.", _STREAM, freshness_budget_ms=60_000)


_register_book_features()
_register_reference_features()


class LivePolymarketProvider:
    """Microstructure from the CLOB stream — the primary source."""

    name = "polymarket_live"
    source = "polymarket_clob"

    def __init__(self, scorer: QualityScorer, feed: PolymarketMarketFeed) -> None:
        self.scorer = scorer
        self.feed = feed

    async def collect(self, context: SnapshotContext) -> list[Observation]:
        instant = context.snapshot_time_ms
        books = self.feed.books
        up = books.up
        tape = self.feed.up_tape

        values: dict[str, float | None] = {
            "pm_book_mid": up.mid,
            "pm_microprice": up.microprice,
            "pm_implied_up": books.implied_up,
            "pm_complement_gap": books.complement_gap,
            "pm_book_spread": up.spread,
            "pm_depth_bid_usdc": up.notional_depth(10, "BUY"),
            "pm_depth_ask_usdc": up.notional_depth(10, "SELL"),
            "pm_slippage_100": up.slippage_for_size(100.0, "BUY"),
            "pm_trade_intensity_60s": tape.trade_intensity(60_000, instant),
            "pm_book_age_ms": float(max(0, up.age_ms(instant))),
        }
        for levels in (1, 5, 10, 20):
            values[f"pm_imbalance_{levels}"] = up.imbalance(levels)
        for window in (10, 30, 60, 300):
            values[f"pm_cvd_{window}s"] = tape.cvd(window * 1000, instant)
            values[f"pm_volume_delta_ratio_{window}s"] = tape.volume_delta_ratio(
                window * 1000, instant
            )

        # The book's own timestamp is the event time; a book that has not ticked
        # is genuinely old and must say so rather than borrow the snapshot's
        # freshness.
        event_time = up.last_update_ms or instant
        flags: tuple[QualityFlag, ...] = ()
        if not self.feed.is_fresh(instant) or not up.is_two_sided:
            flags = (QualityFlag.SOURCE_DEGRADED,)

        return [
            self.scorer.build(
                name=name,
                value=value,
                data_source=self.source,
                event_time_ms=min(event_time, instant),
                observation_time_ms=instant,
                snapshot_time_ms=instant,
                extra_flags=flags,
            )
            for name, value in values.items()
        ]


class LiveReferenceProvider:
    """BTC reference price and flow from the Binance stream."""

    name = "binance_live"
    source = "binance_spot"

    def __init__(
        self, scorer: QualityScorer, feed: BinanceMarketFeed, window_open_price: float | None = None
    ) -> None:
        self.scorer = scorer
        self.feed = feed
        #: Set by the service when a window opens, so displacement is measured
        #: from the price that actually started the window.
        self.window_open_price = window_open_price

    async def collect(self, context: SnapshotContext) -> list[Observation]:
        instant = context.snapshot_time_ms
        tape = self.feed.tape
        price = self.feed.last_price

        vwap = tape.vwap(60_000, instant)
        open_price = self.window_open_price
        change_bps: float | None = None
        if price is not None and open_price:
            change_bps = ((price - open_price) / open_price) * 10_000
        vol_300 = tape.realized_vol(300_000, instant)

        values: dict[str, float | None] = {
            "ref_live_price": price,
            "ref_live_spread_bps": self.feed.spread_bps,
            "ref_quote_imbalance": self.feed.quote_imbalance,
            "ref_vwap_60s": vwap,
            "ref_vwap_deviation_bps": (
                ((price - vwap) / vwap) * 10_000 if price is not None and vwap else None
            ),
            "ref_change_from_open_bps": change_bps,
            "ref_live_distance_in_vol": (
                change_bps / vol_300 if change_bps is not None and vol_300 else None
            ),
            "ref_trade_intensity_60s": tape.trade_intensity(60_000, instant),
        }
        for window in (10, 30, 60, 300):
            values[f"ref_cvd_{window}s"] = tape.cvd(window * 1000, instant)
            values[f"ref_realized_vol_{window}s_bps"] = tape.realized_vol(
                window * 1000, instant
            )

        event_time = min(max(tape.last_trade_ms, self.feed.quote_time_ms) or instant, instant)
        flags: tuple[QualityFlag, ...] = ()
        if not self.feed.is_fresh(instant):
            flags = (QualityFlag.SOURCE_DEGRADED,)

        return [
            self.scorer.build(
                name=name,
                value=value,
                data_source=self.source,
                event_time_ms=event_time,
                observation_time_ms=instant,
                snapshot_time_ms=instant,
                extra_flags=flags,
            )
            for name, value in values.items()
        ]
