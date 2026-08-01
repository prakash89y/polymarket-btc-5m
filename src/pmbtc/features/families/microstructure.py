"""Market-microstructure features.

Venue mechanics rather than market opinion: how the book is shaped, how stable
quotes are, and how much of the price is an artefact of the tick grid.

On a 1-cent tick, a 0.50/0.51 book is *one tick wide* — the tightest state
possible — while the same 0.01 spread at a 0.001 tick size would be ten ticks
and genuinely wide. Expressing spread in ticks rather than probability is the
difference between measuring the market and measuring the grid.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_SRC = "polymarket_clob"
_BOOK = f"{RAW}pm_book"


@feature(
    "ms_spread_ticks",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="spread / tick_size",
    inputs=("ob_spread", f"{RAW}market_meta"),
    units="ticks",
    purpose="Spread in units of the minimum increment. A one-tick spread is the "
    "tightest a market can be, whatever the tick happens to be.",
)
def ms_spread_ticks(context: FeatureContext) -> float | None:
    tick = context.regime.get("tick_size")
    return safe_div(context.value("ob_spread"), tick)


@feature(
    "ms_queue_ratio",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="best_bid_size / best_ask_size",
    inputs=(_BOOK,),
    units="ratio",
    purpose="Relative queue lengths at the touch. Determines which side clears "
    "first, and therefore which way the touch is likely to move.",
)
def ms_queue_ratio(context: FeatureContext) -> float | None:
    book = context.up_book
    if book is None:
        return None
    return safe_div(book.best_bid_size, book.best_ask_size)


@feature(
    "ms_book_levels",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="count(bid levels) + count(ask levels)",
    inputs=(_BOOK,),
    units="levels",
    purpose="How populated the ladder is. A book with three levels behaves nothing "
    "like one with sixty, even at the same spread.",
)
def ms_book_levels(context: FeatureContext) -> float | None:
    book = context.up_book
    if book is None:
        return None
    return float(len(book.bids) + len(book.asks))


@feature(
    "ms_quote_stability",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="1 / (1 + |mid_t - mid_{t-1}| / tick_size)",
    inputs=("ob_mid", f"{RAW}market_meta"),
    units="ratio in (0, 1]",
    purpose="How still the quote is between passes. Flickering quotes mean market "
    "makers are uncertain, and any single reading of the book is less meaningful.",
)
def ms_quote_stability(context: FeatureContext) -> float | None:
    mid, prior = context.value("ob_mid"), context.prior("ob_mid")
    tick = context.regime.get("tick_size") or 0.01
    if mid is None or prior is None or tick <= 0:
        return None
    return 1.0 / (1.0 + abs(mid - prior) / tick)


@feature(
    "ms_depth_asymmetry",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="(depth_bid_usdc - depth_ask_usdc) / (depth_bid_usdc + depth_ask_usdc)",
    inputs=("lq_depth_bid_usdc", "lq_depth_ask_usdc"),
    units="ratio in [-1, 1]",
    purpose="Capital-weighted book skew. Complements share-count imbalance, which "
    "over-weights cheap size far from the touch.",
)
def ms_depth_asymmetry(context: FeatureContext) -> float | None:
    bid, ask = context.value("lq_depth_bid_usdc"), context.value("lq_depth_ask_usdc")
    if bid is None or ask is None:
        return None
    total = bid + ask
    return (bid - ask) / total if total > 0 else None


MICROSTRUCTURE_FEATURES = [
    ms_spread_ticks,
    ms_queue_ratio,
    ms_book_levels,
    ms_quote_stability,
    ms_depth_asymmetry,
]
