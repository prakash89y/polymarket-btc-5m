"""Liquidity features.

Whether a position can actually be taken, and at what cost. These gate
execution as much as they predict direction: a genuine edge in an untradeable
book is not an edge.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_SRC = "polymarket_clob"
_BOOK = f"{RAW}pm_book"


@feature(
    "lq_depth_bid_usdc",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="sum(price * size) over the top 10 bid levels",
    inputs=(_BOOK,),
    units="USDC",
    purpose="Capital resting on the bid — how much can be sold into.",
)
def lq_depth_bid_usdc(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.notional_depth(10, "BUY") if book else None


@feature(
    "lq_depth_ask_usdc",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="sum((1 - price) * size) over the top 10 ask levels",
    inputs=(_BOOK,),
    units="USDC",
    purpose="Capital resting on the ask — how much can be bought.",
)
def lq_depth_ask_usdc(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.notional_depth(10, "SELL") if book else None


@feature(
    "lq_total_depth_usdc",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="depth_bid_usdc + depth_ask_usdc",
    inputs=("lq_depth_bid_usdc", "lq_depth_ask_usdc"),
    units="USDC",
    purpose="Total committed capital. The headline liquidity gate.",
)
def lq_total_depth_usdc(context: FeatureContext) -> float | None:
    bid, ask = context.value("lq_depth_bid_usdc"), context.value("lq_depth_ask_usdc")
    return bid + ask if bid is not None and ask is not None else None


@feature(
    "lq_slippage_100",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="avg_fill_price(100 shares, lifting asks) - best_ask",
    inputs=(_BOOK,),
    units="probability",
    purpose="Actual cost of a realistic clip, walked through the real ladder rather "
    "than assumed from the touch.",
)
def lq_slippage_100(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.slippage_for_size(100.0, "BUY") if book else None


@feature(
    "lq_slippage_500",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="avg_fill_price(500 shares, lifting asks) - best_ask",
    inputs=(_BOOK,),
    units="probability",
    purpose="Cost of a larger clip. The gap between this and the 100-share figure "
    "is how quickly the book thins out.",
)
def lq_slippage_500(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.slippage_for_size(500.0, "BUY") if book else None


@feature(
    "lq_book_convexity",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="slippage_500 / slippage_100",
    inputs=("lq_slippage_500", "lq_slippage_100"),
    units="ratio",
    purpose="How fast liquidity disappears with size. A high ratio means the top of "
    "book is a mirage and real size cannot be done at the quoted price.",
)
def lq_book_convexity(context: FeatureContext) -> float | None:
    return safe_div(context.value("lq_slippage_500"), context.value("lq_slippage_100"))


@feature(
    "lq_spread_to_depth",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="spread / log(1 + total_depth_usdc)",
    inputs=("ob_spread", "lq_total_depth_usdc"),
    units="probability per log-USDC",
    purpose="Cost per unit of available liquidity. Distinguishes a tight-but-thin "
    "book from a wide-but-deep one, which raw spread cannot.",
)
def lq_spread_to_depth(context: FeatureContext) -> float | None:
    from math import log

    spread, depth = context.value("ob_spread"), context.value("lq_total_depth_usdc")
    if spread is None or depth is None or depth < 0:
        return None
    return spread / log(1.0 + depth) if depth > 0 else None


LIQUIDITY_FEATURES = [
    lq_depth_bid_usdc,
    lq_depth_ask_usdc,
    lq_total_depth_usdc,
    lq_slippage_100,
    lq_slippage_500,
    lq_book_convexity,
    lq_spread_to_depth,
]
