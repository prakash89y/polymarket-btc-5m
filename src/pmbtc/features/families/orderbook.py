"""Order-book features.

Resting liquidity: what participants are *offering* to do. Cheap to compute,
available on every tick, and the standard starting point for short-horizon
prediction — with the standard caveat that resting orders can be cancelled,
which is why the trade-flow family exists alongside this one.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_SRC = "polymarket_clob"
_BOOK = f"{RAW}pm_book"


@feature(
    "ob_best_bid",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="max{p : size(p) > 0, p in bids}",
    inputs=(_BOOK,),
    units="probability",
    purpose="Highest price anyone will pay for Up. The floor of executable value.",
)
def ob_best_bid(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.best_bid if book else None


@feature(
    "ob_best_ask",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="min{p : size(p) > 0, p in asks}",
    inputs=(_BOOK,),
    units="probability",
    purpose="Lowest price anyone will sell Up at. The cost of buying immediately.",
)
def ob_best_ask(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.best_ask if book else None


@feature(
    "ob_spread",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="best_ask - best_bid",
    inputs=("ob_best_bid", "ob_best_ask"),
    units="probability",
    purpose="Round-trip cost of demanding liquidity, and a direct proxy for how "
    "confident market makers are.",
)
def ob_spread(context: FeatureContext) -> float | None:
    bid, ask = context.value("ob_best_bid"), context.value("ob_best_ask")
    return ask - bid if bid is not None and ask is not None else None


@feature(
    "ob_mid",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="(best_bid + best_ask) / 2, clamped to [0.001, 0.999]",
    inputs=("ob_best_bid", "ob_best_ask"),
    units="probability",
    purpose="The market's headline probability of Up. The benchmark every model "
    "prediction is measured against.",
)
def ob_mid(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.mid if book else None


@feature(
    "ob_microprice",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="(bid * ask_size + ask * bid_size) / (bid_size + ask_size)",
    inputs=(_BOOK,),
    units="probability",
    purpose="Size-weighted fair value. Leans toward the thinner side, which is the "
    "side about to be consumed, so it leads the mid.",
)
def ob_microprice(context: FeatureContext) -> float | None:
    book = context.up_book
    return book.microprice if book else None


@feature(
    "ob_microprice_edge",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="microprice - mid",
    inputs=("ob_microprice", "ob_mid"),
    units="probability",
    purpose="How far size-weighted value has moved ahead of the mid. A small, "
    "persistent, directional signal.",
)
def ob_microprice_edge(context: FeatureContext) -> float | None:
    micro, mid = context.value("ob_microprice"), context.value("ob_mid")
    return micro - mid if micro is not None and mid is not None else None


def _imbalance(context: FeatureContext, levels: int) -> float | None:
    book = context.up_book
    return book.imbalance(levels) if book else None


@feature(
    "ob_imbalance_1",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="(bid_size_L1 - ask_size_L1) / (bid_size_L1 + ask_size_L1)",
    inputs=(_BOOK,),
    units="ratio in [-1, 1]",
    purpose="Touch imbalance. The classic short-horizon predictor: more size bid "
    "than offered is weak evidence the next tick is up.",
)
def ob_imbalance_1(context: FeatureContext) -> float | None:
    return _imbalance(context, 1)


@feature(
    "ob_imbalance_5",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="(sum bid_size over 5 levels - sum ask_size over 5) / total",
    inputs=(_BOOK,),
    units="ratio in [-1, 1]",
    purpose="Depth-weighted imbalance. Less noisy than the touch, slower to react.",
)
def ob_imbalance_5(context: FeatureContext) -> float | None:
    return _imbalance(context, 5)


@feature(
    "ob_imbalance_20",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="(sum bid_size over 20 levels - sum ask_size over 20) / total",
    inputs=(_BOOK,),
    units="ratio in [-1, 1]",
    purpose="Deep-book imbalance. Closer to positioning than to intent.",
)
def ob_imbalance_20(context: FeatureContext) -> float | None:
    return _imbalance(context, 20)


@feature(
    "ob_imbalance_slope",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="imbalance_1 - imbalance_20",
    inputs=("ob_imbalance_1", "ob_imbalance_20"),
    units="ratio",
    purpose="Whether pressure at the touch agrees with pressure deep in the book. "
    "Disagreement often precedes a reversal.",
)
def ob_imbalance_slope(context: FeatureContext) -> float | None:
    near, far = context.value("ob_imbalance_1"), context.value("ob_imbalance_20")
    return near - far if near is not None and far is not None else None


@feature(
    "ob_complement_gap",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="mid(Up) + mid(Down) - 1",
    inputs=(_BOOK,),
    units="probability",
    purpose="Arbitrage residual across the two complementary tokens. Non-zero means "
    "a real arb or, more often, one stale side — so it is a data-quality signal too.",
)
def ob_complement_gap(context: FeatureContext) -> float | None:
    return context.books.complement_gap if context.books else None


@feature(
    "ob_book_pressure_ratio",
    tier=FeatureTier.REAL_TIME,
    source=_SRC,
    formula="notional_bid_depth_10 / notional_ask_depth_10",
    inputs=(_BOOK,),
    units="ratio",
    purpose="Capital committed to each side, rather than share count. Large size at "
    "a low price is much less capital than the same size near 1.",
)
def ob_book_pressure_ratio(context: FeatureContext) -> float | None:
    book = context.up_book
    if book is None:
        return None
    return safe_div(book.notional_depth(10, "BUY"), book.notional_depth(10, "SELL"))


ORDERBOOK_FEATURES = [
    ob_best_bid,
    ob_best_ask,
    ob_spread,
    ob_mid,
    ob_microprice,
    ob_microprice_edge,
    ob_imbalance_1,
    ob_imbalance_5,
    ob_imbalance_20,
    ob_imbalance_slope,
    ob_complement_gap,
    ob_book_pressure_ratio,
]
