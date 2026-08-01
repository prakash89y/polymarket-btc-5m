"""Feature families.

One module per family, deliberately: a single monolithic feature file becomes
unreviewable at exactly the point where feature correctness starts to matter
more than feature count.

    orderbook     resting liquidity — what participants offer to do
    tradeflow     executed flow — what actually happened
    volume        traded size and its distribution
    liquidity     whether a position can be taken, and at what cost
    volatility    how far price can still travel (the dominant family here)
    microstructure venue mechanics: ticks, queue, quote stability
    probability   the market's own forecast and its dynamics
    timing        exact clock features
    regime        slow-moving context
    crossmarket   prediction market vs underlying
"""

from __future__ import annotations

from pmbtc.features.base import Feature
from pmbtc.features.families.crossmarket import CROSSMARKET_FEATURES
from pmbtc.features.families.liquidity import LIQUIDITY_FEATURES
from pmbtc.features.families.microstructure import MICROSTRUCTURE_FEATURES
from pmbtc.features.families.orderbook import ORDERBOOK_FEATURES
from pmbtc.features.families.probability import PROBABILITY_FEATURES
from pmbtc.features.families.regime import REGIME_FEATURES
from pmbtc.features.families.timing import TIME_FEATURES
from pmbtc.features.families.tradeflow import TRADEFLOW_FEATURES
from pmbtc.features.families.volatility import VOLATILITY_FEATURES
from pmbtc.features.families.volume import VOLUME_FEATURES

FAMILIES: dict[str, list[Feature]] = {
    "orderbook": ORDERBOOK_FEATURES,
    "tradeflow": TRADEFLOW_FEATURES,
    "volume": VOLUME_FEATURES,
    "liquidity": LIQUIDITY_FEATURES,
    "volatility": VOLATILITY_FEATURES,
    "microstructure": MICROSTRUCTURE_FEATURES,
    "probability": PROBABILITY_FEATURES,
    "timing": TIME_FEATURES,
    "regime": REGIME_FEATURES,
    "crossmarket": CROSSMARKET_FEATURES,
}


def all_features() -> list[Feature]:
    """Every declared feature, in a deterministic order."""
    features: list[Feature] = []
    for family in sorted(FAMILIES):
        features.extend(FAMILIES[family])
    return features


__all__ = ["FAMILIES", "all_features"]
