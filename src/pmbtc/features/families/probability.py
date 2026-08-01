"""Probability-dynamics features.

The market's own forecast, and how it is moving. These are the only features
that describe the *thing being predicted* rather than its inputs, which makes
them both the most directly relevant and the easiest to misuse: a model that
learns to copy `ob_mid` will score well on accuracy and add no value at all.

Their purpose is therefore mostly as context for the edge calculation — how
fast the market is revising, and whether it is converging or thrashing.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_SRC = "polymarket_clob"


@feature(
    "pb_conviction",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="|mid - 0.5|",
    inputs=("ob_mid",),
    units="probability",
    purpose="How far the market is from a coin flip. Near zero the outcome is "
    "genuinely open; near 0.5 it is effectively decided.",
)
def pb_conviction(context: FeatureContext) -> float | None:
    mid = context.value("ob_mid")
    return abs(mid - 0.5) if mid is not None else None


@feature(
    "pb_velocity",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="(mid_t - mid_{t-1}) / elapsed_seconds",
    inputs=("ob_mid", f"{RAW}clock"),
    units="probability/second",
    purpose="Rate of revision. A fast-moving probability means information is "
    "arriving; a static one means nothing has changed.",
)
def pb_velocity(context: FeatureContext) -> float | None:
    mid, prior = context.value("ob_mid"), context.prior("ob_mid")
    if mid is None or prior is None or context.elapsed_ms <= 0:
        return None
    return (mid - prior) / (context.elapsed_ms / 1000.0)


@feature(
    "pb_acceleration",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="(velocity_t - velocity_{t-1}) / elapsed_seconds",
    inputs=("pb_velocity", f"{RAW}clock"),
    units="probability/second^2",
    purpose="Whether revision is speeding up or petering out. Deceleration near an "
    "extreme often marks the end of a move.",
)
def pb_acceleration(context: FeatureContext) -> float | None:
    velocity, prior = context.value("pb_velocity"), context.prior("pb_velocity")
    if velocity is None or prior is None or context.elapsed_ms <= 0:
        return None
    return (velocity - prior) / (context.elapsed_ms / 1000.0)


@feature(
    "pb_distance_from_open",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="mid_t - mid_at_first_observation",
    inputs=("ob_mid",),
    units="probability",
    purpose="Total revision since we started watching this market. Large values "
    "mean the window has already resolved most of its uncertainty.",
)
def pb_distance_from_open(context: FeatureContext) -> float | None:
    mid = context.value("ob_mid")
    anchor = context.regime.get("mid_at_open")
    return mid - anchor if mid is not None and anchor is not None else None


@feature(
    "pb_logit",
    tier=FeatureTier.DERIVED,
    source=_SRC,
    formula="log(mid / (1 - mid))",
    inputs=("ob_mid",),
    units="log-odds",
    purpose="Probability on a log-odds scale, where moves are roughly additive. "
    "A 0.50->0.55 shift and a 0.90->0.95 shift are very different events, and "
    "the raw probability hides that; the logit does not.",
)
def pb_logit(context: FeatureContext) -> float | None:
    from math import log

    mid = context.value("ob_mid")
    if mid is None or mid <= 0.0 or mid >= 1.0:
        return None
    return log(mid / (1.0 - mid))


@feature(
    "pb_edge_vs_fair",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="(implied_fair_up - mid) / spread, when spread > 0",
    inputs=("vol_model_market_gap", "ob_spread"),
    units="spreads",
    purpose="Model-market disagreement expressed in units of transaction cost. An "
    "edge smaller than the spread is not tradeable however real it is.",
)
def pb_edge_vs_fair(context: FeatureContext) -> float | None:
    return safe_div(context.value("vol_model_market_gap"), context.value("ob_spread"))


PROBABILITY_FEATURES = [
    pb_conviction,
    pb_velocity,
    pb_acceleration,
    pb_distance_from_open,
    pb_logit,
    pb_edge_vs_fair,
]
