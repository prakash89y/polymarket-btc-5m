"""Regime features.

Slow-moving context: funding, open interest, sentiment. These update on
timescales from minutes to a day, so they cannot possibly *trigger* a trade on a
300-second instrument — and the tier system says so explicitly, since
``FeatureTier.DELAYED`` is not a tradeable signal.

Their honest purpose is conditioning: the same order-book imbalance means
something different in a high-funding, crowded-long regime than in a balanced
one. They are included so the model can learn that conditioning, and they are
expected to be pruned hard by the selection stage. That is a legitimate outcome,
not a failure.

Values arrive through ``context.regime``, populated by the auxiliary collectors.
A missing key yields None rather than a default, because a fabricated neutral
value is indistinguishable to the model from a real one.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature
from pmbtc.features.registry import FeatureTier, ReproducibilityPolicy

_REST = ReproducibilityPolicy.REST_REPLAYABLE
_REGIME = f"{RAW}regime"
_HOUR = 3_600_000


def _regime_value(context: FeatureContext, key: str) -> float | None:
    value = context.regime.get(key)
    return float(value) if value is not None else None


@feature(
    "rg_funding_rate",
    tier=FeatureTier.DELAYED,
    source="coinglass",
    formula="latest perpetual funding rate for BTC",
    inputs=(_REGIME,),
    units="rate per interval",
    purpose="Cost of holding leveraged length. Extreme funding marks crowded "
    "positioning, which changes how the market responds to a shock.",
    reproducibility=_REST,
    group="regime",
    freshness_budget_ms=8 * _HOUR,
)
def rg_funding_rate(context: FeatureContext) -> float | None:
    return _regime_value(context, "funding_rate")


@feature(
    "rg_open_interest_change_1h",
    tier=FeatureTier.DELAYED,
    source="coinglass",
    formula="(oi_now - oi_1h_ago) / oi_1h_ago",
    inputs=(_REGIME,),
    units="fraction",
    purpose="Whether leverage is being added or unwound. Rising OI into a move "
    "means conviction; falling OI means liquidation.",
    reproducibility=_REST,
    group="regime",
    freshness_budget_ms=_HOUR,
)
def rg_open_interest_change_1h(context: FeatureContext) -> float | None:
    return _regime_value(context, "open_interest_change_1h")


@feature(
    "rg_long_short_ratio",
    tier=FeatureTier.DELAYED,
    source="coinglass",
    formula="long_accounts / short_accounts",
    inputs=(_REGIME,),
    units="ratio",
    purpose="Crowd positioning. Useful mainly at extremes, and mainly as a "
    "contrarian conditioner.",
    reproducibility=_REST,
    group="regime",
    freshness_budget_ms=_HOUR,
)
def rg_long_short_ratio(context: FeatureContext) -> float | None:
    return _regime_value(context, "long_short_ratio")


@feature(
    "rg_liquidations_5m_usd",
    tier=FeatureTier.DELAYED,
    source="coinglass",
    formula="sum(liquidation notional over the last 5 minutes)",
    inputs=(_REGIME,),
    units="USD",
    purpose="Forced flow. A liquidation cascade produces exactly the sharp, "
    "one-directional moves that decide these markets.",
    reproducibility=_REST,
    group="regime",
    freshness_budget_ms=300_000,
)
def rg_liquidations_5m_usd(context: FeatureContext) -> float | None:
    return _regime_value(context, "liquidations_5m_usd")


@feature(
    "rg_fear_greed",
    tier=FeatureTier.DELAYED,
    source="fear_greed",
    formula="Fear & Greed index, 0-100",
    inputs=(_REGIME,),
    units="index",
    purpose="Daily sentiment. Included for completeness and expected to be pruned: "
    "a value that changes once a day cannot inform a 300-second forecast.",
    reproducibility=_REST,
    group="regime",
    freshness_budget_ms=48 * _HOUR,
)
def rg_fear_greed(context: FeatureContext) -> float | None:
    return _regime_value(context, "fear_greed")


@feature(
    "rg_macro_event_proximity",
    tier=FeatureTier.DELAYED,
    source="macro_calendar",
    formula="1 / (1 + minutes_to_nearest_high_impact_event)",
    inputs=(_REGIME,),
    units="ratio in (0, 1]",
    purpose="Nearness to a scheduled macro release. Approaching 1 means a "
    "volatility jump is scheduled, and any volatility estimate from the recent "
    "past is about to be wrong.",
    reproducibility=_REST,
    group="regime",
    freshness_budget_ms=_HOUR,
)
def rg_macro_event_proximity(context: FeatureContext) -> float | None:
    minutes = context.regime.get("minutes_to_macro_event")
    if minutes is None or minutes < 0:
        return None
    return 1.0 / (1.0 + float(minutes))


REGIME_FEATURES = [
    rg_funding_rate,
    rg_open_interest_change_1h,
    rg_long_short_ratio,
    rg_liquidations_5m_usd,
    rg_fear_greed,
    rg_macro_event_proximity,
]
