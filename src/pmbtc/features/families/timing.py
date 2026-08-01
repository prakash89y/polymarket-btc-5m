"""Time features.

Exact by construction — the event time *is* the snapshot instant — so these are
the only features with no latency, no staleness, and no possible source
disagreement.

The important one is `tm_sqrt_time_remaining`: uncertainty in a random walk
scales with the square root of time, so this is the correct shape for how much
the outcome can still change, and it is what makes a fixed displacement mean
something different at T-300 than at T-30.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt

from pmbtc.features.base import RAW, FeatureContext, feature
from pmbtc.features.registry import FeatureTier, ReproducibilityPolicy
from pmbtc.utils.timeutils import is_weekend, minute_of_day, session_of

_CLOCK = f"{RAW}clock"
_DERIVED = ReproducibilityPolicy.DERIVED_FROM_ARCHIVE


@feature(
    "tm_seconds_to_settlement",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="(settlement_ms - t) / 1000",
    inputs=(_CLOCK,),
    units="seconds",
    purpose="Time left. Every other time feature is a transform of this one.",
    reproducibility=_DERIVED,
    group="short_term",
    freshness_budget_ms=1_000,
)
def tm_seconds_to_settlement(context: FeatureContext) -> float | None:
    return context.seconds_to_settlement


@feature(
    "tm_window_progress",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="(t - window_open) / (settlement - window_open)",
    inputs=(_CLOCK,),
    units="fraction in [0, 1]",
    purpose="Position within the window, independent of window length.",
    reproducibility=_DERIVED,
    group="short_term",
    freshness_budget_ms=1_000,
)
def tm_window_progress(context: FeatureContext) -> float | None:
    return context.window_progress


@feature(
    "tm_sqrt_time_remaining",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="sqrt(max(0, 1 - window_progress))",
    inputs=("tm_window_progress",),
    units="fraction",
    purpose="The scale of remaining uncertainty. Volatility grows with sqrt(time), "
    "so this — not linear time — is how far the price can still travel.",
    reproducibility=_DERIVED,
    group="short_term",
    freshness_budget_ms=1_000,
)
def tm_sqrt_time_remaining(context: FeatureContext) -> float | None:
    progress = context.value("tm_window_progress")
    if progress is None:
        return None
    return sqrt(max(0.0, 1.0 - progress))


@feature(
    "tm_minute_sin",
    tier=FeatureTier.STATIC,
    source="derived",
    formula="sin(2*pi*minute_of_day / 1440)",
    inputs=(_CLOCK,),
    units="dimensionless",
    purpose="Cyclical time-of-day encoding. Sine and cosine together avoid the "
    "discontinuity at midnight that a raw minute counter would introduce.",
    reproducibility=_DERIVED,
    group="short_term",
    freshness_budget_ms=60_000,
)
def tm_minute_sin(context: FeatureContext) -> float | None:
    return sin(2.0 * pi * minute_of_day(context.now_ms) / 1440.0)


@feature(
    "tm_minute_cos",
    tier=FeatureTier.STATIC,
    source="derived",
    formula="cos(2*pi*minute_of_day / 1440)",
    inputs=(_CLOCK,),
    units="dimensionless",
    purpose="Cyclical time-of-day encoding, paired with the sine term.",
    reproducibility=_DERIVED,
    group="short_term",
    freshness_budget_ms=60_000,
)
def tm_minute_cos(context: FeatureContext) -> float | None:
    return cos(2.0 * pi * minute_of_day(context.now_ms) / 1440.0)


@feature(
    "tm_is_weekend",
    tier=FeatureTier.STATIC,
    source="derived",
    formula="1 if UTC weekday in {Sat, Sun} else 0",
    inputs=(_CLOCK,),
    units="indicator",
    purpose="Crypto trades continuously but weekend liquidity and volatility "
    "regimes differ measurably.",
    reproducibility=_DERIVED,
    group="regime",
    freshness_budget_ms=86_400_000,
)
def tm_is_weekend(context: FeatureContext) -> float | None:
    return 1.0 if is_weekend(context.now_ms) else 0.0


@feature(
    "tm_session_code",
    tier=FeatureTier.STATIC,
    source="derived",
    formula="ordinal of {asia:0, london:1, overlap:2, new_york:3, late_us:4}",
    inputs=(_CLOCK,),
    units="ordinal",
    purpose="Trading session. Kept ordinal rather than one-hot so tree models can "
    "split on it directly; linear models should one-hot it downstream.",
    reproducibility=_DERIVED,
    group="short_term",
    freshness_budget_ms=3_600_000,
)
def tm_session_code(context: FeatureContext) -> float | None:
    order = {"asia": 0.0, "london": 1.0, "london_ny_overlap": 2.0,
             "new_york": 3.0, "late_us": 4.0}
    return order.get(session_of(context.now_ms).value)


TIME_FEATURES = [
    tm_seconds_to_settlement,
    tm_window_progress,
    tm_sqrt_time_remaining,
    tm_minute_sin,
    tm_minute_cos,
    tm_is_weekend,
    tm_session_code,
]
