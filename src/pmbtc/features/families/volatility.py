"""Volatility features.

The single most important family for this instrument. A 5-minute Up/Down market
is, to first order, a question about one number: how far has the price moved
from the open, *relative to how far it typically moves in the time remaining*.

That ratio is `vol_distance_in_sigma`, and it is the closest thing this system
has to a closed-form fair value. Everything else in the model is a correction
to it.
"""

from __future__ import annotations

from math import erf, sqrt

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_REF = "binance_spot"
_REF_TAPE = f"{RAW}ref_trades"
_OPEN = f"{RAW}window_open_price"


#: Minimum trades in a window before a volatility estimate is trusted.
#: A freshly-connected tape has a handful of prints, whose dispersion is far
#: below the true volatility. That tiny denominator makes `distance_in_sigma`
#: enormous and pins the implied fair value at 0 or 1 — a maximally confident
#: prediction built on no data. Observed live at -85 sigma before this guard.
MIN_VOL_TRADES = 20

#: The tape must also *span* enough of the window. Twenty trades in two seconds
#: measure two seconds of volatility, not five minutes of it.
MIN_VOL_SPAN_FRACTION = 0.5


def _vol_bps(context: FeatureContext, seconds: int) -> float | None:
    """Realised volatility, or None when the tape cannot support an estimate."""
    tape = context.ref_tape
    if tape is None:
        return None
    window_ms = seconds * 1000
    trades = tape.window(window_ms, context.now_ms)
    if len(trades) < MIN_VOL_TRADES:
        return None
    span = trades[-1].timestamp_ms - trades[0].timestamp_ms
    if span < window_ms * MIN_VOL_SPAN_FRACTION:
        return None
    return tape.realized_vol(window_ms, context.now_ms)


@feature(
    "vol_realized_30s_bps",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="stdev(trade-to-trade returns over 30s) * 10000",
    inputs=(_REF_TAPE,),
    units="basis points",
    purpose="Very recent volatility. Reacts fastest to a regime change.",
    freshness_budget_ms=30_000,
)
def vol_realized_30s_bps(context: FeatureContext) -> float | None:
    return _vol_bps(context, 30)


@feature(
    "vol_realized_300s_bps",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="stdev(trade-to-trade returns over 300s) * 10000",
    inputs=(_REF_TAPE,),
    units="basis points",
    purpose="Window-scale volatility — the natural denominator for displacement "
    "over a 5-minute market.",
    freshness_budget_ms=300_000,
)
def vol_realized_300s_bps(context: FeatureContext) -> float | None:
    return _vol_bps(context, 300)


@feature(
    "vol_ratio_fast_slow",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="realized_30s / realized_300s",
    inputs=("vol_realized_30s_bps", "vol_realized_300s_bps"),
    units="ratio",
    purpose="Volatility regime shift. Above 1 means the market just woke up, and "
    "a fair value computed from the slower estimate is understating uncertainty.",
    freshness_budget_ms=30_000,
)
def vol_ratio_fast_slow(context: FeatureContext) -> float | None:
    return safe_div(
        context.value("vol_realized_30s_bps"), context.value("vol_realized_300s_bps")
    )


@feature(
    "vol_price_change_bps",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="(price_now - price_at_window_open) / price_at_window_open * 10000",
    inputs=(_REF_TAPE, _OPEN),
    units="basis points",
    purpose="Displacement from the open. The raw quantity the market resolves on.",
)
def vol_price_change_bps(context: FeatureContext) -> float | None:
    tape = context.ref_tape
    price = tape.last_price if tape else None
    open_price = context.window_open_price
    if price is None or not open_price:
        return None
    return ((price - open_price) / open_price) * 10_000


@feature(
    "vol_time_scaled_sigma_bps",
    tier=FeatureTier.DERIVED,
    source=_REF,
    formula="realized_300s_bps * sqrt(seconds_remaining / 300)",
    inputs=("vol_realized_300s_bps", f"{RAW}clock"),
    units="basis points",
    purpose="How far the price can still plausibly travel before settlement. "
    "Volatility scales with the square root of time, so this shrinks as the "
    "window closes — which is why late displacement is so much more decisive.",
)
def vol_time_scaled_sigma_bps(context: FeatureContext) -> float | None:
    sigma = context.value("vol_realized_300s_bps")
    if sigma is None:
        return None
    remaining = max(0.0, context.seconds_to_settlement)
    return sigma * sqrt(remaining / 300.0)


@feature(
    "vol_distance_in_sigma",
    tier=FeatureTier.DERIVED,
    source=_REF,
    formula="price_change_bps / time_scaled_sigma_bps",
    inputs=("vol_price_change_bps", "vol_time_scaled_sigma_bps"),
    units="standard deviations",
    purpose="THE fair-value input. Displacement measured in units of the movement "
    "still available. Large magnitude means the outcome is close to decided.",
)
def vol_distance_in_sigma(context: FeatureContext) -> float | None:
    return safe_div(
        context.value("vol_price_change_bps"), context.value("vol_time_scaled_sigma_bps")
    )


@feature(
    "vol_implied_fair_up",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="Phi(distance_in_sigma), the standard normal CDF",
    inputs=("vol_distance_in_sigma",),
    units="probability",
    purpose="Closed-form P(Up) under a driftless Gaussian random walk. Not a "
    "trading signal on its own — it is the null hypothesis the model must beat, "
    "and the anchor its predictions are compared against.",
)
def vol_implied_fair_up(context: FeatureContext) -> float | None:
    distance = context.value("vol_distance_in_sigma")
    if distance is None:
        return None
    # Standard normal CDF via the error function; no scipy dependency, and
    # deterministic across platforms.
    return 0.5 * (1.0 + erf(distance / sqrt(2.0)))


@feature(
    "vol_model_market_gap",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="implied_fair_up - ob_mid",
    inputs=("vol_implied_fair_up", "ob_mid"),
    units="probability",
    purpose="Disagreement between the random-walk fair value and the book. The "
    "first place to look for edge, and equally the first place to look for a bug.",
)
def vol_model_market_gap(context: FeatureContext) -> float | None:
    fair, mid = context.value("vol_implied_fair_up"), context.value("ob_mid")
    return fair - mid if fair is not None and mid is not None else None


VOLATILITY_FEATURES = [
    vol_realized_30s_bps,
    vol_realized_300s_bps,
    vol_ratio_fast_slow,
    vol_price_change_bps,
    vol_time_scaled_sigma_bps,
    vol_distance_in_sigma,
    vol_implied_fair_up,
    vol_model_market_gap,
]
