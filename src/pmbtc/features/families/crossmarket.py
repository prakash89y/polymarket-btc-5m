"""Cross-market relationship features.

The prediction market and the underlying are two views of the same question. The
interesting quantity is not either view but the *gap* between them: the
prediction market must ultimately agree with the underlying, so a divergence is
either an opportunity or a warning that one feed is wrong.

Because these features are built from two sources, they carry a consistency
annotation (see :mod:`pmbtc.features.consistency`) rather than silently trusting
whichever source happens to be fresher.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_REF_TAPE = f"{RAW}ref_trades"


@feature(
    "xm_ref_pm_lead_bps",
    tier=FeatureTier.REAL_TIME,
    source="derived",
    formula="price_change_bps - (mid - 0.5) * 2 * time_scaled_sigma_bps",
    inputs=("vol_price_change_bps", "ob_mid", "vol_time_scaled_sigma_bps"),
    units="basis points",
    purpose="Underlying displacement minus the displacement the book's probability "
    "implies. Positive means BTC has moved further than the prediction market has "
    "priced — the book is lagging.",
)
def xm_ref_pm_lead_bps(context: FeatureContext) -> float | None:
    change = context.value("vol_price_change_bps")
    mid = context.value("ob_mid")
    sigma = context.value("vol_time_scaled_sigma_bps")
    if change is None or mid is None or sigma is None:
        return None
    implied = (mid - 0.5) * 2.0 * sigma
    return change - implied


@feature(
    "xm_ref_spread_bps",
    tier=FeatureTier.REAL_TIME,
    source="binance_spot",
    formula="(ref_ask - ref_bid) / ref_mid * 10000",
    inputs=(f"{RAW}ref_quote",),
    units="basis points",
    purpose="Liquidity of the underlying. A widening reference spread usually "
    "precedes a volatility burst.",
)
def xm_ref_spread_bps(context: FeatureContext) -> float | None:
    bid, ask = context.ref_bid, context.ref_ask
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 10_000 if mid > 0 else None


@feature(
    "xm_vwap_deviation_bps",
    tier=FeatureTier.REAL_TIME,
    source="binance_spot",
    formula="(price - vwap_60s) / vwap_60s * 10000",
    inputs=(_REF_TAPE,),
    units="basis points",
    purpose="Distance from the 60-second volume-weighted average. Mean reversion "
    "toward VWAP is one of the more durable short-horizon regularities.",
    freshness_budget_ms=60_000,
)
def xm_vwap_deviation_bps(context: FeatureContext) -> float | None:
    tape = context.ref_tape
    if tape is None:
        return None
    vwap = tape.vwap(60_000, context.now_ms)
    price = tape.last_price
    if vwap is None or price is None or vwap <= 0:
        return None
    return ((price - vwap) / vwap) * 10_000


@feature(
    "xm_flow_price_elasticity",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="price_change_bps / ref_cvd_60s",
    inputs=("vol_price_change_bps", "tf_ref_cvd_60s"),
    units="bps per BTC",
    purpose="How much the price moved per unit of signed flow. Low elasticity means "
    "the market absorbed the flow — a sign of a strong resting bid or offer.",
    freshness_budget_ms=60_000,
)
def xm_flow_price_elasticity(context: FeatureContext) -> float | None:
    return safe_div(context.value("vol_price_change_bps"), context.value("tf_ref_cvd_60s"))


CROSSMARKET_FEATURES = [
    xm_ref_pm_lead_bps,
    xm_ref_spread_bps,
    xm_vwap_deviation_bps,
    xm_flow_price_elasticity,
]
