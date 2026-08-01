"""Volume features.

Traded size, separate from its direction. Volume answers "how much conviction is
behind this move", and low volume is the single most common reason a
directional signal fails to persist.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_PM = "polymarket_clob"
_REF = "binance_spot"
_PM_TAPE = f"{RAW}pm_trades"
_REF_TAPE = f"{RAW}ref_trades"


@feature(
    "vm_pm_volume_60s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="sum(size) over Up-token trades in (t-60s, t]",
    inputs=(_PM_TAPE,),
    units="shares",
    purpose="Prediction-market activity. Near zero means the book's price is an "
    "opinion rather than a consensus.",
    freshness_budget_ms=60_000,
)
def vm_pm_volume_60s(context: FeatureContext) -> float | None:
    tape = context.pm_tape
    return tape.volume(60_000, context.now_ms) if tape else None


@feature(
    "vm_pm_notional_60s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="sum(price * size) over Up-token trades in (t-60s, t]",
    inputs=(_PM_TAPE,),
    units="USDC",
    purpose="Capital traded rather than share count — the honest measure when "
    "prices range from 0.01 to 0.99.",
    freshness_budget_ms=60_000,
)
def vm_pm_notional_60s(context: FeatureContext) -> float | None:
    tape = context.pm_tape
    return tape.notional(60_000, context.now_ms) if tape else None


@feature(
    "vm_ref_volume_60s",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="sum(size) over BTC trades in (t-60s, t]",
    inputs=(_REF_TAPE,),
    units="BTC",
    purpose="Underlying activity. Drives the volatility that decides the outcome.",
    freshness_budget_ms=60_000,
)
def vm_ref_volume_60s(context: FeatureContext) -> float | None:
    tape = context.ref_tape
    return tape.volume(60_000, context.now_ms) if tape else None


@feature(
    "vm_volume_surge",
    tier=FeatureTier.DERIVED,
    source=_REF,
    formula="volume_60s / (volume_300s / 5)",
    inputs=(_REF_TAPE,),
    units="ratio",
    purpose="Recent volume against its own five-minute baseline. Above 1 means "
    "activity is accelerating, which usually precedes a volatility expansion.",
    freshness_budget_ms=60_000,
)
def vm_volume_surge(context: FeatureContext) -> float | None:
    tape = context.ref_tape
    if tape is None:
        return None
    recent = tape.volume(60_000, context.now_ms)
    baseline = tape.volume(300_000, context.now_ms)
    return safe_div(recent, baseline / 5.0) if baseline else None


@feature(
    "vm_pm_ref_volume_ratio",
    tier=FeatureTier.DERIVED,
    source="derived",
    formula="pm_notional_60s / ref_volume_60s",
    inputs=("vm_pm_notional_60s", "vm_ref_volume_60s"),
    units="USDC per BTC",
    purpose="Relative engagement between the two markets. A spike means the "
    "prediction market is reacting to something the underlying is not.",
    freshness_budget_ms=60_000,
)
def vm_pm_ref_volume_ratio(context: FeatureContext) -> float | None:
    return safe_div(context.value("vm_pm_notional_60s"), context.value("vm_ref_volume_60s"))


VOLUME_FEATURES = [
    vm_pm_volume_60s,
    vm_pm_notional_60s,
    vm_ref_volume_60s,
    vm_volume_surge,
    vm_pm_ref_volume_ratio,
]
