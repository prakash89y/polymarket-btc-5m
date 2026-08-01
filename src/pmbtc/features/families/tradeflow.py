"""Trade-flow features.

What actually *executed*, as opposed to what was merely offered. Resting orders
can be cancelled; a print cannot be taken back. On short horizons signed
executed flow is the more reliable of the two, and the disagreement between them
is itself informative.

Every window is bounded above by the snapshot instant, so flow measured "over
the last 30 seconds" never includes a trade that arrived afterwards.
"""

from __future__ import annotations

from pmbtc.features.base import RAW, FeatureContext, feature, safe_div
from pmbtc.features.registry import FeatureTier

_PM = "polymarket_clob"
_REF = "binance_spot"
_PM_TAPE = f"{RAW}pm_trades"
_REF_TAPE = f"{RAW}ref_trades"


def _cvd(context: FeatureContext, seconds: int) -> float | None:
    tape = context.pm_tape
    return tape.cvd(seconds * 1000, context.now_ms) if tape else None


@feature(
    "tf_cvd_10s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="sum(size * sign) over trades in (t-10s, t]",
    inputs=(_PM_TAPE,),
    units="shares",
    purpose="Very short-horizon executed pressure on the Up token.",
    freshness_budget_ms=10_000,
)
def tf_cvd_10s(context: FeatureContext) -> float | None:
    return _cvd(context, 10)


@feature(
    "tf_cvd_60s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="sum(size * sign) over trades in (t-60s, t]",
    inputs=(_PM_TAPE,),
    units="shares",
    purpose="Minute-scale executed pressure; less noisy than the 10s version.",
    freshness_budget_ms=60_000,
)
def tf_cvd_60s(context: FeatureContext) -> float | None:
    return _cvd(context, 60)


@feature(
    "tf_cvd_ratio_60s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="cvd_60s / total_volume_60s",
    inputs=(_PM_TAPE,),
    units="ratio in [-1, 1]",
    purpose="Scale-free flow imbalance, comparable between quiet and busy windows "
    "in a way the raw CVD is not.",
    freshness_budget_ms=60_000,
)
def tf_cvd_ratio_60s(context: FeatureContext) -> float | None:
    tape = context.pm_tape
    return tape.volume_delta_ratio(60_000, context.now_ms) if tape else None


@feature(
    "tf_cvd_acceleration",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="cvd_10s - (cvd_60s / 6)",
    inputs=("tf_cvd_10s", "tf_cvd_60s"),
    units="shares",
    purpose="Is recent flow faster than the minute average? Positive means pressure "
    "is building rather than decaying.",
    freshness_budget_ms=10_000,
)
def tf_cvd_acceleration(context: FeatureContext) -> float | None:
    fast, slow = context.value("tf_cvd_10s"), context.value("tf_cvd_60s")
    if fast is None or slow is None:
        return None
    return fast - slow / 6.0


@feature(
    "tf_trade_intensity_60s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="count(trades in (t-60s, t]) / 60",
    inputs=(_PM_TAPE,),
    units="trades/second",
    purpose="Activity level. Near-zero intensity means any other flow feature is "
    "built on a handful of prints and should be distrusted.",
    freshness_budget_ms=60_000,
)
def tf_trade_intensity_60s(context: FeatureContext) -> float | None:
    tape = context.pm_tape
    return tape.trade_intensity(60_000, context.now_ms) if tape else None


@feature(
    "tf_ref_cvd_60s",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="sum(size * sign) over BTC trades in (t-60s, t]",
    inputs=(_REF_TAPE,),
    units="BTC",
    purpose="Executed pressure on the underlying, which is what actually moves the "
    "settlement price.",
    freshness_budget_ms=60_000,
)
def tf_ref_cvd_60s(context: FeatureContext) -> float | None:
    tape = context.ref_tape
    return tape.cvd(60_000, context.now_ms) if tape else None


@feature(
    "tf_ref_cvd_ratio_60s",
    tier=FeatureTier.REAL_TIME,
    source=_REF,
    formula="ref_cvd_60s / ref_volume_60s",
    inputs=(_REF_TAPE,),
    units="ratio in [-1, 1]",
    purpose="Scale-free underlying flow imbalance.",
    freshness_budget_ms=60_000,
)
def tf_ref_cvd_ratio_60s(context: FeatureContext) -> float | None:
    tape = context.ref_tape
    return tape.volume_delta_ratio(60_000, context.now_ms) if tape else None


@feature(
    "tf_flow_divergence",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="cvd_ratio_60s - ref_cvd_ratio_60s",
    inputs=("tf_cvd_ratio_60s", "tf_ref_cvd_ratio_60s"),
    units="ratio",
    purpose="Prediction-market flow against underlying flow. When the book is being "
    "bought while BTC is being sold, one of them is wrong — and historically it is "
    "more often the thinner market.",
    freshness_budget_ms=60_000,
)
def tf_flow_divergence(context: FeatureContext) -> float | None:
    pm, ref = context.value("tf_cvd_ratio_60s"), context.value("tf_ref_cvd_ratio_60s")
    return pm - ref if pm is not None and ref is not None else None


@feature(
    "tf_book_flow_agreement",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="sign(imbalance_5) * sign(cvd_ratio_60s)",
    inputs=("ob_imbalance_5", "tf_cvd_ratio_60s"),
    units="sign in {-1, 0, 1}",
    purpose="Do resting intent and executed flow point the same way? Agreement is a "
    "stronger signal than either alone; disagreement often marks absorption.",
    freshness_budget_ms=60_000,
)
def tf_book_flow_agreement(context: FeatureContext) -> float | None:
    book = context.value("ob_imbalance_5")
    flow = context.value("tf_cvd_ratio_60s")
    if book is None or flow is None:
        return None
    return float((book > 0) - (book < 0)) * float((flow > 0) - (flow < 0))


@feature(
    "tf_avg_trade_size_60s",
    tier=FeatureTier.REAL_TIME,
    source=_PM,
    formula="volume_60s / count_60s",
    inputs=(_PM_TAPE,),
    units="shares",
    purpose="Retail flow arrives in small clips; informed flow tends to be larger. "
    "A jump in average size is worth noticing.",
    freshness_budget_ms=60_000,
)
def tf_avg_trade_size_60s(context: FeatureContext) -> float | None:
    tape = context.pm_tape
    if tape is None:
        return None
    return safe_div(
        tape.volume(60_000, context.now_ms), float(tape.trade_count(60_000, context.now_ms))
    )


TRADEFLOW_FEATURES = [
    tf_cvd_10s,
    tf_cvd_60s,
    tf_cvd_ratio_60s,
    tf_cvd_acceleration,
    tf_trade_intensity_60s,
    tf_ref_cvd_60s,
    tf_ref_cvd_ratio_60s,
    tf_flow_divergence,
    tf_book_flow_agreement,
    tf_avg_trade_size_60s,
]
