"""Market edge validation — Module 8.5.

Every threshold in :mod:`pmbtc.trading.decision` before this module was a
*floor*: a minimum confidence, a minimum edge, a minimum expected value. That
arrangement has a failure mode which the statistical review found in live data,
and it is not a subtle one.

Measured at T-30 on the collected markets, the deployed logistic baseline
produced a **median claimed edge of 0.83 probability points** and a **median
expected value of +680% per five-minute trade**, with 84% of its intents taken
against books quoted at 0.80 or beyond. One trade bought DOWN at 0.035 against a
market pricing UP at 0.975, on a model probability of 0.24.

None of those numbers are edge. They are the signature of a broken model, and
floors cannot catch them, because a model's apparent edge *grows* with its
error. The worse the forecast, the larger the disagreement, the more eagerly the
system trades, and the more reliably it is on the wrong side of a book that was
right. This module supplies the missing ceilings.

**Why log-odds.** Disagreement is measured as ``|logit(model) - logit(market)|``
rather than ``|model - market|``. Calling a 0.50 market at 0.75 is an ordinary
opinion; calling a 0.97 market at 0.72 is an extraordinary one. In probability
space both are 0.25 and indistinguishable. In log-odds they are 1.10 and 2.54,
which is the distinction that matters, and it falls out of the arithmetic rather
than needing a special case for confident books.

**Why the market is the shrinkage target.** The review tested every implied
probability bucket at every horizon and found none mispriced (Bonferroni
p >= 0.25 throughout); the book's observed frequencies tracked its own prices
across the full range. When the prior is "the book is right", the correct
shrinkage target for an untrusted model is the market price, not 0.5. Shrinking
to 0.5 would manufacture edge against confident markets — the precise error
being corrected.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pmbtc.constants import SkipReason
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.trading.validation")

#: Probabilities are clamped before any log-odds transform. logit(0) and
#: logit(1) are infinite, and an infinite disagreement is not a useful number
#: to reason about or to log.
_EPS = 1e-6


def clamp01(p: float, eps: float = _EPS) -> float:
    return min(1.0 - eps, max(eps, p))


def logit(p: float) -> float:
    """Log-odds. The scale on which probability disagreements are comparable."""
    q = clamp01(p)
    return math.log(q / (1.0 - q))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def disagreement_logits(model_prob: float, market_prob: float) -> float:
    """How far the model is from the book, in log-odds. Always non-negative."""
    return abs(logit(model_prob) - logit(market_prob))


def blend_toward_market(model_prob: float, market_prob: float, trust: float) -> float:
    """Shrink the model toward the market prior.

    ``trust = 1`` returns the model unchanged; ``trust = 0`` returns the market,
    which produces exactly zero edge and therefore no trade. Blending happens in
    log-odds so that the result is a proper probabilistic compromise rather than
    an arithmetic average that behaves oddly near 0 and 1.
    """
    if trust >= 1.0:
        return model_prob
    if trust <= 0.0:
        return market_prob
    return inv_logit(trust * logit(model_prob) + (1.0 - trust) * logit(market_prob))


def implausible(prob: float, floor: float) -> bool:
    """Is this probability more certain than any honest forecast could be?"""
    return prob < floor or prob > 1.0 - floor


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


#: Scale factor making MAD a consistent estimator of sigma for normal data.
MAD_TO_SIGMA = 1.4826


@dataclass
class DisagreementMonitor:
    """Robust outlier detection over a model's own recent disagreements.

    Median and MAD rather than mean and standard deviation, because the thing
    being detected — a handful of enormous disagreements — is exactly what
    corrupts a mean and inflates a standard deviation until the outlier looks
    unremarkable. A single 0.83-logit excursion moves the median almost not at
    all, which is what makes it visible against the median.

    This complements the fixed bound rather than replacing it: a fixed ceiling
    catches a model that was always broken, and this catches one that breaks
    after deployment, relative to how it normally behaves.
    """

    capacity: int = 200
    min_observations: int = 30
    observations: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.observations.append(value)
        if len(self.observations) > self.capacity:
            del self.observations[0]

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.observe(value)

    @property
    def ready(self) -> bool:
        return len(self.observations) >= self.min_observations

    def z_score(self, value: float) -> float | None:
        """Robust z-score, or ``None`` while there is too little history.

        Returns ``None`` rather than 0.0 when not ready: "no opinion" and "this
        is perfectly normal" must not be the same answer, or a cold start would
        silently wave everything through as unremarkable.
        """
        if not self.ready:
            return None
        med = _median(self.observations)
        mad = _median([abs(x - med) for x in self.observations])
        if mad <= 0.0:
            # Degenerate spread: everything seen so far is identical. Only an
            # exact match is unremarkable.
            return 0.0 if value == med else math.inf
        return (value - med) / (mad * MAD_TO_SIGMA)

    def summary(self) -> dict[str, Any]:
        if not self.observations:
            return {"n": 0}
        med = _median(self.observations)
        mad = _median([abs(x - med) for x in self.observations])
        return {
            "n": len(self.observations),
            "median": round(med, 6),
            "mad": round(mad, 6),
            "sigma_est": round(mad * MAD_TO_SIGMA, 6),
            "min": round(min(self.observations), 6),
            "max": round(max(self.observations), 6),
        }


# --------------------------------------------------------------------------- #
# The validator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EdgeValidation:
    """Verdict on one model claim, plus the numbers behind it."""

    accepted: bool
    skip_reason: SkipReason | None = None
    detail: str = ""
    disagreement: float = 0.0
    #: Probability after shrinkage toward the market prior.
    adjusted_prob: float = 0.5
    z_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
            "detail": self.detail,
            "disagreement_logits": round(self.disagreement, 6),
            "adjusted_prob": round(self.adjusted_prob, 6),
            "z_score": None if self.z_score is None else round(self.z_score, 4),
        }


class MarketEdgeValidator:
    """Applies the three ceilings and produces the calibration-adjusted view.

    Stateful only in the anomaly monitor, and that state is fed exclusively by
    observations the caller passes in — so a backtest replaying the same rows in
    the same order gets the same verdicts, every time.
    """

    def __init__(
        self,
        *,
        max_disagreement_logits: float,
        min_plausible_prob: float,
        max_disagreement_z: float,
        model_trust: float = 1.0,
        monitor: DisagreementMonitor | None = None,
    ) -> None:
        self.max_disagreement_logits = max_disagreement_logits
        self.min_plausible_prob = min_plausible_prob
        self.max_disagreement_z = max_disagreement_z
        self.model_trust = model_trust
        self.monitor = monitor or DisagreementMonitor()

    @classmethod
    def from_config(cls, config: Any) -> MarketEdgeValidator:
        p = config.prediction
        return cls(
            max_disagreement_logits=p.max_disagreement_logits,
            min_plausible_prob=p.min_plausible_prob,
            max_disagreement_z=p.max_disagreement_z,
            model_trust=p.model_trust,
            monitor=DisagreementMonitor(
                capacity=p.disagreement_history,
                min_observations=p.min_disagreement_history,
            ),
        )

    # ------------------------------------------------------------------ #
    def validate(self, model_prob: float, market_prob: float) -> EdgeValidation:
        """Check one claim. Records the disagreement whether or not it passes.

        The observation is recorded even for rejected claims deliberately: the
        monitor's job is to describe how this model behaves, and excluding its
        worst moments would make the baseline it compares against useless.
        """
        divergence = disagreement_logits(model_prob, market_prob)
        z = self.monitor.z_score(divergence)
        self.monitor.observe(divergence)
        adjusted = blend_toward_market(model_prob, market_prob, self.model_trust)

        def reject(reason: SkipReason, detail: str) -> EdgeValidation:
            return EdgeValidation(False, reason, detail, divergence, adjusted, z)

        if implausible(model_prob, self.min_plausible_prob):
            return reject(
                SkipReason.IMPLAUSIBLE_PROBABILITY,
                f"model probability {model_prob:.5f} is outside the plausible band "
                f"[{self.min_plausible_prob:.3f}, {1 - self.min_plausible_prob:.3f}]",
            )
        if divergence > self.max_disagreement_logits:
            return reject(
                SkipReason.EXCESSIVE_DISAGREEMENT,
                f"model {model_prob:.4f} vs market {market_prob:.4f} is "
                f"{divergence:.2f} logits apart, ceiling {self.max_disagreement_logits:.2f}",
            )
        if z is not None and z > self.max_disagreement_z:
            return reject(
                SkipReason.ANOMALOUS_EDGE,
                f"disagreement {divergence:.2f} logits is {z:.1f} robust sigma above "
                f"this model's recent median, ceiling {self.max_disagreement_z:.1f}",
            )
        return EdgeValidation(True, None, "within bounds", divergence, adjusted, z)
