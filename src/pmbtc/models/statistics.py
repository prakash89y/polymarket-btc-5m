"""Statistical comparison of models.

The failure this module exists to prevent: promoting a model because its Brier
score was 0.2481 against the champion's 0.2483. On a few hundred markets that
difference is indistinguishable from noise, and a system that promotes on it
will churn models forever while trading no real edge.

Three tests, each for a different question:

``paired_bootstrap``
    Is the *difference* in a metric reliably non-zero? Paired on the sample, so
    it removes the variance from which markets happened to be easy.
``mcnemar``
    Do the models disagree asymmetrically on classification? Uses only the
    cases where exactly one was right, which is the only information about
    which is better.
``binomial_vs_breakeven``
    Is a win rate above break-even by more than chance? Used for the paper
    trading gate rather than model comparison.

All are one-sided where a one-sided question is being asked, and all report the
effect size alongside the p-value — a significant but trivial improvement is
still not worth a deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from pmbtc.models.calibration import brier_score, clip_probabilities, log_loss


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of one statistical comparison."""

    test: str
    statistic: float
    p_value: float
    effect: float
    n: int
    detail: str = ""

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def as_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "effect": round(self.effect, 6),
            "n": self.n,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return (
            f"{self.test}: effect={self.effect:+.5f} p={self.p_value:.4f} "
            f"(n={self.n}){' — ' + self.detail if self.detail else ''}"
        )


def paired_bootstrap(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    metric: str = "brier",
    iterations: int = 2_000,
    seed: int = 42,
    alternative: str = "less",
) -> TestResult:
    """Is model A's metric better than model B's, beyond sampling noise?

    Resamples markets with replacement, recomputing the metric difference each
    time. The p-value is the fraction of resamples in which A failed to beat B,
    which is a direct answer to "would this ordering survive a different sample
    of markets".

    ``alternative='less'`` means lower is better, which is correct for Brier and
    log loss.
    """
    y = np.asarray(y_true, dtype=float)
    a = clip_probabilities(prob_a)
    b = clip_probabilities(prob_b)
    n = len(y)
    if n == 0:
        return TestResult(f"paired_bootstrap[{metric}]", 0.0, 1.0, 0.0, 0, "no samples")

    scorer = brier_score if metric == "brier" else log_loss
    observed = scorer(y, a) - scorer(y, b)

    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(iterations):
        index = rng.integers(0, n, size=n)
        difference = scorer(y[index], a[index]) - scorer(y[index], b[index])
        if (difference < 0) if alternative == "less" else (difference > 0):
            wins += 1
    p_value = 1.0 - wins / iterations

    return TestResult(
        test=f"paired_bootstrap[{metric}]",
        statistic=float(observed),
        p_value=float(p_value),
        effect=float(-observed if alternative == "less" else observed),
        n=n,
        detail=f"A better in {wins}/{iterations} resamples",
    )


def mcnemar(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
) -> TestResult:
    """Exact McNemar test on discordant classification pairs.

    Only cases where exactly one model was correct carry information about which
    is better; the exact binomial form is used because the discordant count is
    usually small enough that the chi-square approximation is poor.
    """
    y = np.asarray(y_true, dtype=bool)
    correct_a = (clip_probabilities(prob_a) >= 0.5) == y
    correct_b = (clip_probabilities(prob_b) >= 0.5) == y

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    if discordant == 0:
        return TestResult("mcnemar", 0.0, 1.0, 0.0, 0, "models never disagree")

    # One-sided exact binomial: P(X >= only_a | p=0.5).
    tail = sum(comb(discordant, k) for k in range(only_a, discordant + 1))
    p_value = tail / (2**discordant)
    return TestResult(
        test="mcnemar",
        statistic=float(only_a - only_b),
        p_value=float(min(1.0, p_value)),
        effect=float((only_a - only_b) / discordant),
        n=discordant,
        detail=f"A-only correct {only_a}, B-only correct {only_b}",
    )


def binomial_vs_breakeven(
    wins: int, total: int, breakeven: float = 0.5
) -> TestResult:
    """Is a win rate above break-even by more than chance?

    Exact one-sided binomial. Used by the paper-trading promotion gate, where
    the question is "did this record beat break-even" rather than "did this
    model beat that one".
    """
    if total == 0:
        return TestResult("binomial", 0.0, 1.0, 0.0, 0, "no trades")
    tail = sum(
        comb(total, k) * (breakeven**k) * ((1.0 - breakeven) ** (total - k))
        for k in range(wins, total + 1)
    )
    rate = wins / total
    return TestResult(
        test="binomial",
        statistic=float(wins),
        p_value=float(min(1.0, tail)),
        effect=float(rate - breakeven),
        n=total,
        detail=f"win rate {rate:.4f} vs break-even {breakeven:.4f}",
    )


def diebold_mariano(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
) -> TestResult:
    """Normal-approximation test on the mean loss differential.

    Complements the bootstrap: it is cheap, and where the two disagree the
    bootstrap is trusted, because it makes no distributional assumption about a
    loss differential that is plainly not normal in the tails.
    """
    y = np.asarray(y_true, dtype=float)
    a, b = clip_probabilities(prob_a), clip_probabilities(prob_b)
    if len(y) < 8:
        return TestResult("diebold_mariano", 0.0, 1.0, 0.0, len(y), "too few samples")

    differential = (a - y) ** 2 - (b - y) ** 2
    mean = float(np.mean(differential))
    stdev = float(np.std(differential, ddof=1))
    if stdev == 0:
        return TestResult("diebold_mariano", 0.0, 1.0, 0.0, len(y), "identical losses")

    statistic = mean / (stdev / np.sqrt(len(y)))
    # One-sided normal CDF via erf, avoiding a scipy dependency.
    from math import erf, sqrt

    p_value = 0.5 * (1.0 + erf(statistic / sqrt(2.0)))
    return TestResult(
        test="diebold_mariano",
        statistic=float(statistic),
        p_value=float(p_value),
        effect=float(-mean),
        n=len(y),
        detail="negative statistic favours A",
    )
