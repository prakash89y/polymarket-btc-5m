"""The promotion gate.

Every condition must pass. There is no override parameter, no force flag, and no
"promote anyway" path — deliberately, because every such escape hatch is
eventually used at 2 a.m. by someone convinced this case is different. If a
model should be promoted and the gate says no, the correct fix is to change the
thresholds in config, in a reviewable commit, not to bypass the check.

The conditions:

1. Beats **every** benchmark baseline on the untouched holdout, by the
   configured margin — including the market-probability baseline, which is the
   one that actually matters.
2. The improvement over the strongest baseline is **statistically significant**,
   not merely positive.
3. Calibration is good in absolute terms (ECE below the configured limit).
4. Shadow validation passed: deterministic, reproducible, stable, fast enough.
5. If a champion exists, the challenger beats it significantly too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.models.baselines import BaselineReport, ScoreCard
from pmbtc.models.calibration import EvaluationMetrics
from pmbtc.models.shadow import ShadowResult
from pmbtc.models.statistics import TestResult, paired_bootstrap

log = get_logger("pmbtc.models.promotion")


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class PromotionDecision:
    checks: list[GateCheck] = field(default_factory=list)
    tests: list[TestResult] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def reasons(self) -> list[str]:
        return [str(check) for check in self.checks]

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if self.approved:
            return f"PROMOTE: all {len(self.checks)} gate conditions satisfied"
        return "BLOCKED: " + "; ".join(f.detail for f in self.failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
            "tests": [t.as_dict() for t in self.tests],
        }


def evaluate_promotion(
    config: Config,
    *,
    candidate: EvaluationMetrics,
    candidate_probs: np.ndarray,
    y_holdout: np.ndarray,
    baselines: BaselineReport,
    baseline_probs: dict[str, np.ndarray],
    shadow: ShadowResult,
    champion_probs: np.ndarray | None = None,
) -> PromotionDecision:
    """Run every gate condition. No path through this returns early on success."""
    training = config.training
    decision = PromotionDecision()
    margin = training.min_baseline_improvement

    candidate_card = ScoreCard(
        name="candidate",
        samples=candidate.samples,
        accuracy=candidate.accuracy,
        brier=candidate.brier,
        log_loss=candidate.log_loss,
        mean_prediction=candidate.mean_prediction,
    )

    # 1. Beat every baseline by the configured margin.
    blocking = baselines.blocking(candidate_card, margin)
    decision.checks.append(
        GateCheck(
            "beats_all_baselines",
            not blocking,
            (
                f"candidate brier {candidate.brier:.5f} beats all "
                f"{len(baselines.scores)} baselines by >= {margin}"
                if not blocking
                else "did not beat: "
                + ", ".join(f"{b.name}({b.brier:.5f})" for b in blocking)
            ),
        )
    )

    # 2. Significance against the strongest baseline.
    strongest = baselines.best
    if strongest is not None and strongest.name in baseline_probs:
        test = paired_bootstrap(
            y_holdout, candidate_probs, baseline_probs[strongest.name], metric="brier"
        )
        decision.tests.append(test)
        decision.checks.append(
            GateCheck(
                "significant_vs_best_baseline",
                test.significant(training.promotion_significance),
                f"vs {strongest.name}: {test}",
            )
        )
    else:
        decision.checks.append(
            GateCheck(
                "significant_vs_best_baseline",
                False,
                "no baseline predictions available to test against",
            )
        )

    # 3. Absolute calibration.
    decision.checks.append(
        GateCheck(
            "calibration",
            candidate.ece <= training.max_holdout_calibration_error,
            f"holdout ECE {candidate.ece:.5f} vs limit "
            f"{training.max_holdout_calibration_error}",
        )
    )

    # 4. Shadow validation.
    decision.checks.append(
        GateCheck("shadow_validation", shadow.passed, shadow.summary())
    )

    # 5. Champion comparison, when there is one.
    if champion_probs is not None:
        test = paired_bootstrap(y_holdout, candidate_probs, champion_probs, metric="brier")
        decision.tests.append(test)
        improvement = test.effect
        decision.checks.append(
            GateCheck(
                "beats_champion",
                (
                    improvement >= training.min_brier_improvement
                    and test.significant(training.promotion_significance)
                ),
                f"brier improvement {improvement:+.5f} "
                f"(need {training.min_brier_improvement}), p={test.p_value:.4f}",
            )
        )
    else:
        decision.checks.append(
            GateCheck("beats_champion", True, "no incumbent champion; baselines govern")
        )

    log.info(
        "promotion.evaluated",
        approved=decision.approved,
        failures=[c.name for c in decision.failures],
    )
    return decision
