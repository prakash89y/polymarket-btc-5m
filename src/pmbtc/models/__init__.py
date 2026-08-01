"""Modelling: readiness gates and benchmark baselines.

Module 7 (training) is deliberately gated behind both:

* :func:`~pmbtc.models.readiness.check_readiness` refuses to train until the
  dataset is large, complete, balanced, and clean enough for a result to mean
  anything.
* :mod:`~pmbtc.models.baselines` defines the bar every model must clear on
  unseen data before it may be promoted -- above all, the market's own forecast.
"""

from __future__ import annotations

from pmbtc.models.baselines import (
    Baseline,
    BaselineReport,
    GradientBoostingBaseline,
    LogisticBaseline,
    MarketFavourite,
    MarketUnderdog,
    RandomPredictor,
    ScoreCard,
    default_baselines,
    run_baselines,
    score,
)
from pmbtc.models.calibration import (
    CalibrationCurve,
    Calibrator,
    EvaluationMetrics,
    brier_score,
    calibration_curve,
    evaluate,
    log_loss,
)
from pmbtc.models.explain import Explanation, explain_model
from pmbtc.models.promotion import GateCheck, PromotionDecision, evaluate_promotion
from pmbtc.models.readiness import (
    ReadinessCheck,
    ReadinessReport,
    check_readiness,
    complete_timeline_count,
    readiness_for_store,
)
from pmbtc.models.registry import ModelCard, ModelRegistry, make_model_id, stable_hash
from pmbtc.models.search import (
    DEFAULT_SPACES,
    HyperparameterSearch,
    SearchMethod,
    SearchResult,
)
from pmbtc.models.shadow import ShadowResult, run_shadow_validation
from pmbtc.models.statistics import (
    TestResult,
    binomial_vs_breakeven,
    mcnemar,
    paired_bootstrap,
)
from pmbtc.models.train import ModelTrainer, TrainingData, TrainingResult, write_report
from pmbtc.models.validation import (
    Fold,
    SplitScheme,
    TimeSeriesSplitter,
    assert_no_leakage_between,
)

__all__ = [
    "DEFAULT_SPACES",
    "Baseline",
    "BaselineReport",
    "CalibrationCurve",
    "Calibrator",
    "EvaluationMetrics",
    "Explanation",
    "Fold",
    "GateCheck",
    "GradientBoostingBaseline",
    "HyperparameterSearch",
    "LogisticBaseline",
    "MarketFavourite",
    "MarketUnderdog",
    "ModelCard",
    "ModelRegistry",
    "ModelTrainer",
    "PromotionDecision",
    "RandomPredictor",
    "ReadinessCheck",
    "ReadinessReport",
    "ScoreCard",
    "SearchMethod",
    "SearchResult",
    "ShadowResult",
    "SplitScheme",
    "TestResult",
    "TimeSeriesSplitter",
    "TrainingData",
    "TrainingResult",
    "assert_no_leakage_between",
    "binomial_vs_breakeven",
    "brier_score",
    "calibration_curve",
    "check_readiness",
    "complete_timeline_count",
    "default_baselines",
    "evaluate",
    "evaluate_promotion",
    "explain_model",
    "log_loss",
    "make_model_id",
    "mcnemar",
    "paired_bootstrap",
    "readiness_for_store",
    "run_baselines",
    "run_shadow_validation",
    "score",
    "stable_hash",
    "write_report",
]
