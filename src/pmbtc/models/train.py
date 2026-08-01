"""Training orchestration.

The order of operations is the whole point, so it is written out explicitly:

1. **Readiness gate** — refuse to train on an inadequate dataset.
2. **Data audit** — refuse to train on a *corrupt* dataset (leakage, duplicates,
   provider mismatch, class skew).
3. **Holdout carve-out** — reserve the chronological tail before anything else
   touches the data. Nothing downstream can see it.
4. **Feature selection** — on development folds only.
5. **Hyperparameter search** — on development folds only.
6. **Fit + calibrate** — calibration on its own slice, not on the fit data.
7. **Evaluate** — cross-validated, then once on the holdout.
8. **Baselines** — same holdout, same rows.
9. **Explain**, **shadow validate**, then the **promotion gate**.

Steps 1 and 2 are not optional and have no override. Step 3 happens before step
4 for a reason: a selection routine that has seen the holdout has already
contaminated it, whatever it does afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pmbtc.config import Config
from pmbtc.dataset.audit import AuditReport, audit_dataset
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord
from pmbtc.exceptions import InsufficientDataError
from pmbtc.features.selection import SelectionResult, select_features
from pmbtc.logging_setup import get_logger
from pmbtc.models.baselines import BaselineReport, default_baselines
from pmbtc.models.calibration import Calibrator, EvaluationMetrics, evaluate
from pmbtc.models.explain import Explanation, explain_model
from pmbtc.models.promotion import PromotionDecision, evaluate_promotion
from pmbtc.models.readiness import ReadinessReport, check_readiness
from pmbtc.models.registry import ModelCard, ModelRegistry, make_model_id, stable_hash
from pmbtc.models.search import (
    DEFAULT_SPACES,
    HyperparameterSearch,
    SearchMethod,
    SearchResult,
)
from pmbtc.models.shadow import ShadowResult, run_shadow_validation
from pmbtc.models.validation import (
    SplitScheme,
    TimeSeriesSplitter,
    assert_no_leakage_between,
)
from pmbtc.ops.gitinfo import require_clean_tree
from pmbtc.utils.timeutils import utc_now_ms

log = get_logger("pmbtc.models.train")


@dataclass
class TrainingData:
    """A prepared, ordered training set."""

    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    groups: np.ndarray
    market_prob: np.ndarray
    feature_names: list[str]
    dataset_version: str = ""
    feature_schema_version: str = ""

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class TrainingResult:
    card: ModelCard | None = None
    readiness: ReadinessReport | None = None
    audit: AuditReport | None = None
    selection: SelectionResult | None = None
    search: SearchResult | None = None
    cv_metrics: list[EvaluationMetrics] = field(default_factory=list)
    holdout: EvaluationMetrics | None = None
    baselines: BaselineReport | None = None
    explanation: Explanation | None = None
    shadow: ShadowResult | None = None
    decision: PromotionDecision | None = None
    blocked_reason: str = ""

    @property
    def trained(self) -> bool:
        return self.card is not None

    @property
    def promoted(self) -> bool:
        return bool(self.decision and self.decision.approved)

    def summary(self) -> str:
        if self.blocked_reason:
            return f"training blocked: {self.blocked_reason}"
        if not self.card:
            return "training did not produce a model"
        parts = [f"model {self.card.model_id}"]
        if self.holdout:
            parts.append(str(self.holdout))
        if self.decision:
            parts.append(self.decision.summary())
        return " | ".join(parts)


def _fit_estimator(model_type: str, params: dict[str, Any], seed: int) -> Any:
    """Construct an unfitted estimator. Kept small and explicit."""
    if model_type == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, **params)
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1_000, random_state=seed, **params),
        )
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    raise ValueError(f"Unknown model type: {model_type}")


def _predict_proba(estimator: Any, x: np.ndarray) -> np.ndarray:
    filled = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    except (ValueError, TypeError):
        # Estimators that cannot handle NaN natively get the filled matrix.
        return np.asarray(estimator.predict_proba(filled)[:, 1], dtype=float)


class ModelTrainer:
    """Runs the full training and evaluation pipeline."""

    def __init__(self, config: Config, registry: ModelRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ModelRegistry(
            config.resolved_path(config.app.artifact_dir) / "models"
        )

    # ------------------------------------------------------------------ #
    def train(
        self,
        data: TrainingData,
        *,
        model_type: str = "gradient_boosting",
        markets: list[MarketRecord] | None = None,
        snapshots: list[FeatureSnapshot] | None = None,
        scheme: SplitScheme = SplitScheme.EXPANDING,
        search_method: SearchMethod = SearchMethod.GRID,
        enforce_gates: bool = True,
        holdout_fraction: float = 0.2,
    ) -> TrainingResult:
        result = TrainingResult()
        seed = self.config.model.random_seed

        # --- 1 & 2. Gates -------------------------------------------- #
        if enforce_gates:
            if markets is None or snapshots is None:
                raise InsufficientDataError(
                    "Readiness and audit gates require the dataset records",
                    context={"hint": "pass markets= and snapshots=, or disable gates"},
                )
            result.readiness = check_readiness(self.config, markets, snapshots)
            if not result.readiness.ready:
                result.blocked_reason = result.readiness.summary()
                log.error("train.blocked_readiness", detail=result.blocked_reason)
                return result

            result.audit = audit_dataset(self.config, markets, snapshots)
            if not result.audit.passed:
                result.blocked_reason = result.audit.summary()
                log.error("train.blocked_audit", detail=result.blocked_reason)
                return result

        # --- 3. Holdout, carved out before anything else sees the data - #
        splitter = TimeSeriesSplitter(
            scheme=scheme,
            n_folds=self.config.model.cv_folds,
            embargo_markets=max(1, self.config.model.embargo_windows // 4),
            min_train_markets=max(10, self.config.model.cv_folds * 4),
        )
        develop_rows, holdout_rows = splitter.final_holdout(
            data.timestamps, data.groups, holdout_fraction
        )
        assert_no_leakage_between(
            develop_rows, holdout_rows, data.timestamps, data.groups
        )

        x_dev, y_dev = data.x[develop_rows], data.y[develop_rows]
        ts_dev, groups_dev = data.timestamps[develop_rows], data.groups[develop_rows]
        x_hold, y_hold = data.x[holdout_rows], data.y[holdout_rows]

        # --- 4. Feature selection, development folds only -------------- #
        result.selection = select_features(
            x_dev,
            y_dev,
            list(data.feature_names),
            max_features=self.config.features.selection.max_features,
            vote_threshold=self.config.features.selection.vote_threshold,
            max_correlation=self.config.features.selection.max_correlation,
            seed=seed,
        )
        chosen = list(result.selection.selected) or list(data.feature_names)
        columns = [data.feature_names.index(name) for name in chosen]
        x_dev_sel, x_hold_sel = x_dev[:, columns], x_hold[:, columns]

        # --- 5. Hyperparameter search, development folds only ---------- #
        space = DEFAULT_SPACES.get(model_type, {})

        def fit_predict(
            params: dict[str, Any], xt: np.ndarray, yt: np.ndarray, xv: np.ndarray
        ) -> np.ndarray:
            estimator = _fit_estimator(model_type, params, seed)
            estimator.fit(np.nan_to_num(xt), yt)
            return _predict_proba(estimator, np.nan_to_num(xv))

        result.search = HyperparameterSearch(
            space=space,
            method=search_method,
            max_trials=min(self.config.model.hyperparam_trials, 24),
            seed=seed,
            splitter=splitter,
        ).run(fit_predict, x_dev_sel, y_dev, ts_dev, groups_dev)
        best_params = result.search.best_params

        # --- 6 & 7. Cross-validated evaluation ------------------------- #
        for fold in splitter.split(ts_dev, groups_dev):
            assert_no_leakage_between(fold.train, fold.test, ts_dev, groups_dev)
            predictions = fit_predict(
                best_params, x_dev_sel[fold.train], y_dev[fold.train], x_dev_sel[fold.test]
            )
            result.cv_metrics.append(evaluate(y_dev[fold.test], predictions))

        # Final fit: the last slice of development data is reserved for the
        # calibrator, so it is never fitted on its own training predictions.
        cut = max(1, int(len(y_dev) * 0.85))
        estimator = _fit_estimator(model_type, best_params, seed)
        estimator.fit(np.nan_to_num(x_dev_sel[:cut]), y_dev[:cut])

        calibrator = Calibrator(self.config.model.calibration)
        if cut < len(y_dev):
            calibrator.fit(_predict_proba(estimator, x_dev_sel[cut:]), y_dev[cut:])

        def predict(matrix: np.ndarray) -> np.ndarray:
            return calibrator.transform(_predict_proba(estimator, np.nan_to_num(matrix)))

        holdout_probs = predict(x_hold_sel)
        result.holdout = evaluate(y_hold, holdout_probs)

        # --- 8. Baselines on the identical holdout rows ---------------- #
        market_hold = data.market_prob[holdout_rows]
        scores = []
        baseline_probs: dict[str, np.ndarray] = {}
        for baseline in default_baselines(seed):
            if baseline.requires_fit:
                baseline.fit(np.nan_to_num(x_dev_sel), y_dev)
            probs = baseline.predict_proba(x_hold_sel, market_hold)
            baseline_probs[baseline.name] = np.asarray(probs, dtype=float)
            scores.append(baseline.evaluate(x_hold_sel, y_hold, market_hold))
        result.baselines = BaselineReport(scores)

        # --- 9. Explain, shadow, gate ---------------------------------- #
        result.explanation = explain_model(
            estimator, predict, x_hold_sel, y_hold, chosen,
            curve=result.holdout.curve, seed=seed,
        )
        result.shadow = run_shadow_validation(
            predict, x_hold_sel, y_hold, data.timestamps[holdout_rows]
        )

        champion_probs = None
        champion = self.registry.champion()
        if champion is not None:
            try:
                champion_estimator, champion_calibrator = self.registry.load_estimator(
                    champion.model_id
                )
                if list(champion.features) == chosen:
                    raw = _predict_proba(champion_estimator, np.nan_to_num(x_hold_sel))
                    champion_probs = (
                        champion_calibrator.transform(raw) if champion_calibrator else raw
                    )
                else:
                    log.info("train.champion_feature_mismatch", model_id=champion.model_id)
            except Exception as exc:
                log.warning("train.champion_unavailable", error=str(exc))

        result.decision = evaluate_promotion(
            self.config,
            candidate=result.holdout,
            candidate_probs=holdout_probs,
            y_holdout=y_hold,
            baselines=result.baselines,
            baseline_probs=baseline_probs,
            shadow=result.shadow,
            champion_probs=champion_probs,
        )

        # --- Card and artifact ----------------------------------------- #
        trained_at = utc_now_ms()
        config_root = self.config.resolved_path(self.config.app.base_dir)
        config_hash = stable_hash(
            {
                "model": self.config.model.model_dump(mode="json"),
                "training": self.config.training.model_dump(mode="json"),
                "selection": self.config.features.selection.model_dump(mode="json"),
                "scheme": scheme.value,
            }
        )
        hyper_hash = stable_hash(best_params)
        # What code actually ran. A dirty tree is recorded, not hidden: a model
        # trained from uncommitted changes cannot be rebuilt from history.
        provenance = require_clean_tree(config_root)
        card = ModelCard(
            model_id=make_model_id(
                model_type, data.dataset_version, data.feature_schema_version,
                config_hash, hyper_hash, trained_at,
            ),
            model_type=model_type,
            dataset_version=data.dataset_version,
            feature_schema_version=data.feature_schema_version,
            config_hash=config_hash,
            hyperparameter_hash=hyper_hash,
            trained_at_ms=trained_at,
            features=tuple(chosen),
            hyperparameters=best_params,
            training_samples=int(cut),
            seed=seed,
            metrics={
                "holdout": result.holdout.as_dict(),
                "cross_validation": {
                    "folds": len(result.cv_metrics),
                    "mean_brier": round(
                        float(np.mean([m.brier for m in result.cv_metrics])), 6
                    ) if result.cv_metrics else None,
                    "mean_ece": round(
                        float(np.mean([m.ece for m in result.cv_metrics])), 6
                    ) if result.cv_metrics else None,
                },
                "calibration_curve": result.holdout.curve.as_dict()
                if result.holdout.curve else {},
            },
            baseline_metrics={s.name: asdict(s) for s in result.baselines.scores},
            statistical_tests=[t.as_dict() for t in result.decision.tests],
            validation={
                "scheme": scheme.value,
                "folds": len(result.cv_metrics),
                "holdout_rows": len(holdout_rows),
                "develop_rows": len(develop_rows),
                "selection": {
                    "selected": len(chosen),
                    "considered": result.selection.considered,
                },
                "search": result.search.as_dict() if result.search else {},
            },
            explainability=result.explanation.as_dict(),
            shadow=result.shadow.as_dict(),
            git_commit=provenance.commit,
            git_branch=provenance.branch,
            git_tag=provenance.tag,
            git_dirty=provenance.dirty,
        )
        result.card = card
        self.registry.save(card, estimator, calibrator)

        if result.decision.approved:
            self.registry.promote(card.model_id, result.decision.reasons)
        else:
            log.info("train.not_promoted", model_id=card.model_id,
                     reasons=[c.name for c in result.decision.failures])
        return result


def write_report(result: TrainingResult, path: Path) -> Path:
    """Write the full evaluation report next to the model artifact."""
    import json

    payload: dict[str, Any] = {
        "summary": result.summary(),
        "promoted": result.promoted,
        "blocked_reason": result.blocked_reason,
    }
    if result.card:
        payload["model"] = result.card.as_dict()
    if result.baselines:
        payload["baselines"] = [asdict(s) for s in result.baselines.scores]
    if result.decision:
        payload["promotion"] = result.decision.as_dict()
    if result.readiness:
        payload["readiness"] = {
            "ready": result.readiness.ready,
            "checks": [
                {"name": c.name, "actual": c.actual, "required": c.required,
                 "passed": c.passed}
                for c in result.readiness.checks
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
