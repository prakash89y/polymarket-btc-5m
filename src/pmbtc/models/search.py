"""Hyperparameter search — inside the training folds, never outside them.

The search evaluates candidates using the same time-aware splitter the final
model uses, on the development portion of the data only. The holdout is not
passed to this module at all, which is the structural version of "do not tune on
the test set".

Grid and random search are built in. Bayesian optimisation is optional: if
``optuna`` is installed it is used, otherwise the search falls back to random
with a recorded note rather than silently pretending it ran.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from pmbtc.logging_setup import get_logger
from pmbtc.models.calibration import brier_score
from pmbtc.models.registry import stable_hash
from pmbtc.models.validation import TimeSeriesSplitter, assert_no_leakage_between

log = get_logger("pmbtc.models.search")


class SearchMethod(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class Trial:
    params: dict[str, Any]
    score: float
    fold_scores: list[float] = field(default_factory=list)

    @property
    def params_hash(self) -> str:
        return stable_hash(self.params)


@dataclass
class SearchResult:
    best: Trial | None
    trials: list[Trial] = field(default_factory=list)
    method: str = "grid"
    note: str = ""

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self.best.params) if self.best else {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "note": self.note,
            "trials": len(self.trials),
            "best_params": self.best_params,
            "best_score": round(self.best.score, 6) if self.best else None,
            "top_5": [
                {"params": t.params, "score": round(t.score, 6)}
                for t in sorted(self.trials, key=lambda t: t.score)[:5]
            ],
        }


def grid_candidates(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Every combination, in a deterministic order."""
    keys = sorted(space)
    for values in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def random_candidates(
    space: dict[str, list[Any]], n: int, seed: int
) -> Iterator[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = sorted(space)
    seen: set[str] = set()
    # Bounded attempts: with a small grid, random sampling would otherwise spin
    # forever looking for combinations that do not exist.
    for _ in range(n * 10):
        if len(seen) >= n:
            return
        candidate = {k: space[k][int(rng.integers(0, len(space[k])))] for k in keys}
        key = stable_hash(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


@dataclass
class HyperparameterSearch:
    """Cross-validated search over the development set."""

    space: dict[str, list[Any]]
    method: SearchMethod = SearchMethod.GRID
    max_trials: int = 24
    seed: int = 42
    splitter: TimeSeriesSplitter = field(default_factory=TimeSeriesSplitter)

    def run(
        self,
        fit_predict: Callable[[dict[str, Any], np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        groups: np.ndarray,
    ) -> SearchResult:
        """Score every candidate by mean out-of-fold Brier.

        ``fit_predict(params, x_train, y_train, x_test) -> probabilities``.
        The caller supplies it so this module never needs to know about a
        specific estimator, and so nothing here can reach outside its folds.
        """
        if not self.space:
            return SearchResult(best=None, method=self.method.value, note="empty space")

        note = ""
        if self.method is SearchMethod.BAYESIAN:
            try:
                import optuna  # noqa: F401
            except ImportError:
                note = "optuna unavailable; fell back to random search"
                log.info("search.bayesian_unavailable")

        candidates = (
            list(grid_candidates(self.space))
            if self.method is SearchMethod.GRID
            else list(random_candidates(self.space, self.max_trials, self.seed))
        )
        candidates = candidates[: self.max_trials]

        folds = list(self.splitter.split(timestamps, groups))
        if not folds:
            return SearchResult(best=None, method=self.method.value, note="no folds")

        trials: list[Trial] = []
        for params in candidates:
            fold_scores: list[float] = []
            for fold in folds:
                assert_no_leakage_between(fold.train, fold.test, timestamps, groups)
                try:
                    predictions = fit_predict(
                        params, x[fold.train], y[fold.train], x[fold.test]
                    )
                    fold_scores.append(brier_score(y[fold.test], predictions))
                except Exception as exc:
                    log.warning("search.trial_failed", params=params, error=str(exc))
                    fold_scores.append(1.0)
            trials.append(
                Trial(
                    params=params,
                    score=float(np.mean(fold_scores)),
                    fold_scores=fold_scores,
                )
            )

        # Ties broken by the parameter hash so the winner is reproducible.
        best = min(trials, key=lambda t: (t.score, t.params_hash)) if trials else None
        log.info(
            "search.completed",
            method=self.method.value,
            trials=len(trials),
            best_score=round(best.score, 6) if best else None,
        )
        return SearchResult(
            best=best, trials=trials, method=self.method.value, note=note
        )


#: Conservative default spaces. Small on purpose: with a few thousand markets, a
#: large search is a licence to overfit the validation folds.
DEFAULT_SPACES: dict[str, dict[str, list[Any]]] = {
    "gradient_boosting": {
        "max_iter": [100, 200, 400],
        "learning_rate": [0.03, 0.06, 0.1],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.0, 1.0],
    },
    "logistic": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [5, 20],
    },
}
