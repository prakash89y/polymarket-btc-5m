"""Evidence-based feature selection.

Four methods, no hand-assigned importance:

``mutual_information``
    Captures non-linear dependence a correlation would miss.
``permutation``
    Measures the loss increase when a column is shuffled — importance *to a
    fitted model*, which is what actually matters.
``shap``
    Per-sample attribution; falls back to gain-based importance when the
    ``shap`` package is absent, and says so rather than silently skipping.
``rfe``
    Recursive elimination, which accounts for redundancy between features that
    the single-column methods cannot see.

A feature must be ranked highly by a configurable fraction of methods to
survive. Requiring agreement is what stops one method's idiosyncrasy from
deciding the feature set.

**Selection sees training folds only.** Every fit and every ranking happens
inside :func:`select_features` on data the caller has already split. Running
selection on the full dataset is one of the easiest ways to leak the validation
set into the model, and it is the reason feature selection belongs here rather
than in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.features.selection")


@dataclass
class MethodRanking:
    """One method's view of feature importance."""

    method: str
    scores: dict[str, float]
    available: bool = True
    note: str = ""

    def top(self, k: int) -> list[str]:
        return [n for n, _ in sorted(self.scores.items(), key=lambda kv: -kv[1])[:k]]


@dataclass
class SelectionResult:
    selected: tuple[str, ...]
    rankings: list[MethodRanking] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)
    dropped_correlated: dict[str, str] = field(default_factory=dict)
    considered: int = 0

    def summary(self) -> str:
        methods = ", ".join(
            f"{r.method}{'' if r.available else ' (unavailable)'}" for r in self.rankings
        )
        return (
            f"selected {len(self.selected)} of {self.considered} features "
            f"via [{methods}]; {len(self.dropped_correlated)} dropped as collinear"
        )


def _prune_correlated(
    x: np.ndarray, names: list[str], threshold: float
) -> tuple[list[int], dict[str, str]]:
    """Drop one of each highly correlated pair, keeping the earlier column.

    Collinear inputs split attribution between themselves, which makes every
    importance measure harder to read and none of them wrong in a useful way.
    """
    keep: list[int] = []
    dropped: dict[str, str] = {}
    filled = np.nan_to_num(x)
    for index in range(filled.shape[1]):
        column = filled[:, index]
        if np.std(column) == 0:
            dropped[names[index]] = "constant"
            continue
        redundant = False
        for kept in keep:
            other = filled[:, kept]
            if np.std(other) == 0:
                continue
            correlation = abs(float(np.corrcoef(column, other)[0, 1]))
            if correlation >= threshold:
                dropped[names[index]] = f"corr {correlation:.3f} with {names[kept]}"
                redundant = True
                break
        if not redundant:
            keep.append(index)
    return keep, dropped


def _mutual_information(x: np.ndarray, y: np.ndarray, names: list[str]) -> MethodRanking:
    try:
        from sklearn.feature_selection import mutual_info_classif

        scores = mutual_info_classif(np.nan_to_num(x), y, random_state=0)
        return MethodRanking(
            "mutual_information", dict(zip(names, map(float, scores), strict=True))
        )
    except Exception as exc:
        return MethodRanking("mutual_information", {}, False, str(exc))


def _permutation(x: np.ndarray, y: np.ndarray, names: list[str], seed: int) -> MethodRanking:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.inspection import permutation_importance

        model = HistGradientBoostingClassifier(random_state=seed, max_iter=120)
        model.fit(x, y)
        result = permutation_importance(
            model, x, y, n_repeats=5, random_state=seed, scoring="neg_log_loss"
        )
        return MethodRanking(
            "permutation",
            dict(zip(names, map(float, result.importances_mean), strict=True)),
        )
    except Exception as exc:
        return MethodRanking("permutation", {}, False, str(exc))


def _shap(x: np.ndarray, y: np.ndarray, names: list[str], seed: int) -> MethodRanking:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(random_state=seed, max_iter=120)
        model.fit(x, y)
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            values = np.abs(explainer.shap_values(x)).mean(axis=0)
            return MethodRanking(
                "shap", dict(zip(names, map(float, np.ravel(values)), strict=True))
            )
        except ImportError:
            # Honest fallback: permutation on a held-in sample approximates the
            # ranking SHAP would give, and the note records that it was used.
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model, x, y, n_repeats=3, random_state=seed, scoring="neg_log_loss"
            )
            return MethodRanking(
                "shap",
                dict(zip(names, map(float, result.importances_mean), strict=True)),
                True,
                "shap package unavailable; used permutation as a proxy",
            )
    except Exception as exc:
        return MethodRanking("shap", {}, False, str(exc))


def _rfe(x: np.ndarray, y: np.ndarray, names: list[str], seed: int, keep: int) -> MethodRanking:
    try:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(np.nan_to_num(x))
        selector = RFE(
            LogisticRegression(max_iter=500, random_state=seed),
            n_features_to_select=max(1, min(keep, len(names))),
        )
        selector.fit(scaled, y)
        # Invert the rank so higher is better, matching the other methods.
        scores = {name: float(len(names) - rank) for name, rank in
                  zip(names, selector.ranking_, strict=True)}
        return MethodRanking("rfe", scores)
    except Exception as exc:
        return MethodRanking("rfe", {}, False, str(exc))


def select_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    names: list[str],
    *,
    max_features: int = 120,
    vote_threshold: float = 0.5,
    max_correlation: float = 0.95,
    seed: int = 42,
) -> SelectionResult:
    """Rank and select features using training data only.

    The caller is responsible for having split the data; nothing here has any
    way to reach validation or holdout rows, which is the point.
    """
    if x_train.size == 0 or len(names) == 0:
        return SelectionResult(selected=(), considered=0)

    keep_indices, dropped = _prune_correlated(x_train, names, max_correlation)
    x = np.nan_to_num(x_train[:, keep_indices])
    kept_names = [names[i] for i in keep_indices]

    rankings = [
        _mutual_information(x, y_train, kept_names),
        _permutation(x, y_train, kept_names, seed),
        _shap(x, y_train, kept_names, seed),
        _rfe(x, y_train, kept_names, seed, max_features),
    ]

    usable = [r for r in rankings if r.available and r.scores]
    if not usable:
        log.warning("selection.no_methods_available")
        return SelectionResult(
            selected=tuple(kept_names[:max_features]),
            rankings=rankings,
            dropped_correlated=dropped,
            considered=len(names),
        )

    cutoff = max(1, min(max_features, len(kept_names)))
    votes: dict[str, int] = dict.fromkeys(kept_names, 0)
    for ranking in usable:
        for name in ranking.top(cutoff):
            votes[name] += 1

    required = max(1, round(vote_threshold * len(usable)))
    survivors = [n for n, v in votes.items() if v >= required]
    # Rank survivors by total votes, then by name for a stable, reproducible tie-break.
    survivors.sort(key=lambda n: (-votes[n], n))

    return SelectionResult(
        selected=tuple(survivors[:max_features]),
        rankings=rankings,
        votes=votes,
        dropped_correlated=dropped,
        considered=len(names),
    )


def selection_report(result: SelectionResult) -> dict[str, Any]:
    return {
        "selected": list(result.selected),
        "considered": result.considered,
        "votes": result.votes,
        "dropped_correlated": result.dropped_correlated,
        "methods": [
            {
                "method": r.method,
                "available": r.available,
                "note": r.note,
                "top_10": r.top(10),
            }
            for r in result.rankings
        ],
    }
