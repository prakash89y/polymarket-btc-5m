"""Explainability for promoted models.

Not decoration. A model that cannot be explained cannot be debugged, and on this
instrument the most likely explanation for a strong result is a leak — so the
first thing to look at when a model performs well is *which feature it is
leaning on*. A model whose top feature is the market price has learned to copy
the book; one whose top feature is a regime column that updates daily has found
an artefact.

Interactions are computed by a cheap, honest method: measure how much a
feature's contribution changes when another is perturbed. It is not a Shapley
interaction index, and it does not claim to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pmbtc.logging_setup import get_logger
from pmbtc.models.calibration import CalibrationCurve, brier_score

log = get_logger("pmbtc.models.explain")


@dataclass
class Explanation:
    """Global importance, SHAP summary, interactions, and calibration."""

    importance: dict[str, float] = field(default_factory=dict)
    shap_summary: dict[str, float] = field(default_factory=dict)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    method_notes: list[str] = field(default_factory=list)

    def top(self, n: int = 15) -> list[tuple[str, float]]:
        return sorted(self.importance.items(), key=lambda kv: -kv[1])[:n]

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_features": [
                {"feature": name, "importance": round(value, 6)} for name, value in self.top(25)
            ],
            "shap_summary": {
                k: round(v, 6)
                for k, v in sorted(self.shap_summary.items(), key=lambda kv: -kv[1])[:25]
            },
            "interactions": self.interactions[:15],
            "calibration": self.calibration,
            "notes": self.method_notes,
        }

    def render(self, n: int = 15) -> str:
        lines = ["global feature importance (permutation, Brier degradation):"]
        top = self.top(n)
        scale = max((v for _, v in top), default=1.0) or 1.0
        for name, value in top:
            bar = "#" * max(0, min(40, int(40 * value / scale)))
            lines.append(f"  {name:<32} {value:+.5f} {bar}")
        if self.interactions:
            lines.append("strongest interactions:")
            for item in self.interactions[:5]:
                lines.append(
                    f"  {item['a']} x {item['b']}: {item['strength']:.5f}"
                )
        return "\n".join(lines)


def permutation_importance(
    predict: Any,
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    repeats: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Brier degradation when each column is shuffled.

    Measured on data the model did not train on; the number is "how much worse
    the model gets without this feature", which is the question that matters.
    """
    rng = np.random.default_rng(seed)
    baseline = brier_score(y, predict(x))
    scores: dict[str, float] = {}
    for index, name in enumerate(names):
        degradations: list[float] = []
        for _ in range(repeats):
            shuffled = x.copy()
            rng.shuffle(shuffled[:, index])
            degradations.append(brier_score(y, predict(shuffled)) - baseline)
        scores[name] = float(np.mean(degradations))
    return scores


def shap_summary(
    estimator: Any, x: np.ndarray, names: list[str], *, sample: int = 500, seed: int = 42
) -> tuple[dict[str, float], str]:
    """Mean absolute SHAP value per feature, with an honest fallback."""
    try:
        import shap

        rng = np.random.default_rng(seed)
        subset = x if len(x) <= sample else x[rng.choice(len(x), sample, replace=False)]
        explainer = shap.TreeExplainer(estimator)
        values = np.abs(explainer.shap_values(subset))
        if values.ndim == 3:
            values = values[:, :, -1]
        means = values.mean(axis=0)
        return dict(zip(names, map(float, np.ravel(means)), strict=True)), "shap"
    except ImportError:
        return {}, "shap package not installed; global importance used instead"
    except Exception as exc:
        return {}, f"shap failed: {exc}"


def interaction_strength(
    predict: Any,
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    candidates: list[str],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """How much two features' effects depend on each other.

    For each pair: shuffle A alone, shuffle B alone, shuffle both. If the effect
    of shuffling both exceeds the sum of the individual effects, the features
    interact. Restricted to the top candidates, since the pair count is
    quadratic.
    """
    rng = np.random.default_rng(seed)
    index_of = {name: i for i, name in enumerate(names)}
    baseline = brier_score(y, predict(x))

    def degrade(columns: list[int]) -> float:
        shuffled = x.copy()
        for column in columns:
            rng.shuffle(shuffled[:, column])
        return brier_score(y, predict(shuffled)) - baseline

    results: list[dict[str, Any]] = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            if a not in index_of or b not in index_of:
                continue
            ia, ib = index_of[a], index_of[b]
            joint = degrade([ia, ib])
            individual = degrade([ia]) + degrade([ib])
            results.append(
                {"a": a, "b": b, "strength": round(float(joint - individual), 6)}
            )
    return sorted(results, key=lambda item: -abs(item["strength"]))


def explain_model(
    estimator: Any,
    predict: Any,
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    curve: CalibrationCurve | None = None,
    *,
    seed: int = 42,
    interaction_top_k: int = 6,
) -> Explanation:
    """Produce the full explanation bundle for a promoted model."""
    explanation = Explanation()
    explanation.importance = permutation_importance(predict, x, y, names, seed=seed)

    summary, note = shap_summary(estimator, x, names, seed=seed)
    explanation.shap_summary = summary
    if note != "shap":
        explanation.method_notes.append(note)

    top_names = [name for name, _ in explanation.top(interaction_top_k)]
    explanation.interactions = interaction_strength(
        predict, x, y, names, top_names, seed=seed
    )
    if curve is not None:
        explanation.calibration = curve.as_dict()

    log.info(
        "model.explained",
        features=len(names),
        top=[name for name, _ in explanation.top(5)],
    )
    return explanation
