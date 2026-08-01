"""Calibration metrics and calibrators.

For this system calibration matters more than accuracy, and the reason is
mechanical rather than aesthetic: position size is a function of the predicted
probability. Kelly stake for a binary contract bought at price ``c`` is
``(p - c) / (1 - c)``. A model that says 0.80 when the truth is 0.60 does not
merely mis-rank — it sizes the position as though the edge were three times
larger than it is, and it does that most aggressively exactly when it is most
wrong.

So every model reports Brier, log loss, a calibration curve, and ECE, and the
promotion gate tests calibration explicitly rather than inferring it from
accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pmbtc.constants import PRICE_MAX, PRICE_MIN


def clip_probabilities(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), PRICE_MIN, PRICE_MAX)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better."""
    return float(np.mean((clip_probabilities(y_prob) - np.asarray(y_true, float)) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = clip_probabilities(y_prob)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((clip_probabilities(y_prob) >= 0.5) == np.asarray(y_true, bool)))


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return abs(self.mean_predicted - self.observed_rate)


@dataclass
class CalibrationCurve:
    bins: list[CalibrationBin] = field(default_factory=list)
    samples: int = 0

    @property
    def ece(self) -> float:
        """Expected calibration error: sample-weighted mean gap.

        The headline calibration number. 0.02 means that, on average, a stated
        probability is two points away from the realised frequency.
        """
        if not self.samples:
            return 1.0
        return sum(b.count * b.gap for b in self.bins) / self.samples

    @property
    def mce(self) -> float:
        """Maximum calibration error — the worst bin, not the average.

        A model can have a fine ECE and still be badly wrong in the confident
        tail, which is precisely where it sizes largest.
        """
        return max((b.gap for b in self.bins if b.count > 0), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "samples": self.samples,
            "bins": [
                {
                    "range": [round(b.lower, 3), round(b.upper, 3)],
                    "count": b.count,
                    "mean_predicted": round(b.mean_predicted, 6),
                    "observed_rate": round(b.observed_rate, 6),
                    "gap": round(b.gap, 6),
                }
                for b in self.bins
            ],
        }

    def render(self) -> str:
        """Text calibration plot — readable in a terminal and in a log."""
        lines = ["predicted -> observed   n     gap"]
        for b in self.bins:
            if not b.count:
                continue
            marker = "#" * min(40, int(b.observed_rate * 40))
            lines.append(
                f"[{b.lower:.2f},{b.upper:.2f})  {b.mean_predicted:.3f} -> "
                f"{b.observed_rate:.3f}  {b.count:<5} {b.gap:+.3f} {marker}"
            )
        lines.append(f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  n={self.samples}")
        return "\n".join(lines)


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> CalibrationCurve:
    """Reliability diagram with equal-width probability bins.

    Equal-width rather than equal-count: the question "when the model says 0.9,
    how often is it right" is about a fixed probability range, and quantile bins
    would blur it across whatever range happened to be popular.
    """
    p = clip_probabilities(y_prob)
    y = np.asarray(y_true, dtype=float)
    if len(p) == 0:
        return CalibrationCurve()

    edges = np.linspace(0.0, 1.0, bins + 1)
    curve = CalibrationCurve(samples=len(p))
    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (p >= lower) & (p < upper) if i < bins - 1 else (p >= lower) & (p <= upper)
        count = int(mask.sum())
        curve.bins.append(
            CalibrationBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_predicted=float(p[mask].mean()) if count else 0.0,
                observed_rate=float(y[mask].mean()) if count else 0.0,
            )
        )
    return curve


@dataclass
class EvaluationMetrics:
    """The full metric set every model reports."""

    samples: int
    accuracy: float
    brier: float
    log_loss: float
    ece: float
    mce: float
    mean_prediction: float
    base_rate: float
    curve: CalibrationCurve | None = None

    @property
    def brier_skill(self) -> float:
        """Skill against always predicting the base rate.

        Positive means the model beats a constant forecast; zero or negative
        means it has learned nothing useful, whatever its accuracy says.
        """
        reference = self.base_rate * (1.0 - self.base_rate)
        return 1.0 - self.brier / reference if reference > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "accuracy": round(self.accuracy, 6),
            "brier": round(self.brier, 6),
            "log_loss": round(self.log_loss, 6),
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "brier_skill": round(self.brier_skill, 6),
            "mean_prediction": round(self.mean_prediction, 6),
            "base_rate": round(self.base_rate, 6),
        }

    def __str__(self) -> str:
        return (
            f"n={self.samples} acc={self.accuracy:.4f} brier={self.brier:.4f} "
            f"logloss={self.log_loss:.4f} ece={self.ece:.4f} "
            f"skill={self.brier_skill:+.4f}"
        )


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> EvaluationMetrics:
    """Compute every reported metric in one pass."""
    y = np.asarray(y_true, dtype=float)
    p = clip_probabilities(y_prob)
    curve = calibration_curve(y, p, bins)
    return EvaluationMetrics(
        samples=len(y),
        accuracy=accuracy(y, p) if len(y) else 0.0,
        brier=brier_score(y, p) if len(y) else 1.0,
        log_loss=log_loss(y, p) if len(y) else float("inf"),
        ece=curve.ece,
        mce=curve.mce,
        mean_prediction=float(p.mean()) if len(p) else 0.0,
        base_rate=float(y.mean()) if len(y) else 0.0,
        curve=curve,
    )


# --------------------------------------------------------------------------- #
# Calibrators
# --------------------------------------------------------------------------- #
class Calibrator:
    """Maps raw model scores onto calibrated probabilities.

    Fitted on its own slice of the training data, never on the data used to fit
    the model — a calibrator trained on the model's own training predictions
    learns the model's overfit rather than correcting it.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self._model: Any = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> Calibrator:
        p = clip_probabilities(y_prob).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        if self.method == "none" or len(np.unique(y)) < 2:
            self._model = None
            return self
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(p.ravel(), y)
            self._model = model
        elif self.method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1_000)
            model.fit(p, y)
            self._model = model
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        p = clip_probabilities(y_prob)
        if self._model is None:
            return p
        if self.method == "isotonic":
            return clip_probabilities(self._model.predict(p))
        return clip_probabilities(self._model.predict_proba(p.reshape(-1, 1))[:, 1])

    @property
    def fitted(self) -> bool:
        return self._model is not None or self.method == "none"
