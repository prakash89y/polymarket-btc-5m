"""Benchmark baselines.

No model earns deployment by being good in the abstract. It earns it by beating
these, on data it has never seen. Most of them are trivial, which is the point:
a "sophisticated" model that cannot beat *always back the market favourite* has
learned nothing except how to be expensive.

The five required baselines:

``RandomPredictor``
    Coin flip. The floor. A model below this is worse than nothing.
``MarketFavourite``
    Always predict whatever the Polymarket book favours. This is the one that
    matters. The market price is a well-informed forecast, and beating it is the
    entire premise of the project. Expect it to be hard.
``MarketUnderdog``
    Always predict against the book. Included because it is the exact inverse of
    the favourite: if it wins, the market is systematically biased and something
    is wrong with our understanding, not right with our model.
``LogisticBaseline``
    A linear model on the same features. If the gradient-boosted ensemble cannot
    beat this, the extra complexity is unjustified.
``GradientBoostingBaseline``
    A default-hyperparameter GBDT. The honest "just throw sklearn at it" bar
    that any bespoke modelling has to clear.

Everything is scored with Brier and log loss as well as accuracy, because this
system trades on calibrated probabilities, not on argmax. A model that is 52%
accurate but well calibrated is tradeable; one that is 55% accurate and
overconfident is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from pmbtc.constants import PRICE_MAX, PRICE_MIN
from pmbtc.logging_setup import get_logger

log_ = get_logger("pmbtc.models.baselines")


@dataclass(frozen=True, slots=True)
class ScoreCard:
    """How a predictor did on a set of markets."""

    name: str
    samples: int
    accuracy: float
    brier: float
    log_loss: float
    #: Mean predicted probability, to expose a predictor that is simply always
    #: confident in one direction.
    mean_prediction: float

    def beats(self, other: ScoreCard, margin: float = 0.0) -> bool:
        """Better means better *calibrated*: lower Brier by at least a margin."""
        return (other.brier - self.brier) >= margin

    def __str__(self) -> str:
        return (
            f"{self.name:<26} n={self.samples:<6} acc={self.accuracy:.4f} "
            f"brier={self.brier:.4f} logloss={self.log_loss:.4f} "
            f"mean_p={self.mean_prediction:.3f}"
        )


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values, PRICE_MIN, PRICE_MAX)


def score(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> ScoreCard:
    """Accuracy, Brier, and log loss for probabilistic predictions."""
    if len(y_true) == 0:
        return ScoreCard(name, 0, 0.0, 1.0, float("inf"), 0.0)
    probs = _clip(np.asarray(y_prob, dtype=float))
    truth = np.asarray(y_true, dtype=float)
    accuracy = float(np.mean((probs >= 0.5).astype(float) == truth))
    brier = float(np.mean((probs - truth) ** 2))
    logloss = float(
        -np.mean(truth * np.log(probs) + (1.0 - truth) * np.log(1.0 - probs))
    )
    return ScoreCard(name, len(truth), accuracy, brier, logloss, float(np.mean(probs)))


class Baseline(ABC):
    """A predictor that emits P(Up)."""

    name = "baseline"
    #: Whether the predictor needs fitting at all.
    requires_fit = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> Baseline:
        return self

    @abstractmethod
    def predict_proba(self, x: np.ndarray, market_prob: np.ndarray | None = None) -> np.ndarray:
        """Return P(Up) for each row."""

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, market_prob: np.ndarray | None = None
    ) -> ScoreCard:
        return score(self.name, y, self.predict_proba(x, market_prob))


class RandomPredictor(Baseline):
    """Coin flip at a constant 0.5. The floor every model must clear."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def predict_proba(self, x: np.ndarray, market_prob: np.ndarray | None = None) -> np.ndarray:
        # A constant 0.5 rather than uniform noise: it is the *best* random
        # predictor by Brier score, so it is the honest version of this floor.
        return np.full(len(x), 0.5)


class MarketFavourite(Baseline):
    """Take the market's own probability at face value.

    The benchmark that actually matters. Beating a liquid book's forecast is the
    whole thesis; if the ensemble cannot, there is no edge to trade.
    """

    name = "market_favourite"

    def predict_proba(self, x: np.ndarray, market_prob: np.ndarray | None = None) -> np.ndarray:
        if market_prob is None:
            return np.full(len(x), 0.5)
        return _clip(np.asarray(market_prob, dtype=float))


class MarketUnderdog(Baseline):
    """The exact inverse of the market's forecast.

    If this wins, do not celebrate — the market is not systematically wrong on a
    liquid BTC book, so a win here means our labels, our market mapping, or our
    settlement understanding is inverted somewhere.
    """

    name = "market_underdog"

    def predict_proba(self, x: np.ndarray, market_prob: np.ndarray | None = None) -> np.ndarray:
        if market_prob is None:
            return np.full(len(x), 0.5)
        return _clip(1.0 - np.asarray(market_prob, dtype=float))


class LogisticBaseline(Baseline):
    """Regularised linear model on the feature matrix."""

    name = "logistic"
    requires_fit = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticBaseline:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Scaling matters here: features range from probabilities in [0,1] to
        # basis points in the thousands, and an unscaled linear model would be
        # dominated by whichever column happens to be largest.
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1_000, random_state=self.seed),
        )
        self._model.fit(np.nan_to_num(x), y)
        return self

    def predict_proba(self, x: np.ndarray, market_prob: np.ndarray | None = None) -> np.ndarray:
        if self._model is None:
            return np.full(len(x), 0.5)
        return _clip(self._model.predict_proba(np.nan_to_num(x))[:, 1])


class GradientBoostingBaseline(Baseline):
    """Default-hyperparameter GBDT — the "just use sklearn" bar."""

    name = "gradient_boosting"
    requires_fit = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> GradientBoostingBaseline:
        from sklearn.ensemble import HistGradientBoostingClassifier

        # HistGradientBoosting handles NaN natively, which matters because
        # missing features are a real and recorded state in this dataset.
        self._model = HistGradientBoostingClassifier(
            random_state=self.seed, max_iter=200, early_stopping=True
        )
        self._model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray, market_prob: np.ndarray | None = None) -> np.ndarray:
        if self._model is None:
            return np.full(len(x), 0.5)
        return _clip(self._model.predict_proba(x)[:, 1])


def default_baselines(seed: int = 42) -> list[Baseline]:
    """Every baseline a candidate model must beat."""
    return [
        RandomPredictor(seed),
        MarketFavourite(),
        MarketUnderdog(),
        LogisticBaseline(seed),
        GradientBoostingBaseline(seed),
    ]


@dataclass
class BaselineReport:
    scores: list[ScoreCard]

    @property
    def best(self) -> ScoreCard | None:
        """Lowest Brier — the bar a candidate model must clear."""
        return min(self.scores, key=lambda s: s.brier) if self.scores else None

    def clears_all(self, candidate: ScoreCard, margin: float = 0.0) -> bool:
        return all(candidate.beats(s, margin) for s in self.scores)

    def blocking(self, candidate: ScoreCard, margin: float = 0.0) -> list[ScoreCard]:
        """Baselines the candidate failed to beat — the actionable list."""
        return [s for s in self.scores if not candidate.beats(s, margin)]

    def __str__(self) -> str:
        return "\n".join(str(s) for s in sorted(self.scores, key=lambda s: s.brier))


def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    market_prob_test: np.ndarray | None = None,
    seed: int = 42,
) -> BaselineReport:
    """Fit and score every baseline on an unseen test set."""
    scores: list[ScoreCard] = []
    for baseline in default_baselines(seed):
        try:
            if baseline.requires_fit:
                baseline.fit(x_train, y_train)
            scores.append(baseline.evaluate(x_test, y_test, market_prob_test))
        except Exception as exc:
            log_.warning("baseline.failed", baseline=baseline.name, error=str(exc))
    return BaselineReport(scores)
