"""Adapters from "a thing that predicts" to the engine's probability callable.

The engine deliberately knows nothing about estimators, so anything that can put
a number on P(UP) can be backtested: a Module 7 model, one of the Module 7
baselines, or a null strategy used as a control. These adapters are the whole of
that translation layer.

:class:`MarketProbabilityModel` is the control that matters. It forecasts exactly
what the book forecasts, so it has no edge by construction and must produce
approximately zero trades. If a run of it *does* trade, the edge calculation is
wrong somewhere — that is a bug in the cost model, not a discovery.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pmbtc.backtest.engine import Row
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.backtest.adapters")


class SupportsPredictProba(Protocol):
    """The sklearn-style surface Module 7's estimators and baselines share."""

    def predict_proba(self, x: Any) -> Any: ...  # pragma: no cover - protocol


def _value(row: Row, column: str) -> float:
    raw = row.get(column)
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number


class MarketProbabilityModel:
    """The book's own forecast. The null strategy, and the bar to beat."""

    name = "market"

    def __init__(self, column: str = "f_ob_mid", fallback: float = 0.5) -> None:
        self.column = column
        self.fallback = fallback

    def __call__(self, row: Row) -> float:
        value = _value(row, self.column)
        if value != value or not (0.0 < value < 1.0):  # NaN or out of range
            return self.fallback
        return value


class ConstantModel:
    """Always the same probability. Used to exercise the gates in tests."""

    name = "constant"

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, row: Row) -> float:
        return self.probability


class EstimatorModel:
    """Wraps a fitted estimator, pinning the column order it was trained on.

    The column list is captured at construction and used for every row, so a
    dataset that gains a feature between training and backtesting cannot
    silently shift the inputs under the model — the same discipline the feature
    matrix's version fingerprint enforces upstream.
    """

    def __init__(
        self,
        estimator: SupportsPredictProba,
        feature_columns: Sequence[str],
        *,
        name: str = "estimator",
        calibrator: Any = None,
    ) -> None:
        self.estimator = estimator
        self.feature_columns = list(feature_columns)
        self.name = name
        self.calibrator = calibrator

    def __call__(self, row: Row) -> float:
        import numpy as np

        x = np.array(
            [[_value(row, column) for column in self.feature_columns]], dtype=float
        )
        proba = self.estimator.predict_proba(np.nan_to_num(x))
        array = np.asarray(proba, dtype=float)
        probability = float(array[0, 1] if array.ndim == 2 and array.shape[1] > 1 else array[0])
        if self.calibrator is not None and getattr(self.calibrator, "fitted", False):
            probability = float(self.calibrator.transform(np.array([probability]))[0])
        return min(1.0, max(0.0, probability))
