"""Shadow validation: replay a trained model across history, in order.

The last check before paper trading. It answers four questions that offline
metrics cannot:

**Are predictions deterministic?** The same row, predicted twice, must give
bit-identical output. A model that does not is unusable — its backtest cannot be
reproduced and its live behaviour cannot be audited.

**Are probabilities reproducible from a reloaded artifact?** Predicting through
the saved pickle must match predicting through the in-memory object. This
catches the class of bug where something stateful failed to serialise.

**Is calibration stable over time?** A model can be well calibrated overall and
badly calibrated in the most recent third — which is the only third that
resembles tomorrow. Calibration is therefore measured per chronological block,
not just in aggregate.

**Is inference fast enough?** On a 300-second instrument with a 20-second safety
window, a model taking 500 ms per prediction is fine and one taking 30 seconds
is not. Measured, not assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pmbtc.logging_setup import get_logger
from pmbtc.models.calibration import EvaluationMetrics, evaluate

log = get_logger("pmbtc.models.shadow")


@dataclass
class ShadowResult:
    """Outcome of a chronological replay."""

    samples: int = 0
    deterministic: bool = False
    reproducible_from_artifact: bool = False
    overall: EvaluationMetrics | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    calibration_drift: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.deterministic
            and self.reproducible_from_artifact
            and not self.failures
            and self.samples > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "deterministic": self.deterministic,
            "reproducible_from_artifact": self.reproducible_from_artifact,
            "overall": self.overall.as_dict() if self.overall else {},
            "blocks": self.blocks,
            "calibration_drift": round(self.calibration_drift, 6),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_max_ms": round(self.latency_max_ms, 3),
            "failures": self.failures,
            "passed": self.passed,
        }

    def summary(self) -> str:
        if self.passed:
            return (
                f"shadow OK: n={self.samples} ece={self.overall.ece:.4f} "
                f"drift={self.calibration_drift:.4f} "
                f"p95={self.latency_p95_ms:.2f}ms"
                if self.overall
                else "shadow OK"
            )
        return "shadow FAILED: " + "; ".join(self.failures or ["unknown"])


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def run_shadow_validation(
    predict: Any,
    x: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    *,
    reloaded_predict: Any = None,
    blocks: int = 4,
    max_latency_ms: float = 250.0,
    max_calibration_drift: float = 0.05,
) -> ShadowResult:
    """Replay chronologically and verify the four properties."""
    result = ShadowResult(samples=len(y))
    if len(y) == 0:
        result.failures.append("no samples to replay")
        return result

    order = np.argsort(timestamps, kind="stable")
    x_ordered, y_ordered = x[order], y[order]

    # --- determinism and latency, one row at a time ------------------- #
    latencies: list[float] = []
    predictions = np.empty(len(y_ordered), dtype=float)
    for i in range(len(y_ordered)):
        row = x_ordered[i : i + 1]
        start = time.perf_counter()
        value = float(np.ravel(predict(row))[0])
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions[i] = value

    repeat = np.array(
        [float(np.ravel(predict(x_ordered[i : i + 1]))[0]) for i in range(len(y_ordered))]
    )
    result.deterministic = bool(np.array_equal(predictions, repeat))
    if not result.deterministic:
        differing = int(np.sum(predictions != repeat))
        result.failures.append(
            f"predictions are not deterministic ({differing} of {len(predictions)} differ)"
        )

    # --- reproducibility from the saved artifact ----------------------- #
    if reloaded_predict is None:
        result.reproducible_from_artifact = True
    else:
        reloaded = np.ravel(reloaded_predict(x_ordered)).astype(float)
        result.reproducible_from_artifact = bool(
            np.array_equal(predictions, reloaded)
        )
        if not result.reproducible_from_artifact:
            worst = float(np.max(np.abs(predictions - reloaded)))
            result.failures.append(
                f"reloaded artifact disagrees with the in-memory model (max diff {worst:.3g})"
            )

    result.latency_p50_ms = _percentile(latencies, 0.50)
    result.latency_p95_ms = _percentile(latencies, 0.95)
    result.latency_max_ms = max(latencies)
    if result.latency_p95_ms > max_latency_ms:
        result.failures.append(
            f"p95 inference latency {result.latency_p95_ms:.1f}ms exceeds "
            f"{max_latency_ms:.0f}ms"
        )

    # --- calibration overall and per chronological block --------------- #
    result.overall = evaluate(y_ordered, predictions)
    block_size = max(1, len(y_ordered) // max(1, blocks))
    eces: list[float] = []
    for index in range(0, len(y_ordered), block_size):
        chunk_y = y_ordered[index : index + block_size]
        chunk_p = predictions[index : index + block_size]
        if len(chunk_y) < 10:
            continue
        metrics = evaluate(chunk_y, chunk_p)
        eces.append(metrics.ece)
        result.blocks.append(
            {
                "start": int(timestamps[order][index]),
                "n": len(chunk_y),
                "brier": round(metrics.brier, 6),
                "ece": round(metrics.ece, 6),
                "accuracy": round(metrics.accuracy, 6),
            }
        )

    # Drift is the spread of per-block ECE: a model that is well calibrated
    # early and badly calibrated late is not stable, however good the average.
    result.calibration_drift = float(max(eces) - min(eces)) if len(eces) > 1 else 0.0
    if result.calibration_drift > max_calibration_drift:
        result.failures.append(
            f"calibration drifts by {result.calibration_drift:.4f} across blocks "
            f"(limit {max_calibration_drift:.4f})"
        )

    log.info("shadow.completed", **{k: v for k, v in result.as_dict().items() if k != "blocks"})
    return result
