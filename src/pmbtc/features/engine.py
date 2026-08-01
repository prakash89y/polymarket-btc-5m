"""The feature engine: deterministic, incremental computation.

Determinism is the property everything else rests on. Three things guarantee it:

1. **Fixed compute order.** The graph's topological sort breaks ties by name, so
   the order is identical on every run and every machine.
2. **Pure features.** A feature reads only its context and returns a value. No
   clock, no network, no global state.
3. **No hidden accumulation.** Cross-pass state is confined to ``previous``,
   which is an explicit input rather than something a feature reaches for.

Together these mean that replaying archived frames reproduces live values
exactly — not approximately. :meth:`FeatureEngine.fingerprint` makes that
checkable in one comparison rather than column by column.

Incremental updates are correct-by-construction rather than best-effort: the
engine asks the graph which features a set of changed raw sources actually
affects, transitively, and recomputes exactly those. Anything not recomputed is
provably unaffected, so a stale value is impossible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from pmbtc.features.base import Feature, FeatureContext
from pmbtc.features.families import all_features
from pmbtc.features.graph import FeatureGraph, build_graph
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS

log = get_logger("pmbtc.features.engine")

features_computed = METRICS.counter(
    "pmbtc_features_computed_total", "Feature evaluations, by mode."
)
feature_errors = METRICS.counter("pmbtc_feature_errors_total", "Feature computation errors.")
feature_latency = METRICS.histogram(
    "pmbtc_feature_pass_ms", "Wall time for one feature computation pass."
)

#: Values are rounded before storage and fingerprinting. Float arithmetic is
#: deterministic for a fixed operation order, but rounding removes any residual
#: sensitivity to library versions in transcendental functions (erf, log, sqrt)
#: and makes the fingerprint stable across platforms.
VALUE_PRECISION = 10


@dataclass
class FeatureVector:
    """One computed pass: values plus enough provenance to audit them."""

    now_ms: int
    values: dict[str, float | None]
    #: Features that were recomputed this pass, as opposed to carried forward.
    recomputed: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    pass_index: int = 0

    def __len__(self) -> int:
        return len(self.values)

    @property
    def populated(self) -> int:
        return sum(1 for v in self.values.values() if v is not None)

    @property
    def coverage(self) -> float:
        return self.populated / len(self.values) if self.values else 0.0

    def fingerprint(self) -> str:
        """Stable hash of the values. Equal fingerprints mean equal outputs."""
        parts = [
            f"{name}={'null' if value is None else format(value, f'.{VALUE_PRECISION}f')}"
            for name, value in sorted(self.values.items())
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "now_ms": self.now_ms,
            "pass_index": self.pass_index,
            "coverage": round(self.coverage, 4),
            "recomputed": list(self.recomputed),
            "errors": self.errors,
            "fingerprint": self.fingerprint(),
            "values": self.values,
        }


class FeatureEngine:
    """Computes the feature set from a context, incrementally and deterministically."""

    def __init__(self, features: list[Feature] | None = None) -> None:
        self.graph: FeatureGraph = build_graph(features or all_features())
        self._values: dict[str, float | None] = {name: None for name in self.graph.order}
        self._previous: dict[str, float | None] = {}
        self._last_now_ms: int = 0
        self._passes: int = 0

    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.graph.order

    def reset(self) -> None:
        """Clear all state. Called between markets, never within one."""
        self._values = {name: None for name in self.graph.order}
        self._previous = {}
        self._last_now_ms = 0
        self._passes = 0

    # ------------------------------------------------------------------ #
    def compute(
        self,
        context: FeatureContext,
        *,
        changed_sources: set[str] | None = None,
    ) -> FeatureVector:
        """Run one pass.

        ``changed_sources`` names the raw inputs that moved since the last pass.
        When omitted every feature is recomputed, which is the right default for
        a snapshot; the live loop passes the changed set so a book tick does not
        drag the whole regime family along with it.
        """
        with feature_latency.time():
            targets = (
                self.graph.order
                if changed_sources is None
                else self.graph.affected_by(changed_sources)
            )
            # Carry forward the previous pass so cross-pass features (velocity,
            # stability) see a genuine prior rather than this pass's own value.
            context.previous = dict(self._previous)
            context.elapsed_ms = (
                context.now_ms - self._last_now_ms if self._last_now_ms else 0
            )
            context.computed = dict(self._values)

            errors: dict[str, str] = {}
            for name in targets:
                feature = self.graph.features[name]
                try:
                    value = feature.compute(context)
                except Exception as exc:
                    # One broken feature must not void the pass; it is recorded
                    # as missing and named, which is strictly more useful than
                    # an exception that loses the other ninety.
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    feature_errors.inc(feature=name)
                    log.warning("feature.error", feature=name, error=errors[name])
                    value = None
                value = _quantise(value)
                context.computed[name] = value
                self._values[name] = value

            features_computed.inc(
                len(targets), mode="full" if changed_sources is None else "incremental"
            )
            self._previous = dict(self._values)
            self._last_now_ms = context.now_ms
            self._passes += 1

            return FeatureVector(
                now_ms=context.now_ms,
                values=dict(self._values),
                recomputed=tuple(targets),
                errors=errors,
                pass_index=self._passes,
            )

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        by_family: dict[str, int] = {}
        for feature in self.graph.features.values():
            key = feature.spec.group
            by_family[key] = by_family.get(key, 0) + 1
        return {
            "features": len(self.graph),
            "raw_sources": list(self.graph.raw_sources),
            "by_group": by_family,
            "passes": self._passes,
        }


def _quantise(value: float | None) -> float | None:
    """Round to a fixed precision, and reject non-finite values.

    An infinite or NaN feature silently poisons every model that sees it, so it
    is converted to a recorded missing value instead.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return round(result, VALUE_PRECISION)
