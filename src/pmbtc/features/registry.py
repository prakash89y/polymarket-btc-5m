"""Feature registry: tiers, provenance, and reproducibility policy.

Three rules are enforced here rather than left to convention:

**Every feature declares its freshness tier.** A model that cannot tell a
5-millisecond order-book imbalance from a 24-hour-old on-chain metric will
happily weight them equally. The tier is carried on every observation into the
dataset, so the training pipeline can filter, weight, or bucket by it.

**No anonymous features.** A feature that is not registered cannot be recorded.
That sounds bureaucratic until the first time a column appears in a training set
and nobody can say where it came from, whether it was live or backfilled, or
whether it existed at decision time.

**Every feature declares whether it can be reconstructed later.** A live feature
computed from a WebSocket stream is gone the instant it passes unless the stream
itself is archived. Such a feature is either archived at collection time or
excluded from training — never silently trusted, because a backtest cannot
reproduce it and a model trained on it cannot be validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pmbtc.exceptions import ConfigError


class FeatureTier(StrEnum):
    """How fresh a feature is, by construction.

    REAL_TIME
        Sub-second. Streamed from a live socket; reflects the market as of
        milliseconds ago. Order book, trade tape.
    NEAR_REAL_TIME
        Seconds. Polled fast, or streamed with aggregation. 1s bars, funding.
    DELAYED
        Tens of seconds to minutes. Cached REST endpoints. Gamma quotes live
        here — measured at 29-72 seconds stale, which is why they are metadata
        and not a trading signal.
    DERIVED
        Computed from other features or from the clock. Exact, zero latency,
        but only as fresh as its inputs.
    STATIC
        Fixed for the life of a market: tick size, settlement provider, tie
        rule. Never stale, never a signal on its own.
    """

    REAL_TIME = "real_time"
    NEAR_REAL_TIME = "near_real_time"
    DELAYED = "delayed"
    DERIVED = "derived"
    STATIC = "static"

    @property
    def is_tradeable_signal(self) -> bool:
        """Whether a feature at this tier may drive an entry decision.

        DELAYED features can inform regime context; they must never be the
        thing that triggers a trade on a 300-second instrument.
        """
        return self in {FeatureTier.REAL_TIME, FeatureTier.NEAR_REAL_TIME, FeatureTier.DERIVED}


class ReproducibilityPolicy(StrEnum):
    """How a feature can be reconstructed for a backtest.

    REST_REPLAYABLE
        Re-fetchable from a public historical endpoint at any time.
    ARCHIVED_STREAM
        Only exists because we archived the raw stream. Reproducible exactly as
        long as the archive is kept.
    DERIVED_FROM_ARCHIVE
        Computable from other archived data.
    NOT_REPRODUCIBLE
        Cannot be reconstructed. Excluded from training by default -- see
        :meth:`FeatureSpec.usable_for_training`.
    """

    REST_REPLAYABLE = "rest_replayable"
    ARCHIVED_STREAM = "archived_stream"
    DERIVED_FROM_ARCHIVE = "derived_from_archive"
    NOT_REPRODUCIBLE = "not_reproducible"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The declaration every feature must have before it can be recorded."""

    name: str
    tier: FeatureTier
    source: str
    description: str
    reproducibility: ReproducibilityPolicy
    #: Priority group from the agreed taxonomy: microstructure / short_term /
    #: regime. Controls selection budget in Module 6, never importance.
    group: str = "microstructure"
    #: Age at which this feature stops carrying information, in milliseconds.
    #: Used to compute freshness and to flag staleness at snapshot time.
    freshness_budget_ms: int = 5_000

    # --- documentation contract (Module 6) ----------------------------- #
    #: The mathematical definition, written so a reader can reimplement it
    #: without reading the code. Required for every engineered feature.
    formula: str = ""
    #: Names this feature is computed from: other features, or ``raw:<key>``
    #: for direct inputs. Drives the dependency graph and the compute order.
    inputs: tuple[str, ...] = ()
    #: Unit of the value, so nobody has to guess whether it is bps or a ratio.
    units: str = "dimensionless"
    #: What it is *for* — the hypothesis it encodes. A feature nobody can state
    #: a purpose for is a feature that should not be in the model.
    purpose: str = ""
    #: Minimum observation quality for this feature to be considered usable.
    min_quality: float = 0.0

    @property
    def documented(self) -> bool:
        """Whether this feature meets the documentation contract."""
        return bool(self.formula and self.purpose and self.units)

    @property
    def usable_for_training(self) -> bool:
        """A feature that cannot be reconstructed cannot be validated."""
        return self.reproducibility is not ReproducibilityPolicy.NOT_REPRODUCIBLE

    def freshness(self, age_ms: int) -> float:
        """Normalised freshness in ``[0, 1]``: 1.0 brand new, 0.0 past budget.

        Recorded on every observation so the model always knows how current
        each input was, without having to reason about absolute timestamps.
        """
        if self.freshness_budget_ms <= 0:
            return 1.0
        if age_ms <= 0:
            return 1.0
        return max(0.0, 1.0 - age_ms / self.freshness_budget_ms)


class FeatureRegistry:
    """The set of features this system is allowed to record."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ConfigError(
                f"Feature {spec.name!r} is already registered with a different declaration",
                context={"existing": str(existing), "new": str(spec)},
            )
        self._specs[spec.name] = spec
        return spec

    def register_many(self, specs: list[FeatureSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ConfigError(
                f"Feature {name!r} is not registered; anonymous features are not permitted",
                context={"registered": len(self._specs)},
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def by_tier(self, tier: FeatureTier) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self._specs.values() if s.tier is tier)

    def by_group(self, group: str) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self._specs.values() if s.group == group)

    def trainable(self) -> tuple[FeatureSpec, ...]:
        """Features that a backtest could reconstruct."""
        return tuple(s for s in self._specs.values() if s.usable_for_training)

    def unreproducible(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self._specs.values() if not s.usable_for_training)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for spec in self._specs.values():
            counts[spec.tier.value] = counts.get(spec.tier.value, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._specs)


#: Process-wide registry. Providers register their features at import time, so
#: the declaration lives next to the code that produces the value.
REGISTRY = FeatureRegistry()


def register(
    name: str,
    tier: FeatureTier,
    source: str,
    description: str,
    reproducibility: ReproducibilityPolicy,
    group: str = "microstructure",
    freshness_budget_ms: int = 5_000,
) -> FeatureSpec:
    """Convenience wrapper around :meth:`FeatureRegistry.register`."""
    return REGISTRY.register(
        FeatureSpec(
            name=name,
            tier=tier,
            source=source,
            description=description,
            reproducibility=reproducibility,
            group=group,
            freshness_budget_ms=freshness_budget_ms,
        )
    )
