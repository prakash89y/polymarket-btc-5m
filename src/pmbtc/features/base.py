"""Feature definitions and the computation context.

A feature is a *declaration* plus a pure function. The declaration carries the
formula, inputs, units, freshness budget, and reproducibility policy; the
function turns a context into a number. Nothing else is permitted to produce a
value that reaches the dataset.

Two properties are enforced structurally:

**Purity.** ``compute`` receives everything it may read through the context and
returns a value. It may not call the clock, touch the network, or consult global
state. That is what makes replay bit-for-bit identical to live: given the same
context, there is nothing left that could differ.

**Declared dependencies.** A feature names its inputs, so the engine can order
the computation, detect cycles, and recompute only what a new tick actually
affected. A feature that reads something it did not declare will see ``None``
rather than a stale value, which fails loudly instead of quietly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pmbtc.features.registry import (
    REGISTRY,
    FeatureSpec,
    FeatureTier,
    ReproducibilityPolicy,
)

if TYPE_CHECKING:
    # Annotation-only. Importing these at runtime would create a cycle:
    # pmbtc.live imports the service, which imports the canonical feature
    # provider, which imports this module.
    from pmbtc.live.book import BookPair, OrderBook
    from pmbtc.live.tape import TradeTape

#: Prefix marking a raw input rather than another feature.
RAW = "raw:"


@dataclass
class FeatureContext:
    """Everything a feature is allowed to read.

    ``now_ms`` is the *snapshot instant*, never the wall clock. Every windowed
    computation is bounded by it, so a feature physically cannot see a trade or
    a book update that arrived after the moment it claims to describe.
    """

    now_ms: int
    window_open_ms: int
    settlement_ms: int
    #: Polymarket books for both outcome tokens.
    books: BookPair | None = None
    #: Executed trades on the Up token.
    pm_tape: TradeTape | None = None
    #: Reference venue trades (Binance).
    ref_tape: TradeTape | None = None
    #: Reference top-of-book.
    ref_bid: float | None = None
    ref_ask: float | None = None
    #: Price of the reference series at the window open.
    window_open_price: float | None = None
    #: Slow-moving context: funding, open interest, sentiment.
    regime: dict[str, float] = field(default_factory=dict)
    #: Values computed earlier in this pass, keyed by feature name.
    computed: dict[str, float | None] = field(default_factory=dict)
    #: Values from the previous pass, for change/velocity features.
    previous: dict[str, float | None] = field(default_factory=dict)
    #: Millis between this pass and the previous one; 0 on the first pass.
    elapsed_ms: int = 0

    # ------------------------------------------------------------------ #
    @property
    def up_book(self) -> OrderBook | None:
        return self.books.up if self.books else None

    @property
    def seconds_to_settlement(self) -> float:
        return (self.settlement_ms - self.now_ms) / 1000.0

    @property
    def window_progress(self) -> float:
        span = self.settlement_ms - self.window_open_ms
        return (self.now_ms - self.window_open_ms) / span if span > 0 else 0.0

    def value(self, name: str) -> float | None:
        """Read a previously computed feature. Undeclared reads return None."""
        return self.computed.get(name)

    def prior(self, name: str) -> float | None:
        return self.previous.get(name)


class Feature(ABC):
    """One engineered feature: a declaration plus a pure computation."""

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.spec.inputs

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Declared inputs that are other features, not raw sources."""
        return tuple(i for i in self.spec.inputs if not i.startswith(RAW))

    @property
    def raw_inputs(self) -> frozenset[str]:
        """Raw source keys this feature reads, without the ``raw:`` prefix."""
        return frozenset(i[len(RAW) :] for i in self.spec.inputs if i.startswith(RAW))

    @abstractmethod
    def compute(self, context: FeatureContext) -> float | None:
        """Return the value, or None when it cannot be computed.

        Returning None is a first-class outcome, not an error: an empty book has
        no imbalance, and inventing a zero would be a lie the model would learn.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Feature {self.name}>"


def define(
    name: str,
    *,
    tier: FeatureTier,
    source: str,
    formula: str,
    inputs: tuple[str, ...],
    units: str,
    purpose: str,
    reproducibility: ReproducibilityPolicy = ReproducibilityPolicy.ARCHIVED_STREAM,
    group: str = "microstructure",
    freshness_budget_ms: int = 2_000,
    description: str = "",
) -> FeatureSpec:
    """Declare and register a feature spec.

    Every argument that documents the feature is required by signature rather
    than by convention, so an undocumented feature cannot be declared at all.
    """
    return REGISTRY.register(
        FeatureSpec(
            name=name,
            tier=tier,
            source=source,
            description=description or purpose,
            reproducibility=reproducibility,
            group=group,
            freshness_budget_ms=freshness_budget_ms,
            formula=formula,
            inputs=inputs,
            units=units,
            purpose=purpose,
        )
    )


class ComputedFeature(Feature):
    """A feature backed by a plain function, for the common case."""

    def __init__(self, spec: FeatureSpec, fn: Any) -> None:
        super().__init__(spec)
        self._fn = fn

    def compute(self, context: FeatureContext) -> float | None:
        return self._fn(context)


def feature(
    name: str,
    *,
    tier: FeatureTier,
    source: str,
    formula: str,
    inputs: tuple[str, ...],
    units: str,
    purpose: str,
    reproducibility: ReproducibilityPolicy = ReproducibilityPolicy.ARCHIVED_STREAM,
    group: str = "microstructure",
    freshness_budget_ms: int = 2_000,
) -> Any:
    """Decorator turning a function into a registered :class:`Feature`."""

    def wrap(fn: Any) -> ComputedFeature:
        spec = define(
            name,
            tier=tier,
            source=source,
            formula=formula,
            inputs=inputs,
            units=units,
            purpose=purpose,
            reproducibility=reproducibility,
            group=group,
            freshness_budget_ms=freshness_budget_ms,
            description=(fn.__doc__ or purpose).strip().split("\n")[0],
        )
        return ComputedFeature(spec, fn)

    return wrap


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Division that returns None rather than raising or producing infinity.

    Used everywhere a ratio is formed. An infinite feature poisons a model
    silently; a missing one is recorded and filterable.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator
