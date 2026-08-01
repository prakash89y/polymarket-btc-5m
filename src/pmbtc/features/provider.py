"""The canonical feature provider — the only path from market data to features.

Before this module the system had two ways to produce a feature: the Module 5
live providers, which read the book and emitted values directly, and the Module 6
engine, which nothing in production called. Two implementations of the same idea
is one too many; whichever one drifts, the dataset is wrong and no test would
catch it.

So there is now exactly one. Live collection, archive replay, dataset export and
the training loader all go through :class:`EngineeredFeatureProvider`, which
builds a :class:`~pmbtc.features.base.FeatureContext` and hands it to the
deterministic engine. Nothing else engineers a feature.

The context is built from raw state only — books, tapes, quotes, the window open
price — never from anything derived. That is what makes live and replay
identical: replay reconstructs the same raw state from archived frames, and the
same raw state necessarily produces the same features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pmbtc.dataset.providers import SnapshotContext
from pmbtc.dataset.quality import QualityScorer
from pmbtc.dataset.schema import Observation, QualityFlag
from pmbtc.features.base import FeatureContext
from pmbtc.features.engine import FeatureEngine, FeatureVector
from pmbtc.features.families import all_features
from pmbtc.features.graph import build_graph
from pmbtc.features.matrix import feature_set_version
from pmbtc.features.registry import REGISTRY, FeatureTier
from pmbtc.logging_setup import get_logger

if TYPE_CHECKING:
    # Annotation-only: the live service imports this module, so importing the
    # feed classes at runtime would close the cycle.
    from pmbtc.live.binance import BinanceMarketFeed
    from pmbtc.live.book import BookPair
    from pmbtc.live.clob import PolymarketMarketFeed
    from pmbtc.live.tape import TradeTape

log = get_logger("pmbtc.features.provider")

#: Names of the engineered feature set — the canonical columns. Distinct from
#: the registry as a whole, which still contains the legacy Module 4/5 raw
#: measurements for backward compatibility with archived datasets.
ENGINEERED_FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted(f.name for f in all_features())
)


def canonical_feature_schema_version() -> str:
    """The one version string every component must agree on.

    Live collection stamps it on each snapshot, replay checks it, the dataset
    export records it in its manifest, and the trainer refuses data that does
    not match. A single function so there is nothing to keep in sync.
    """
    return feature_set_version(build_graph(all_features()))


def build_feature_context(
    *,
    now_ms: int,
    window_open_ms: int,
    settlement_ms: int,
    books: BookPair | None = None,
    pm_tape: TradeTape | None = None,
    ref_tape: TradeTape | None = None,
    ref_bid: float | None = None,
    ref_ask: float | None = None,
    window_open_price: float | None = None,
    tick_size: float | None = None,
    regime: dict[str, float] | None = None,
) -> FeatureContext:
    """Assemble a context from raw state.

    The single place a context is constructed, so live and replay cannot build
    one differently. Every argument is raw market state; nothing computed is
    accepted, because a computed input is an opportunity for the two paths to
    disagree.
    """
    values = dict(regime or {})
    if tick_size is not None:
        values["tick_size"] = tick_size
    return FeatureContext(
        now_ms=now_ms,
        window_open_ms=window_open_ms,
        settlement_ms=settlement_ms,
        books=books,
        pm_tape=pm_tape,
        ref_tape=ref_tape,
        ref_bid=ref_bid,
        ref_ask=ref_ask,
        window_open_price=window_open_price,
        regime=values,
    )


def context_from_feeds(
    context: SnapshotContext,
    clob: PolymarketMarketFeed | None,
    reference: BinanceMarketFeed | None,
    *,
    window_open_price: float | None = None,
    tick_size: float | None = None,
    regime: dict[str, float] | None = None,
) -> FeatureContext:
    """Build a feature context from live feed objects."""
    return build_feature_context(
        now_ms=context.snapshot_time_ms,
        window_open_ms=context.window_open_ms,
        settlement_ms=context.settlement_time_ms,
        books=clob.books if clob else None,
        pm_tape=clob.up_tape if clob else None,
        ref_tape=reference.tape if reference else None,
        ref_bid=reference.best_bid if reference else None,
        ref_ask=reference.best_ask if reference else None,
        window_open_price=window_open_price,
        tick_size=tick_size,
        regime=regime,
    )


class EngineeredFeatureProvider:
    """Runs the canonical feature pipeline and emits dataset observations.

    Implements the Module 4 ``FeatureProvider`` protocol, so the collector needs
    no special case: it sees one provider that happens to produce every feature.

    Provenance for each observation comes from the feature's own registry
    declaration — its tier decides the event time semantics, and its freshness
    budget decides staleness — rather than being invented here.
    """

    name = "engineered"
    source = "engineered"

    def __init__(
        self,
        scorer: QualityScorer,
        *,
        clob: PolymarketMarketFeed | None = None,
        reference: BinanceMarketFeed | None = None,
        engine: FeatureEngine | None = None,
    ) -> None:
        self.scorer = scorer
        self.clob = clob
        self.reference = reference
        self.engine = engine or FeatureEngine()
        #: Set by the service when the window opens.
        self.window_open_price: float | None = None
        self.tick_size: float | None = None
        self.regime: dict[str, float] = {}
        #: The most recent vector, so the caller can persist the matrix row.
        self.last_vector: FeatureVector | None = None

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear cross-pass state. Called between markets, never within one."""
        self.engine.reset()
        self.window_open_price = None
        self.last_vector = None

    async def collect(self, context: SnapshotContext) -> list[Observation]:
        feature_context = context_from_feeds(
            context,
            self.clob,
            self.reference,
            window_open_price=self.window_open_price,
            tick_size=self.tick_size,
            regime=self.regime,
        )
        vector = self.engine.compute(feature_context)
        self.last_vector = vector
        return self._to_observations(vector, context.snapshot_time_ms)

    # ------------------------------------------------------------------ #
    def _to_observations(
        self, vector: FeatureVector, instant: int
    ) -> list[Observation]:
        """Turn engine output into dataset observations with real provenance."""
        book_age = self._book_event_time(instant)
        ref_age = self._reference_event_time(instant)

        observations: list[Observation] = []
        for name in self.engine.feature_names:
            value = vector.values.get(name)
            spec = REGISTRY.get(name)
            # A derived or static feature is exact as of the snapshot instant;
            # a streamed one is only as fresh as the feed that produced it.
            if spec.tier in {FeatureTier.DERIVED, FeatureTier.STATIC}:
                event_time = instant
            elif spec.source == "binance_spot":
                event_time = ref_age
            else:
                event_time = book_age

            flags: tuple[QualityFlag, ...] = ()
            if name in vector.errors:
                flags = (QualityFlag.SOURCE_DEGRADED,)

            observations.append(
                self.scorer.build(
                    name=name,
                    value=value,
                    data_source=spec.source,
                    event_time_ms=min(event_time, instant),
                    observation_time_ms=instant,
                    snapshot_time_ms=instant,
                    extra_flags=flags,
                )
            )
        return observations

    def _book_event_time(self, instant: int) -> int:
        if self.clob and self.clob.books.up.last_update_ms:
            return self.clob.books.up.last_update_ms
        return instant

    def _reference_event_time(self, instant: int) -> int:
        if self.reference is None:
            return instant
        newest = max(self.reference.tape.last_trade_ms, self.reference.quote_time_ms)
        return newest or instant


def observations_to_values(observations: list[Observation]) -> dict[str, float | None]:
    """Recover the feature vector from stored observations.

    Used by replay and export verification: a snapshot written to the dataset
    must be reducible back to exactly the vector that produced it.
    """
    return {
        obs.name: obs.value
        for obs in observations
        if obs.name in set(ENGINEERED_FEATURE_NAMES)
    }


def vector_fingerprint(values: dict[str, float | None]) -> str:
    """Fingerprint a feature mapping the same way the engine does.

    Lets a dataset row, a replayed vector, and a live vector be compared with a
    single string equality rather than column by column.
    """
    import hashlib

    from pmbtc.features.engine import VALUE_PRECISION

    def render(name: str) -> str:
        value = values.get(name)
        return "null" if value is None else format(value, f".{VALUE_PRECISION}f")

    parts = [f"{name}={render(name)}" for name in sorted(values)]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def describe_pipeline() -> dict[str, Any]:
    """Summary used by the CLI and the integration checks."""
    return {
        "feature_schema_version": canonical_feature_schema_version(),
        "features": len(ENGINEERED_FEATURE_NAMES),
        "names": list(ENGINEERED_FEATURE_NAMES),
    }
