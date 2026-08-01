"""Latency budgets and observation quality.

A feature is only useful if it could have been known in time to act on. This
module turns that into an enforced rule: every source declares a latency budget
and a staleness budget, and readings that miss them are flagged -- and, past the
hard budget, excluded from training by default rather than quietly averaged in.

Flagging rather than dropping is deliberate. A dropped row is invisible; a
flagged row is a filter away from being dropped and a query away from being
counted. "How often is CoinGlass late?" must be answerable from the dataset
itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from pmbtc.dataset.schema import Observation, QualityFlag
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.dataset.quality")


@dataclass(frozen=True, slots=True)
class SourceBudget:
    """Latency and staleness limits for one data source.

    ``max_latency_ms``
        How long after an event we may still learn of it and call the reading
        timely. Exceeded -> ``DELAYED``.
    ``hard_latency_ms``
        Past this the reading could not plausibly have informed a decision.
        Exceeded -> ``OUT_OF_BUDGET`` and the value is excluded from training by
        the default filter.
    ``max_staleness_ms``
        How old the underlying event may be relative to the snapshot instant.
        Exceeded -> ``STALE``.
    """

    name: str
    max_latency_ms: int
    hard_latency_ms: int
    max_staleness_ms: int
    confidence: float = 1.0


#: Budgets scale with how fast the underlying series actually moves. An order
#: book quote is worthless a second late; the Fear & Greed index updates daily
#: and being an hour behind changes nothing.
DEFAULT_BUDGETS: dict[str, SourceBudget] = {
    "polymarket_gamma": SourceBudget("polymarket_gamma", 2_000, 15_000, 60_000, 0.9),
    "polymarket_clob": SourceBudget("polymarket_clob", 500, 3_000, 5_000, 1.0),
    "binance_spot": SourceBudget("binance_spot", 1_000, 5_000, 10_000, 1.0),
    "binance_futures": SourceBudget("binance_futures", 1_000, 5_000, 10_000, 1.0),
    "coinbase": SourceBudget("coinbase", 1_500, 6_000, 15_000, 0.95),
    "bybit": SourceBudget("bybit", 1_500, 6_000, 15_000, 0.95),
    "chainlink": SourceBudget("chainlink", 2_000, 10_000, 30_000, 1.0),
    "coinglass": SourceBudget("coinglass", 30_000, 120_000, 300_000, 0.8),
    "derived": SourceBudget("derived", 100, 1_000, 1_000, 1.0),
    # Regime-tier sources: slow by nature, and that is fine.
    "fear_greed": SourceBudget("fear_greed", 3_600_000, 172_800_000, 172_800_000, 0.6),
    "glassnode": SourceBudget("glassnode", 3_600_000, 172_800_000, 172_800_000, 0.6),
    "cryptoquant": SourceBudget("cryptoquant", 3_600_000, 172_800_000, 172_800_000, 0.6),
    "macro_calendar": SourceBudget("macro_calendar", 3_600_000, 172_800_000, 172_800_000, 0.7),
    "news": SourceBudget("news", 60_000, 600_000, 3_600_000, 0.5),
    "twitter": SourceBudget("twitter", 60_000, 600_000, 3_600_000, 0.4),
}

#: Used for a source nobody declared: strict, so an unregistered source is
#: conspicuous rather than silently trusted.
FALLBACK_BUDGET = SourceBudget("unknown", 1_000, 5_000, 10_000, 0.5)


class QualityScorer:
    """Applies budgets to raw readings, producing flagged observations."""

    def __init__(self, budgets: dict[str, SourceBudget] | None = None) -> None:
        self.budgets = dict(DEFAULT_BUDGETS)
        if budgets:
            self.budgets.update(budgets)

    def budget_for(self, source: str) -> SourceBudget:
        return self.budgets.get(source, FALLBACK_BUDGET)

    # ------------------------------------------------------------------ #
    def build(
        self,
        *,
        name: str,
        value: float | None,
        data_source: str,
        event_time_ms: int,
        observation_time_ms: int,
        snapshot_time_ms: int,
        extra_flags: tuple[QualityFlag, ...] = (),
        source_confidence: float | None = None,
    ) -> Observation:
        """Construct an observation with its quality flags already resolved."""
        budget = self.budget_for(data_source)
        flags: list[QualityFlag] = [f for f in extra_flags if f is not QualityFlag.OK]

        if value is None:
            flags.append(QualityFlag.MISSING)
        else:
            latency = observation_time_ms - event_time_ms
            if latency > budget.hard_latency_ms:
                flags.append(QualityFlag.OUT_OF_BUDGET)
            elif latency > budget.max_latency_ms:
                flags.append(QualityFlag.DELAYED)

            staleness = snapshot_time_ms - event_time_ms
            if staleness > budget.max_staleness_ms:
                flags.append(QualityFlag.STALE)

        confidence = budget.confidence if source_confidence is None else source_confidence
        return Observation(
            name=name,
            value=value,
            data_source=data_source,
            event_time_ms=event_time_ms,
            observation_time_ms=observation_time_ms,
            flags=tuple(dict.fromkeys(flags)) or (QualityFlag.OK,),
            source_confidence=confidence,
        )

    def missing(self, name: str, data_source: str, snapshot_time_ms: int) -> Observation:
        """A feature that could not be produced at all.

        Recorded explicitly rather than omitted: a column that is absent looks
        the same as a column that was never requested, and the difference
        matters when auditing coverage.
        """
        return Observation(
            name=name,
            value=None,
            data_source=data_source,
            event_time_ms=snapshot_time_ms,
            observation_time_ms=snapshot_time_ms,
            flags=(QualityFlag.MISSING,),
            source_confidence=0.0,
        )


@dataclass(frozen=True, slots=True)
class QualityFilter:
    """Training-time exclusion rules."""

    min_snapshot_quality: float = 0.5
    min_coverage: float = 0.6
    exclude_flags: frozenset[QualityFlag] = frozenset({QualityFlag.OUT_OF_BUDGET})

    def accepts_observation(self, obs: Observation) -> bool:
        return not (set(obs.flags) & self.exclude_flags)

    def accepts_snapshot(self, quality: float, coverage: float) -> bool:
        return quality >= self.min_snapshot_quality and coverage >= self.min_coverage
