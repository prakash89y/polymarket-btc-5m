"""Dataset record types.

The unit of the dataset is **one 5-minute market**. Around it hang a fixed
timeline of feature snapshots and, once the venue resolves it, exactly one
label.

Three rules are encoded in the types rather than left to discipline:

1. **An observation knows when it happened and when we saw it.** ``event_time``
   and ``observation_time`` are separate fields, never conflated. Their
   difference is the latency, and a feature that arrived too late to have been
   usable is not silently accepted.

2. **Snapshots are immutable and horizon-addressed.** A snapshot is identified
   by ``(condition_id, horizon_seconds)``. Recording the same pair twice is an
   error, not an update, so history cannot be rewritten by a later run.

3. **Labels live on the market, not on the snapshot.** A snapshot physically
   cannot carry the outcome, because the outcome did not exist when the snapshot
   was taken. The join happens at export time, in one place, under the leakage
   guard.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pmbtc.constants import Outcome, SettlementSource

#: Bump when the meaning or set of feature columns changes. Recorded in every
#: export manifest so a model can always be traced to the schema it was
#: trained on.
FEATURE_SCHEMA_VERSION = "4.0"
DATASET_RECORD_VERSION = "4.0"


class QualityFlag(StrEnum):
    """Why an observation is less than perfect.

    Flags are additive: an observation can be both stale and interpolated. The
    training pipeline filters on these, so they are a fixed vocabulary.
    """

    OK = "ok"
    MISSING = "missing"
    STALE = "stale"
    ESTIMATED = "estimated"
    INTERPOLATED = "interpolated"
    DELAYED = "delayed"
    OUT_OF_BUDGET = "out_of_budget"
    SOURCE_DEGRADED = "source_degraded"


class Observation(BaseModel):
    """One measured value, with full provenance.

    Every feature in the dataset is one of these. The provenance is not
    decoration: ``event_time_ms`` is what the leakage guard checks, and
    ``latency_ms`` is what decides whether the value was actually available in
    time to have been acted on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float | None
    data_source: str
    #: When the underlying event occurred at the source.
    event_time_ms: int
    #: When this process received it.
    observation_time_ms: int
    flags: tuple[QualityFlag, ...] = (QualityFlag.OK,)
    #: How much the source itself is trusted, independent of this reading.
    source_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def latency_ms(self) -> int:
        """Delay between the event and our knowledge of it.

        Can legitimately be negative by a few milliseconds when a venue's clock
        leads ours; the quality scorer treats that as noise rather than a
        paradox, but a large negative value means a clock problem.
        """
        return self.observation_time_ms - self.event_time_ms

    @property
    def is_usable(self) -> bool:
        return self.value is not None and QualityFlag.OUT_OF_BUDGET not in self.flags

    @property
    def quality_score(self) -> float:
        """Single number in [0, 1] combining flags and source confidence.

        Deliberately blunt. Its job is to make "exclude the bottom decile" a
        one-line filter, not to be a calibrated probability of correctness.
        """
        if self.value is None:
            return 0.0
        penalties = {
            QualityFlag.OUT_OF_BUDGET: 1.0,
            QualityFlag.MISSING: 1.0,
            QualityFlag.STALE: 0.5,
            QualityFlag.ESTIMATED: 0.35,
            QualityFlag.INTERPOLATED: 0.3,
            QualityFlag.DELAYED: 0.2,
            QualityFlag.SOURCE_DEGRADED: 0.25,
        }
        worst = max((penalties.get(flag, 0.0) for flag in self.flags), default=0.0)
        return max(0.0, (1.0 - worst) * self.source_confidence)


class FeatureSnapshot(BaseModel):
    """All features for one market at one point on the countdown.

    ``horizon_seconds`` is seconds *before settlement*: 300 is the market open,
    1 is the last second. The snapshot's nominal timestamp is
    ``settlement_time_ms - horizon_seconds * 1000``; ``captured_at_ms`` is when
    the collector actually ran, and the difference between them is scheduling
    jitter, which is recorded rather than hidden.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_id: str
    slug: str = ""
    horizon_seconds: int
    #: The instant this snapshot represents. Nothing in it may postdate this.
    snapshot_time_ms: int
    captured_at_ms: int
    settlement_time_ms: int
    observations: tuple[Observation, ...] = ()
    #: Clock health at capture time; a snapshot taken on a bad clock is suspect
    #: even if every observation looks fine.
    clock_offset_ms: float = 0.0
    clock_status: str = "unknown"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, int]:
        """Identity. Recording this twice is an error, never an update."""
        return (self.condition_id, self.horizon_seconds)

    @property
    def scheduling_jitter_ms(self) -> int:
        return self.captured_at_ms - self.snapshot_time_ms

    @property
    def values(self) -> dict[str, float | None]:
        return {obs.name: obs.value for obs in self.observations}

    @property
    def missing_count(self) -> int:
        return sum(1 for obs in self.observations if obs.value is None)

    @property
    def coverage(self) -> float:
        if not self.observations:
            return 0.0
        return 1.0 - self.missing_count / len(self.observations)

    @property
    def quality_score(self) -> float:
        """Mean observation quality; 0 when there is nothing to score."""
        if not self.observations:
            return 0.0
        return sum(obs.quality_score for obs in self.observations) / len(self.observations)

    @property
    def mean_latency_ms(self) -> float:
        usable = [obs.latency_ms for obs in self.observations if obs.value is not None]
        return sum(usable) / len(usable) if usable else 0.0

    def flat_record(self) -> dict[str, Any]:
        """Column-per-feature row, for the tabular exports."""
        row: dict[str, Any] = {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "horizon_seconds": self.horizon_seconds,
            "snapshot_time_ms": self.snapshot_time_ms,
            "captured_at_ms": self.captured_at_ms,
            "settlement_time_ms": self.settlement_time_ms,
            "scheduling_jitter_ms": self.scheduling_jitter_ms,
            "clock_offset_ms": self.clock_offset_ms,
            "clock_status": self.clock_status,
            "snapshot_quality": self.quality_score,
            "snapshot_coverage": self.coverage,
            "mean_latency_ms": self.mean_latency_ms,
            "feature_schema_version": self.feature_schema_version,
        }
        for obs in self.observations:
            row[f"f_{obs.name}"] = obs.value
            row[f"q_{obs.name}"] = obs.quality_score
            row[f"lat_{obs.name}"] = obs.latency_ms
        return row


class MarketRecord(BaseModel):
    """Everything known about one market: metadata, timing, and the label.

    The label fields are ``None`` until the venue resolves the market. That is a
    real state, not an error -- resolution lags settlement -- and the collector
    backfills them later without touching any snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- identity ---
    condition_id: str
    market_id: str = ""
    event_id: str = ""
    series_id: str = ""
    series_slug: str = ""
    slug: str = ""
    question: str = ""

    # --- timing (all UTC epoch millis, from the synchronised clock) ---
    discovery_time_ms: int
    open_time_ms: int
    #: Last instant an order may be submitted under the safety window. Derived,
    #: but stored explicitly because the safety window is configurable and a
    #: dataset must record the rules it was collected under.
    lock_time_ms: int
    settlement_time_ms: int

    # --- settlement ---
    settlement_provider: SettlementSource
    settlement_spec_hash: str
    settlement_parser_version: str = ""
    trading_pair: str = ""
    tie_rule: str = ""

    # --- label: official Polymarket outcome only ---
    official_outcome: Outcome | None = None
    #: Final settled probability of the winning side as published by the venue.
    settlement_probability: float | None = None
    yes_final_price: float | None = None
    no_final_price: float | None = None
    resolved_at_ms: int | None = None
    resolution_source: str = ""

    dataset_record_version: str = DATASET_RECORD_VERSION

    @property
    def is_labelled(self) -> bool:
        return self.official_outcome is not None

    @property
    def label(self) -> int | None:
        """Binary target: 1 when the market resolved Up."""
        if self.official_outcome is None:
            return None
        return 1 if self.official_outcome is Outcome.UP else 0

    @property
    def duration_seconds(self) -> int:
        return (self.settlement_time_ms - self.open_time_ms) // 1000

    def flat_record(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["label"] = self.label
        row["is_labelled"] = self.is_labelled
        return row
