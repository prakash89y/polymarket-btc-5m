"""The canonical settlement specification and its evidence model.

Every fact in a :class:`SettlementSpecification` carries a
:class:`FieldEvidence` recording *where it came from* and *how certain we are*.
That is the whole point: "the provider is Chainlink" is worthless without
"...because the structured ``resolutionSource`` field said so **and** the prose
independently agreed".

Confidence rules
----------------
Structured metadata corroborated by prose  -> 1.0  (tradeable)
Structured metadata alone                  -> 0.9  (not tradeable under the
                                                    default gate of 1.0)
Prose alone                                -> 0.7  (never tradeable)
Structured and prose disagree              -> 0.0  (hard reject, conflict)

The gate in ``settlement.min_detection_confidence`` defaults to 1.0, so only the
first case can trade. A market that is merely *probably* Chainlink is skipped.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pmbtc.constants import SettlementSource


class EvidenceSource(StrEnum):
    """Where a extracted fact came from, ordered by trustworthiness."""

    STRUCTURED_FIELD = "structured_field"      # e.g. market.resolutionSource
    STRUCTURED_SERIES = "structured_series"    # e.g. events[].series[].recurrence
    STRUCTURED_TIME = "structured_time"        # e.g. events[].startTime / endDate
    SLUG = "slug"                              # e.g. btc-updown-5m-<epoch>
    TEXT_PATTERN = "text_pattern"              # parsed from the prose rules
    PROVIDER_DEFAULT = "provider_default"      # declared by our descriptor
    DERIVED = "derived"                        # computed from other fields
    MISSING = "missing"


class TieRule(StrEnum):
    """What happens when the settlement price exactly equals the reference.

    Not academic: the BTC 5-minute family resolves ``>=`` to **Up**, so an exact
    tie is a win for Up. At 1-cent tick sizes and 18-decimal price reports ties
    are rare, but a model that assumes the wrong tie rule is mis-specified at
    precisely the moment the market is closest to a coin flip.
    """

    TIE_UP = "tie_up"
    TIE_DOWN = "tie_down"
    TIE_FIFTY_FIFTY = "tie_fifty_fifty"
    TIE_VOID = "tie_void"
    UNKNOWN = "unknown"


class VerificationStatus(StrEnum):
    """Outcome of verifying a detected spec against configuration."""

    VERIFIED = "verified"
    REJECTED_CONFLICT = "rejected_conflict"
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"
    REJECTED_PROVIDER_MISMATCH = "rejected_provider_mismatch"
    REJECTED_PAIR_MISMATCH = "rejected_pair_mismatch"
    REJECTED_INTERVAL_MISMATCH = "rejected_interval_mismatch"
    REJECTED_TIMING = "rejected_timing"
    REJECTED_INCOMPLETE = "rejected_incomplete"
    REJECTED_PROVIDER_CHANGED = "rejected_provider_changed"

    @property
    def is_tradeable(self) -> bool:
        return self is VerificationStatus.VERIFIED


class FieldEvidence(BaseModel):
    """Provenance for a single extracted fact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: EvidenceSource
    confidence: float = Field(ge=0.0, le=1.0)
    #: Dotted path into the Gamma payload, so a disputed field is auditable.
    locator: str = ""
    #: Verbatim excerpt supporting the value. Truncated; the raw payload is
    #: captured separately by the store.
    excerpt: str = ""
    #: Set when structured metadata and prose disagreed.
    conflict: str = ""

    @property
    def corroborated(self) -> bool:
        return self.confidence >= 1.0 and not self.conflict


#: Fields that must be present and fully corroborated before any order is sent.
REQUIRED_FIELDS: tuple[str, ...] = (
    "provider",
    "trading_pair",
    "interval_seconds",
    "window_open_ms",
    "window_close_ms",
    "tie_rule",
    "price_decimals",
)


class SettlementSpecification(BaseModel):
    """Normalised, provider-agnostic description of how one market settles.

    Immutable by construction. Two markets with identical settlement mechanics
    produce identical :attr:`spec_hash` values, which is how the store detects a
    provider change inside a series.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- identity ---
    market_id: str
    condition_id: str
    slug: str
    question: str = ""
    series_slug: str = ""

    # --- settlement mechanics ---
    provider: SettlementSource
    venue: str
    trading_pair: str
    resolution_source_url: str = ""
    #: Window length in seconds, from the series recurrence where available.
    interval_seconds: int
    #: Half-open window [open, close). ``window_close_ms`` is the settlement
    #: instant: the price at this moment decides the outcome.
    window_open_ms: int
    window_close_ms: int
    #: All timestamps in this object are UTC epoch millis. Market *titles* are
    #: written in ET; that string is never used for arithmetic.
    timezone: str = "UTC"
    tie_rule: TieRule
    #: How the reference (opening) price is established.
    reference_rule: str = ""
    #: Reporting precision of the provider, and how values are rounded.
    price_decimals: int
    rounding_mode: str = "half_even"
    timestamp_semantics: str = "instant"

    # --- market mechanics (used by execution, recorded here for audit) ---
    outcomes: tuple[str, ...] = ("Up", "Down")
    tick_size: float | None = None
    min_order_size: float | None = None
    fees_enabled: bool | None = None

    # --- provenance ---
    evidence: dict[str, FieldEvidence] = Field(default_factory=dict)
    detected_at_ms: int = 0
    parser_version: str = "2.0"

    # ------------------------------------------------------------------ #
    @property
    def duration_seconds(self) -> int:
        return (self.window_close_ms - self.window_open_ms) // 1000

    @property
    def settlement_timestamp_ms(self) -> int:
        """Alias with intent: this is the instant the payout is decided."""
        return self.window_close_ms

    @property
    def confidence(self) -> float:
        """Overall confidence = the weakest required field.

        A chain is as strong as its weakest link, and a settlement spec is a
        chain: knowing the provider with certainty is useless if the settlement
        instant was guessed.
        """
        if not self.evidence:
            return 0.0
        scores = [
            self.evidence[name].confidence if name in self.evidence else 0.0
            for name in REQUIRED_FIELDS
        ]
        return min(scores) if scores else 0.0

    @property
    def conflicts(self) -> dict[str, str]:
        """Fields where structured metadata and prose disagreed."""
        return {name: ev.conflict for name, ev in self.evidence.items() if ev.conflict}

    @property
    def missing_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in REQUIRED_FIELDS
            if name not in self.evidence
            or self.evidence[name].source is EvidenceSource.MISSING
        )

    @property
    def spec_hash(self) -> str:
        """Stable fingerprint of the settlement *mechanics* only.

        Deliberately excludes identity and timestamps: two consecutive 5-minute
        markets settle identically and must hash identically, so that a change in
        this value means a genuine change in how the family resolves.
        """
        payload = "|".join(
            str(part)
            for part in (
                self.provider.value,
                self.venue,
                self.trading_pair,
                self.interval_seconds,
                self.tie_rule.value,
                self.price_decimals,
                self.rounding_mode,
                self.timestamp_semantics,
                self.reference_rule,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def evidence_summary(self) -> dict[str, str]:
        """Compact ``field -> "source@confidence"`` map for logs."""
        return {
            name: f"{ev.source.value}@{ev.confidence:.2f}" for name, ev in self.evidence.items()
        }
