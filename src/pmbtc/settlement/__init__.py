"""Settlement verification engine — the immutable core of the system.

Pipeline:

    raw Gamma market
        -> parse_settlement_spec()      canonical spec + per-field evidence
        -> SettlementSpecStore.record() durable history, change detection
        -> SettlementVerifier.verify()  config gates -> trading_enabled
        -> SettlementReport             operator report + audit log

Nothing downstream may place an order without a
:class:`~pmbtc.settlement.verifier.VerificationResult` whose
``trading_enabled`` is true.
"""

from __future__ import annotations

from pmbtc.settlement.parser import (
    PARSER_VERSION,
    SettlementParseError,
    parse_settlement_spec,
)
from pmbtc.settlement.providers import (
    PRICE_PROVIDERS,
    PROVIDER_REGISTRY,
    ProviderDescriptor,
    SettlementPriceProvider,
    canonical_pair,
    descriptor_for,
)
from pmbtc.settlement.report import SettlementReport
from pmbtc.settlement.spec import (
    REQUIRED_FIELDS,
    EvidenceSource,
    FieldEvidence,
    SettlementSpecification,
    TieRule,
    VerificationStatus,
)
from pmbtc.settlement.store import (
    SettlementChange,
    SettlementSpecStore,
    open_spec_store,
)
from pmbtc.settlement.verifier import Check, SettlementVerifier, VerificationResult

__all__ = [
    "PARSER_VERSION",
    "PRICE_PROVIDERS",
    "PROVIDER_REGISTRY",
    "REQUIRED_FIELDS",
    "Check",
    "EvidenceSource",
    "FieldEvidence",
    "ProviderDescriptor",
    "SettlementChange",
    "SettlementParseError",
    "SettlementPriceProvider",
    "SettlementReport",
    "SettlementSpecStore",
    "SettlementSpecification",
    "SettlementVerifier",
    "TieRule",
    "VerificationResult",
    "VerificationStatus",
    "canonical_pair",
    "descriptor_for",
    "open_spec_store",
    "parse_settlement_spec",
]
