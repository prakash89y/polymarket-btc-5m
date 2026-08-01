"""Gamma API layer: discovery, lifecycle, health, liquidity, schema, replay.

    GammaClient          network + archival capture
      -> SchemaChecker   fail safe on incompatible payloads
      -> parser          canonical settlement spec (Module 2)
      -> LiquiditySnapshot
      -> MarketHealthChecker
      -> SettlementVerifier
      -> LifecycleTracker
    ReplaySession        regression-test the parser against the archive
"""

from __future__ import annotations

from pmbtc.gamma.client import GammaClient, ResponseArchive
from pmbtc.gamma.discovery import (
    DiscoveryResult,
    MarketCandidate,
    MarketDiscovery,
    RejectedMarket,
)
from pmbtc.gamma.health import HealthIssue, HealthResult, MarketHealthChecker
from pmbtc.gamma.lifecycle import (
    LifecycleTracker,
    MarketState,
    Transition,
    classify,
    open_lifecycle_tracker,
)
from pmbtc.gamma.liquidity import (
    BookLevel,
    LiquiditySnapshot,
    LiquidityStore,
    open_liquidity_store,
    snapshot_from_market,
)
from pmbtc.gamma.replay import ReplayReport, ReplaySession, interpretation_of
from pmbtc.gamma.schema import (
    DriftSeverity,
    SchemaChecker,
    SchemaCheckResult,
    SchemaRegistry,
    SchemaViolation,
    open_schema_checker,
)

__all__ = [
    "BookLevel",
    "DiscoveryResult",
    "DriftSeverity",
    "GammaClient",
    "HealthIssue",
    "HealthResult",
    "LifecycleTracker",
    "LiquiditySnapshot",
    "LiquidityStore",
    "MarketCandidate",
    "MarketDiscovery",
    "MarketHealthChecker",
    "MarketState",
    "RejectedMarket",
    "ReplayReport",
    "ReplaySession",
    "ResponseArchive",
    "SchemaCheckResult",
    "SchemaChecker",
    "SchemaRegistry",
    "SchemaViolation",
    "Transition",
    "classify",
    "interpretation_of",
    "open_lifecycle_tracker",
    "open_liquidity_store",
    "open_schema_checker",
    "snapshot_from_market",
]
