"""Historical dataset: collection, quality, labelling, versioning, export.

    discovery (Module 3)
      -> HistoricalCollector   countdown scheduler, T-300 ... T-1
      -> FeatureProvider(s)    observations with full provenance
      -> QualityScorer         latency budgets -> quality flags
      -> LeakageGuard          write-time refusal of future information
      -> DatasetStore          append-only, snapshots immutable
      -> resolve_label()       official Polymarket outcome only
      -> DatasetExporter       parquet / arrow / csv / sqlite + manifest
      -> DatasetStats          coverage, balance, latency, quality report
"""

from __future__ import annotations

from pmbtc.dataset.collector import CollectionStats, HistoricalCollector
from pmbtc.dataset.export import DatasetExporter, ExportResult, build_rows
from pmbtc.dataset.labels import (
    LabelAudit,
    LabelResolution,
    audit_against_reference,
    resolve_label,
)
from pmbtc.dataset.leakage import (
    LeakageFinding,
    LeakageGuard,
    LeakageKind,
    assert_no_label_leakage,
    scan_monotonicity,
    scan_snapshots,
)
from pmbtc.dataset.manifest import DatasetManifest, build_manifest, config_hash, git_commit
from pmbtc.dataset.providers import (
    BinanceReferenceProvider,
    FeatureProvider,
    PolymarketQuoteProvider,
    SnapshotContext,
    TimeFeatureProvider,
)
from pmbtc.dataset.quality import QualityFilter, QualityScorer, SourceBudget
from pmbtc.dataset.schema import (
    DATASET_RECORD_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureSnapshot,
    MarketRecord,
    Observation,
    QualityFlag,
)
from pmbtc.dataset.stats import DatasetStats, compute_stats, render_stats, stats_for_store
from pmbtc.dataset.store import DatasetStore, ImmutableRecordError, open_dataset_store

__all__ = [
    "DATASET_RECORD_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "BinanceReferenceProvider",
    "CollectionStats",
    "DatasetExporter",
    "DatasetManifest",
    "DatasetStats",
    "DatasetStore",
    "ExportResult",
    "FeatureProvider",
    "FeatureSnapshot",
    "HistoricalCollector",
    "ImmutableRecordError",
    "LabelAudit",
    "LabelResolution",
    "LeakageFinding",
    "LeakageGuard",
    "LeakageKind",
    "MarketRecord",
    "Observation",
    "PolymarketQuoteProvider",
    "QualityFilter",
    "QualityFlag",
    "QualityScorer",
    "SnapshotContext",
    "SourceBudget",
    "TimeFeatureProvider",
    "assert_no_label_leakage",
    "audit_against_reference",
    "build_manifest",
    "build_rows",
    "compute_stats",
    "config_hash",
    "git_commit",
    "open_dataset_store",
    "render_stats",
    "resolve_label",
    "scan_monotonicity",
    "scan_snapshots",
    "stats_for_store",
]
