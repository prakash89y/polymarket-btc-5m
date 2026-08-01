"""Dataset export.

Produces the training table: one row per ``(market, horizon)``, with the label
joined on from the market record. Parquet is the default -- columnar, typed,
compressed, and read natively by every training stack in Module 7.

The export re-runs the leakage scan over everything it is about to write. That
is redundant with the write-time guard, and deliberately so: the guard protects
against bugs in the collector, and this protects against snapshots written by
an older build of it. An export that would emit leaked rows fails instead.

The label join happens here and only here, in one function, so there is a single
place to audit for the question "could the target have reached the features?".
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pmbtc.config import Config
from pmbtc.dataset.leakage import LeakageGuard, scan_snapshots
from pmbtc.dataset.manifest import DatasetManifest, build_manifest
from pmbtc.dataset.quality import QualityFilter
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord
from pmbtc.dataset.store import DatasetStore
from pmbtc.exceptions import DataError, LookaheadError
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.dataset.export")

ExportFormat = Literal["parquet", "arrow", "csv", "sqlite"]

#: Market columns safe to attach to a feature row. Everything outcome-derived is
#: excluded from the feature space and lives only in the label columns.
_METADATA_COLUMNS = (
    "market_id",
    "event_id",
    "series_id",
    "series_slug",
    "discovery_time_ms",
    "open_time_ms",
    "lock_time_ms",
    "settlement_time_ms",
    "settlement_provider",
    "settlement_spec_hash",
    "trading_pair",
    "tie_rule",
)

_LABEL_COLUMNS = (
    "label",
    "official_outcome",
    "settlement_probability",
    "yes_final_price",
    "no_final_price",
    "resolved_at_ms",
)


@dataclass
class ExportResult:
    path: Path
    manifest: DatasetManifest
    rows: int
    columns: int
    format: str
    excluded_rows: int = 0

    def __str__(self) -> str:
        return (
            f"{self.rows} row(s) x {self.columns} column(s) -> {self.path.name} "
            f"[{self.manifest.fingerprint}]"
        )


def build_rows(
    markets: dict[str, MarketRecord],
    snapshots: list[FeatureSnapshot],
    *,
    labelled_only: bool = True,
    quality_filter: QualityFilter | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Join snapshots to their market metadata and label.

    Returns ``(rows, excluded)``. Rows are sorted by settlement time then
    horizon, so a chronological split is a slice rather than a sort.
    """
    quality_filter = quality_filter or QualityFilter()
    rows: list[dict[str, Any]] = []
    excluded = 0

    for snapshot in snapshots:
        market = markets.get(snapshot.condition_id)
        if market is None:
            excluded += 1
            continue
        if labelled_only and market.label is None:
            excluded += 1
            continue
        if not quality_filter.accepts_snapshot(snapshot.quality_score, snapshot.coverage):
            excluded += 1
            continue

        row = snapshot.flat_record()
        market_row = market.flat_record()
        for column in _METADATA_COLUMNS:
            row[column] = market_row.get(column)
        for column in _LABEL_COLUMNS:
            row[column] = market_row.get(column)
        rows.append(row)

    rows.sort(key=lambda r: (r["settlement_time_ms"], -r["horizon_seconds"]))
    return rows, excluded


class DatasetExporter:
    """Writes the training table in the supported formats."""

    def __init__(self, config: Config, store: DatasetStore) -> None:
        self.config = config
        self.store = store
        self.export_dir = config.resolved_path(config.dataset.export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def export(
        self,
        fmt: ExportFormat | None = None,
        *,
        name: str = "dataset",
        labelled_only: bool = True,
        quality_filter: QualityFilter | None = None,
        enforce_leakage_scan: bool = True,
    ) -> ExportResult:
        fmt = fmt or self.config.dataset.default_format
        snapshots = self.store.snapshots()
        markets = {m.condition_id: m for m in self.store.markets()}

        if enforce_leakage_scan:
            findings = scan_snapshots(
                snapshots, LeakageGuard(self.config.dataset.observation_tolerance_ms)
            )
            if findings:
                log.error("export.leakage_detected", count=len(findings))
                raise LookaheadError(
                    "Refusing to export: leaked observations found in the dataset",
                    context={
                        "findings": "; ".join(str(f) for f in findings[:10]),
                        "total": len(findings),
                    },
                )

        rows, excluded = build_rows(
            markets, snapshots, labelled_only=labelled_only, quality_filter=quality_filter
        )
        if not rows:
            raise DataError(
                "Nothing to export",
                context={
                    "snapshots": len(snapshots),
                    "markets": len(markets),
                    "excluded": excluded,
                    "hint": "markets may not be labelled yet",
                },
            )

        settlement_ms = [r["settlement_time_ms"] for r in rows]
        providers: dict[str, int] = {}
        for market in markets.values():
            key = market.settlement_provider.value
            providers[key] = providers.get(key, 0) + 1

        manifest = build_manifest(
            self.config,
            market_count=len(markets),
            snapshot_count=len(snapshots),
            labelled_count=sum(1 for m in markets.values() if m.is_labelled),
            period_start_ms=min(settlement_ms),
            period_end_ms=max(settlement_ms),
            settlement_providers=providers,
            notes=f"{excluded} snapshot(s) excluded by filters",
        )

        writer = {
            "parquet": self._write_parquet,
            "arrow": self._write_arrow,
            "csv": self._write_csv,
            "sqlite": self._write_sqlite,
        }[fmt]
        path = writer(rows, name)

        manifest.files = {fmt: path.name}
        manifest_path = self.export_dir / f"{name}.manifest.json"
        manifest.write(manifest_path)
        manifest.files["manifest"] = manifest_path.name

        result = ExportResult(
            path=path,
            manifest=manifest,
            rows=len(rows),
            columns=len(rows[0]),
            format=fmt,
            excluded_rows=excluded,
        )
        log.info(
            "export.written",
            path=str(path),
            rows=result.rows,
            columns=result.columns,
            fingerprint=manifest.fingerprint,
        )
        return result

    # ------------------------------------------------------------------ #
    def _table(self, rows: list[dict[str, Any]]) -> Any:
        import pyarrow as pa

        # Union of keys: providers can legitimately differ in which features
        # they produced, and a missing column must become a null, not a crash.
        columns = list(dict.fromkeys(key for row in rows for key in row))
        return pa.table({name: [row.get(name) for row in rows] for name in columns})

    def _write_parquet(self, rows: list[dict[str, Any]], name: str) -> Path:
        import pyarrow.parquet as pq

        path = self.export_dir / f"{name}.parquet"
        pq.write_table(self._table(rows), path, compression="zstd")
        return path

    def _write_arrow(self, rows: list[dict[str, Any]], name: str) -> Path:
        import pyarrow as pa

        # Arrow IPC file format directly, not the deprecated feather wrapper.
        path = self.export_dir / f"{name}.arrow"
        table = self._table(rows)
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        return path

    def _write_csv(self, rows: list[dict[str, Any]], name: str) -> Path:
        import csv

        path = self.export_dir / f"{name}.csv"
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_sqlite(self, rows: list[dict[str, Any]], name: str) -> Path:
        path = self.export_dir / f"{name}.sqlite"
        if path.exists():
            path.unlink()
        columns = list(dict.fromkeys(key for row in rows for key in row))

        def sql_type(column: str) -> str:
            sample = next((r[column] for r in rows if r.get(column) is not None), None)
            if isinstance(sample, bool):
                return "INTEGER"
            if isinstance(sample, int):
                return "INTEGER"
            if isinstance(sample, float):
                return "REAL"
            return "TEXT"

        with sqlite3.connect(path) as connection:
            definition = ", ".join(f'"{c}" {sql_type(c)}' for c in columns)
            connection.execute(f"CREATE TABLE samples ({definition})")
            placeholders = ",".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO samples VALUES ({placeholders})",
                [tuple(row.get(c) for c in columns) for row in rows],
            )
            # The two queries every consumer runs first.
            connection.execute(
                "CREATE INDEX idx_settlement ON samples(settlement_time_ms)"
            )
            connection.execute("CREATE INDEX idx_horizon ON samples(horizon_seconds)")
            connection.commit()
        return path
