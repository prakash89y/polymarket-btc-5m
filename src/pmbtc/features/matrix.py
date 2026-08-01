"""The versioned feature matrix — Module 6's output.

Built by replaying archived raw frames through the same engine the live system
uses, so it is reproducible without any live service. The matrix carries a
version fingerprint covering the feature set, the engine precision, and the
declared formulas: change any feature's definition and the fingerprint changes,
which makes "which feature set was this model trained on" answerable rather than
assumed.

Rows are one per ``(market, horizon)``, matching the dataset's snapshot grid, so
the matrix joins to labels on exactly the key the dataset already uses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmbtc.features.engine import VALUE_PRECISION, FeatureEngine, FeatureVector
from pmbtc.features.graph import FeatureGraph
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.features.matrix")


def feature_set_version(graph: FeatureGraph) -> str:
    """Fingerprint of the feature definitions themselves.

    Covers names, formulas, inputs, and units — everything that changes what a
    column *means*. Two matrices with the same version are directly comparable;
    two with different versions are not, whatever their column names suggest.
    """
    parts: list[str] = [f"precision={VALUE_PRECISION}"]
    for name in graph.order:
        spec = graph.features[name].spec
        parts.append(
            f"{name}|{spec.formula}|{','.join(spec.inputs)}|{spec.units}|{spec.tier.value}"
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class MatrixRow:
    condition_id: str
    slug: str
    horizon_seconds: int
    snapshot_time_ms: int
    settlement_time_ms: int
    values: dict[str, float | None]
    label: int | None = None
    contested: tuple[str, ...] = ()
    fingerprint: str = ""

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "horizon_seconds": self.horizon_seconds,
            "snapshot_time_ms": self.snapshot_time_ms,
            "settlement_time_ms": self.settlement_time_ms,
            "label": self.label,
            "contested_count": len(self.contested),
            "row_fingerprint": self.fingerprint,
        }
        for name, value in self.values.items():
            row[f"f_{name}"] = value
        return row


@dataclass
class FeatureMatrix:
    """A versioned, reproducible feature matrix."""

    rows: list[MatrixRow] = field(default_factory=list)
    feature_names: tuple[str, ...] = ()
    version: str = ""
    built_at_ms: int = 0
    source: str = "archive"

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def labelled(self) -> list[MatrixRow]:
        return [r for r in self.rows if r.label is not None]

    def fingerprint(self) -> str:
        """Hash over every row fingerprint — the matrix's content identity."""
        joined = "|".join(r.fingerprint for r in self.rows)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def coverage(self) -> dict[str, float]:
        if not self.rows:
            return {}
        return {
            name: sum(1 for r in self.rows if r.values.get(name) is not None) / len(self.rows)
            for name in self.feature_names
        }

    def to_arrays(self, names: list[str] | None = None) -> tuple[Any, Any, list[str]]:
        """``(X, y, column_names)`` for the labelled rows only."""
        import numpy as np

        columns = names or list(self.feature_names)
        labelled = self.labelled
        x = np.array(
            [[_nan_if_none(r.values.get(c)) for c in columns] for r in labelled], dtype=float
        )
        y = np.array([r.label for r in labelled], dtype=int)
        return x, y, columns

    def manifest(self) -> dict[str, Any]:
        return {
            "feature_set_version": self.version,
            "matrix_fingerprint": self.fingerprint(),
            "built_at_ms": self.built_at_ms,
            "source": self.source,
            "rows": len(self.rows),
            "labelled_rows": len(self.labelled),
            "features": list(self.feature_names),
            "value_precision": VALUE_PRECISION,
        }

    def write(self, directory: Path, name: str = "features") -> dict[str, Path]:
        """Write Parquet plus a manifest. Parquet is the training format."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory.mkdir(parents=True, exist_ok=True)
        flat = [r.flat() for r in self.rows]
        columns = list(dict.fromkeys(k for row in flat for k in row))
        table = pa.table({c: [row.get(c) for row in flat] for c in columns})
        parquet_path = directory / f"{name}.parquet"
        pq.write_table(table, parquet_path, compression="zstd")

        manifest_path = directory / f"{name}.manifest.json"
        manifest_path.write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")
        log.info(
            "matrix.written",
            path=str(parquet_path),
            rows=len(self.rows),
            version=self.version,
            fingerprint=self.fingerprint(),
        )
        return {"parquet": parquet_path, "manifest": manifest_path}


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else value


class MatrixBuilder:
    """Accumulates engine output into a matrix."""

    def __init__(self, engine: FeatureEngine | None = None) -> None:
        self.engine = engine or FeatureEngine()
        self.version = feature_set_version(self.engine.graph)
        self.rows: list[MatrixRow] = []

    def add(
        self,
        *,
        condition_id: str,
        slug: str,
        horizon_seconds: int,
        snapshot_time_ms: int,
        settlement_time_ms: int,
        vector: FeatureVector,
        label: int | None = None,
        contested: tuple[str, ...] = (),
    ) -> MatrixRow:
        row = MatrixRow(
            condition_id=condition_id,
            slug=slug,
            horizon_seconds=horizon_seconds,
            snapshot_time_ms=snapshot_time_ms,
            settlement_time_ms=settlement_time_ms,
            values=dict(vector.values),
            label=label,
            contested=contested,
            fingerprint=vector.fingerprint(),
        )
        self.rows.append(row)
        return row

    def build(self, built_at_ms: int, source: str = "archive") -> FeatureMatrix:
        # Sorted by settlement then countdown, so a chronological split is a
        # slice rather than a sort — and identical on every build.
        rows = sorted(self.rows, key=lambda r: (r.settlement_time_ms, -r.horizon_seconds))
        return FeatureMatrix(
            rows=rows,
            feature_names=self.engine.feature_names,
            version=self.version,
            built_at_ms=built_at_ms,
            source=source,
        )
