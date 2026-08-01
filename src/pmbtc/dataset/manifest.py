"""Dataset versioning: what a training run can point at and say "this, exactly".

Every export carries a manifest recording the dataset version, the period it
covers, both schema versions, the git commit, and a hash of the configuration
that produced it. Together those pin down the *interpretation* of the data as
well as the data itself -- a dataset collected under a 20-second safety window
is not the same dataset as one collected under 30, even if every byte of price
data matches.

The config hash covers only the fields that affect collection semantics. Hashing
the whole config would change the fingerprint when someone edits a log level,
and a fingerprint that changes for irrelevant reasons is one people learn to
ignore.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmbtc import __version__
from pmbtc.config import Config
from pmbtc.dataset.schema import DATASET_RECORD_VERSION, FEATURE_SCHEMA_VERSION
from pmbtc.logging_setup import get_logger
from pmbtc.settlement.parser import PARSER_VERSION
from pmbtc.utils.timeutils import isoformat, utc_now_ms

log = get_logger("pmbtc.dataset.manifest")


def git_commit(repo_root: Path | None = None) -> str:
    """Current commit, or ``"unavailable"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unavailable"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root) if repo_root else None,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return f"{commit}-dirty" if dirty.stdout.strip() else commit


#: Config sections whose values change what the data *means*.
SEMANTIC_SECTIONS = ("app", "settlement", "dataset", "clock", "health", "polymarket")


def config_hash(config: Config) -> str:
    """Stable fingerprint of the collection-relevant configuration."""
    dumped = config.model_dump(mode="json")
    subset = {section: dumped.get(section) for section in SEMANTIC_SECTIONS}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class DatasetManifest:
    """Everything needed to reproduce a training run."""

    dataset_version: str
    feature_schema_version: str
    settlement_schema_version: str
    dataset_record_version: str
    package_version: str
    git_commit: str
    config_hash: str
    exported_at_ms: int
    period_start_ms: int | None
    period_end_ms: int | None
    market_count: int
    snapshot_count: int
    labelled_count: int
    horizons: tuple[int, ...] = ()
    settlement_providers: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "dataset_version": self.dataset_version,
            "feature_schema_version": self.feature_schema_version,
            "settlement_schema_version": self.settlement_schema_version,
            "dataset_record_version": self.dataset_record_version,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "exported_at_ms": self.exported_at_ms,
            "exported_at": isoformat(self.exported_at_ms),
            "collection_period": {
                "start_ms": self.period_start_ms,
                "end_ms": self.period_end_ms,
                "start": isoformat(self.period_start_ms) if self.period_start_ms else None,
                "end": isoformat(self.period_end_ms) if self.period_end_ms else None,
            },
            "counts": {
                "markets": self.market_count,
                "snapshots": self.snapshot_count,
                "labelled_markets": self.labelled_count,
            },
            "horizons": list(self.horizons),
            "settlement_providers": self.settlement_providers,
            "files": self.files,
            "notes": self.notes,
        }
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @property
    def fingerprint(self) -> str:
        """One string identifying this exact dataset build."""
        return (
            f"{self.dataset_version}+fs{self.feature_schema_version}"
            f"+ss{self.settlement_schema_version}+cfg{self.config_hash}"
            f"+git{self.git_commit[:8]}"
        )


def build_manifest(
    config: Config,
    *,
    market_count: int,
    snapshot_count: int,
    labelled_count: int,
    period_start_ms: int | None,
    period_end_ms: int | None,
    settlement_providers: dict[str, int] | None = None,
    notes: str = "",
) -> DatasetManifest:
    return DatasetManifest(
        dataset_version=config.dataset.dataset_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        settlement_schema_version=PARSER_VERSION,
        dataset_record_version=DATASET_RECORD_VERSION,
        package_version=__version__,
        git_commit=git_commit(config.resolved_path(config.app.base_dir)),
        config_hash=config_hash(config),
        exported_at_ms=utc_now_ms(),
        period_start_ms=period_start_ms,
        period_end_ms=period_end_ms,
        market_count=market_count,
        snapshot_count=snapshot_count,
        labelled_count=labelled_count,
        horizons=tuple(config.dataset.snapshot_horizons_s),
        settlement_providers=settlement_providers or {},
        notes=notes,
    )
