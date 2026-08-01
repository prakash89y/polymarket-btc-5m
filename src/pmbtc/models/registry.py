"""Model registry: identity, reproducibility, and artifacts.

Every trained model gets an ID derived from what actually determines its
behaviour — dataset version, feature-set version, training config, and
hyperparameters. Two models with the same ID are the same model; a model whose
ID does not match its recorded inputs was not produced by the pipeline that
claims to have produced it.

Artifacts are written once and never mutated. Promotion changes a pointer, not
a file, so rolling back is repointing rather than retraining.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmbtc.exceptions import ModelError
from pmbtc.logging_setup import get_logger
from pmbtc.utils.timeutils import isoformat, utc_now_ms

log = get_logger("pmbtc.models.registry")


def stable_hash(payload: Any, length: int = 12) -> str:
    """Deterministic hash of any JSON-serialisable structure."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


@dataclass
class ModelCard:
    """Everything needed to identify, reproduce, and judge one model."""

    model_id: str
    model_type: str
    dataset_version: str
    feature_schema_version: str
    config_hash: str
    hyperparameter_hash: str
    trained_at_ms: int
    #: Features the model was actually fitted on, in order.
    features: tuple[str, ...] = ()
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    statistical_tests: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    shadow: dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    promotion_reasons: list[str] = field(default_factory=list)
    training_samples: int = 0
    seed: int = 42

    # --- git provenance: what code actually ran ------------------------ #
    #: Filled from the working tree at training time. Together with the
    #: dataset and feature-schema versions these make an experiment
    #: reproducible from history rather than merely described by it.
    git_commit: str = "unavailable"
    git_branch: str = "unavailable"
    git_tag: str = ""
    git_dirty: bool = False

    @property
    def reproducible_from_git(self) -> bool:
        """True when the exact training code exists in git history."""
        return self.git_commit != "unavailable" and not self.git_dirty

    @property
    def trained_at(self) -> str:
        return isoformat(self.trained_at_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "dataset_version": self.dataset_version,
            "feature_schema_version": self.feature_schema_version,
            "config_hash": self.config_hash,
            "hyperparameter_hash": self.hyperparameter_hash,
            "trained_at_ms": self.trained_at_ms,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "seed": self.seed,
            "features": list(self.features),
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "statistical_tests": self.statistical_tests,
            "validation": self.validation,
            "explainability": self.explainability,
            "shadow": self.shadow,
            "promoted": self.promoted,
            "promotion_reasons": self.promotion_reasons,
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "tag": self.git_tag,
                "dirty": self.git_dirty,
                "reproducible": self.reproducible_from_git,
            },
        }

    def reproducibility_key(self) -> str:
        """What must match for two runs to be considered the same experiment."""
        return stable_hash(
            {
                "dataset": self.dataset_version,
                "features": self.feature_schema_version,
                "config": self.config_hash,
                "hyperparameters": self.hyperparameter_hash,
                "model_type": self.model_type,
                "seed": self.seed,
                "git_commit": self.git_commit,
            }
        )


def make_model_id(
    model_type: str,
    dataset_version: str,
    feature_schema_version: str,
    config_hash: str,
    hyperparameter_hash: str,
    trained_at_ms: int,
) -> str:
    """Human-scannable, collision-resistant model identifier."""
    stamp = datetime.fromtimestamp(trained_at_ms / 1000, tz=UTC).strftime("%Y%m%dT%H%M%S")
    digest = stable_hash(
        {
            "t": model_type,
            "d": dataset_version,
            "f": feature_schema_version,
            "c": config_hash,
            "h": hyperparameter_hash,
            "ts": trained_at_ms,
        },
        length=8,
    )
    return f"{model_type}-{stamp}-{digest}"


class ModelRegistry:
    """Immutable model artifacts plus a mutable production pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.root / "production.json"

    # ------------------------------------------------------------------ #
    def model_dir(self, model_id: str) -> Path:
        return self.root / model_id

    def save(self, card: ModelCard, estimator: Any, calibrator: Any = None) -> Path:
        """Write a model and its card. Refuses to overwrite."""
        directory = self.model_dir(card.model_id)
        if directory.exists():
            raise ModelError(
                "Model artifact already exists; artifacts are immutable",
                context={"model_id": card.model_id},
            )
        directory.mkdir(parents=True)
        (directory / "card.json").write_text(
            json.dumps(card.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        with (directory / "estimator.pkl").open("wb") as handle:
            pickle.dump({"estimator": estimator, "calibrator": calibrator}, handle)
        log.info(
            "model.saved",
            model_id=card.model_id,
            model_type=card.model_type,
            metrics=card.metrics.get("holdout", {}),
        )
        return directory

    def load_card(self, model_id: str) -> ModelCard:
        path = self.model_dir(model_id) / "card.json"
        if not path.exists():
            raise ModelError("No such model", context={"model_id": model_id})
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelCard(
            model_id=data["model_id"],
            model_type=data["model_type"],
            dataset_version=data["dataset_version"],
            feature_schema_version=data["feature_schema_version"],
            config_hash=data["config_hash"],
            hyperparameter_hash=data["hyperparameter_hash"],
            trained_at_ms=int(data["trained_at_ms"]),
            features=tuple(data.get("features", [])),
            hyperparameters=data.get("hyperparameters", {}),
            metrics=data.get("metrics", {}),
            baseline_metrics=data.get("baseline_metrics", {}),
            statistical_tests=data.get("statistical_tests", []),
            validation=data.get("validation", {}),
            explainability=data.get("explainability", {}),
            shadow=data.get("shadow", {}),
            promoted=bool(data.get("promoted", False)),
            promotion_reasons=data.get("promotion_reasons", []),
            training_samples=int(data.get("training_samples", 0)),
            seed=int(data.get("seed", 42)),
            git_commit=str(data.get("git", {}).get("commit", "unavailable")),
            git_branch=str(data.get("git", {}).get("branch", "unavailable")),
            git_tag=str(data.get("git", {}).get("tag", "")),
            git_dirty=bool(data.get("git", {}).get("dirty", False)),
        )

    def load_estimator(self, model_id: str) -> tuple[Any, Any]:
        path = self.model_dir(model_id) / "estimator.pkl"
        if not path.exists():
            raise ModelError("No estimator artifact", context={"model_id": model_id})
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload["estimator"], payload.get("calibrator")

    def list_models(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "card.json").exists()
        )

    # ------------------------------------------------------------------ #
    def promote(self, model_id: str, reasons: list[str]) -> None:
        """Point production at a model. Only the gate calls this."""
        card = self.load_card(model_id)
        card.promoted = True
        card.promotion_reasons = reasons
        (self.model_dir(model_id) / "card.json").write_text(
            json.dumps(card.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        self.pointer_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "promoted_at_ms": utc_now_ms(),
                    "promoted_at": isoformat(utc_now_ms()),
                    "reasons": reasons,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("model.promoted", model_id=model_id, reasons=reasons)

    def production_model_id(self) -> str | None:
        if not self.pointer_path.exists():
            return None
        try:
            return str(json.loads(self.pointer_path.read_text(encoding="utf-8"))["model_id"])
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def champion(self) -> ModelCard | None:
        model_id = self.production_model_id()
        return self.load_card(model_id) if model_id else None
