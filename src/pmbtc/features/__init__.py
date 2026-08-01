"""Feature declarations and the registry that enforces them.

Module 6 builds the engineering pipeline on top of this; Module 5 already
depends on it, because no feature may be recorded without first declaring its
tier, source, and reproducibility policy.
"""

from __future__ import annotations

from pmbtc.features.registry import (
    REGISTRY,
    FeatureRegistry,
    FeatureSpec,
    FeatureTier,
    ReproducibilityPolicy,
    register,
)

__all__ = [
    "REGISTRY",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureTier",
    "ReproducibilityPolicy",
    "register",
]
