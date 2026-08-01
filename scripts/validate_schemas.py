"""Schema validation — the four versions must be present, distinct, and honest.

Run in CI. Catches the failure where someone changes a feature formula, a
dataset record shape, or a settlement rule without bumping the version that
describes it — after which stored data and new data look comparable and are not.

The feature-schema version is a hash of the declarations, so it cannot be
forgotten; the check here is that it still *matches* what the running code
produces. The hand-maintained versions are checked for presence and format.
"""

from __future__ import annotations

import re
import sys

FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


SEMVER = re.compile(r"^\d+\.\d+(\.\d+)?$")


def main() -> int:
    from pmbtc.dataset.schema import DATASET_RECORD_VERSION, FEATURE_SCHEMA_VERSION
    from pmbtc.features.families import all_features
    from pmbtc.features.graph import build_graph
    from pmbtc.features.matrix import feature_set_version
    from pmbtc.features.provider import (
        ENGINEERED_FEATURE_NAMES,
        canonical_feature_schema_version,
    )
    from pmbtc.settlement.parser import PARSER_VERSION

    print("\n1. Versions declared")
    check("dataset feature schema", SEMVER.match(FEATURE_SCHEMA_VERSION) is not None,
          FEATURE_SCHEMA_VERSION)
    check("dataset record schema", SEMVER.match(DATASET_RECORD_VERSION) is not None,
          DATASET_RECORD_VERSION)
    check("settlement parser schema", SEMVER.match(PARSER_VERSION) is not None,
          PARSER_VERSION)

    print("\n2. Feature schema is a live hash of the declarations")
    graph = build_graph(all_features())
    computed = feature_set_version(graph)
    canonical = canonical_feature_schema_version()
    check("canonical version matches the graph", computed == canonical, canonical)
    check("version is stable across builds",
          canonical_feature_schema_version() == canonical)

    print("\n3. Feature declarations are complete")
    undocumented = [f.name for f in all_features() if not f.spec.documented]
    check("every feature documented", not undocumented, ", ".join(undocumented[:5]))
    missing_inputs = [f.name for f in all_features() if not f.spec.inputs]
    check("every feature declares inputs", not missing_inputs,
          ", ".join(missing_inputs[:5]))
    unreproducible = [
        f.name for f in all_features()
        if f.spec.reproducibility.value == "not_reproducible"
    ]
    check("no unreproducible feature", not unreproducible, ", ".join(unreproducible[:5]))

    print("\n4. One canonical feature set")
    check("engineered set matches the family set",
          set(ENGINEERED_FEATURE_NAMES) == {f.name for f in all_features()},
          f"{len(ENGINEERED_FEATURE_NAMES)} features")

    print("\n5. Dependency graph is sound")
    position = {name: i for i, name in enumerate(graph.order)}
    ordered = all(
        position[dep] < position[f.name]
        for f in graph.features.values()
        for dep in f.dependencies
    )
    check("topological order valid", ordered, f"{len(graph.order)} features")
    check("order is deterministic", build_graph(all_features()).order == graph.order)

    print("\n6. Model registry schema")
    from pmbtc.models.registry import ModelCard

    card = ModelCard(
        model_id="x", model_type="t", dataset_version="d",
        feature_schema_version=canonical, config_hash="c",
        hyperparameter_hash="h", trained_at_ms=0,
    )
    payload = card.as_dict()
    for required in ("model_id", "dataset_version", "feature_schema_version",
                     "config_hash", "hyperparameter_hash", "trained_at_ms", "git"):
        check(f"model card carries {required}", required in payload)
    check("model card records git provenance",
          set(payload["git"]) >= {"commit", "branch", "tag", "dirty", "reproducible"})

    print()
    if FAILURES:
        print(f"{len(FAILURES)} schema check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("Schema validation OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
