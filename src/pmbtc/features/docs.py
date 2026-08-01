"""Automatic feature documentation.

Generated from the declarations themselves, so it cannot drift from the code:
if a feature's formula changes, the document changes on the next build. A
hand-maintained feature dictionary is wrong within a week.

Also enforces the documentation contract — a feature missing a formula, units,
or a stated purpose is a build failure, not a gap someone will fill in later.
"""

from __future__ import annotations

from pathlib import Path

from pmbtc.exceptions import ConfigError
from pmbtc.features.base import RAW
from pmbtc.features.graph import FeatureGraph


def undocumented(graph: FeatureGraph) -> list[str]:
    """Features that fail the documentation contract."""
    return sorted(
        name for name, feature in graph.features.items() if not feature.spec.documented
    )


def require_documentation(graph: FeatureGraph) -> None:
    missing = undocumented(graph)
    if missing:
        raise ConfigError(
            "Features are missing required documentation (formula, units, purpose)",
            context={"features": ", ".join(missing)},
        )


def generate_markdown(graph: FeatureGraph) -> str:
    """Full feature dictionary, grouped by family."""
    by_family: dict[str, list[str]] = {}
    for name in graph.order:
        feature = graph.features[name]
        module = type(feature).__module__
        family = feature.spec.source
        del module
        by_family.setdefault(family, []).append(name)

    lines: list[str] = [
        "# Feature dictionary",
        "",
        "Generated from the feature declarations. Do not edit by hand — rebuild with",
        "`pmbtc feature-docs`.",
        "",
        f"**{len(graph)} features** across {len(by_family)} sources, "
        f"{len(graph.raw_sources)} raw inputs.",
        "",
        "## Raw inputs",
        "",
    ]
    for source in graph.raw_sources:
        dependents = len(graph.impact.get(source, frozenset()))
        lines.append(f"- `{source}` — feeds {dependents} feature(s)")
    lines.append("")

    for family in sorted(by_family):
        lines += [f"## Source: `{family}`", ""]
        for name in by_family[family]:
            spec = graph.features[name].spec
            inputs = ", ".join(
                f"`{i[len(RAW):]}` (raw)" if i.startswith(RAW) else f"[`{i}`](#{i})"
                for i in spec.inputs
            ) or "none"
            lines += [
                f"### `{name}`",
                "",
                f"{spec.purpose}",
                "",
                "| property | value |",
                "|---|---|",
                f"| formula | `{spec.formula}` |",
                f"| inputs | {inputs} |",
                f"| units | {spec.units} |",
                f"| tier | `{spec.tier.value}` |",
                f"| freshness budget | {spec.freshness_budget_ms} ms |",
                f"| reproducibility | `{spec.reproducibility.value}` |",
                f"| group | {spec.group} |",
                f"| trainable | {'yes' if spec.usable_for_training else 'no'} |",
                "",
            ]

    lines += [
        "## Dependency graph",
        "",
        "```mermaid",
        graph.to_mermaid(),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_docs(graph: FeatureGraph, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit LF: the same generator must emit identical bytes on Windows
    # and on a Linux CI runner, or the staleness check reports false drift.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(generate_markdown(graph))
    return path
