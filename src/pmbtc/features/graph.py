"""Feature dependency graph: ordering, cycle detection, and impact analysis.

Three jobs:

**Order.** Features are computed in topological order, so a feature that depends
on another always sees a finished value rather than a stale one from the
previous pass.

**Cycles.** A dependency cycle is a construction error, and it is detected at
build time with the offending path named — not discovered later as a feature
that mysteriously never updates.

**Impact.** Given the raw sources a tick touched, the graph returns exactly the
features that need recomputing, transitively. That is what makes incremental
updates correct rather than merely fast: recompute too little and values go
stale, too much and the engine cannot keep up with 150 frames a second.

Ordering is deterministic. Ties in the topological sort are broken by name, so
the compute order is identical on every run and on every machine — a
prerequisite for bit-for-bit reproducibility.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pmbtc.exceptions import ConfigError
from pmbtc.features.base import Feature


class DependencyCycleError(ConfigError):
    """The feature graph contains a cycle and cannot be ordered."""


class UnknownDependencyError(ConfigError):
    """A feature declares an input that no feature or raw source provides."""


@dataclass
class FeatureGraph:
    """A validated, topologically ordered set of features."""

    features: dict[str, Feature]
    order: tuple[str, ...]
    #: raw source key -> features that read it, directly or transitively.
    impact: dict[str, frozenset[str]]

    def __len__(self) -> int:
        return len(self.features)

    @property
    def raw_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.impact))

    def ordered(self) -> list[Feature]:
        return [self.features[name] for name in self.order]

    def affected_by(self, sources: set[str]) -> tuple[str, ...]:
        """Features needing recomputation after these raw sources changed.

        Returned in topological order, so the caller can compute straight down
        the list without re-sorting.
        """
        dirty: set[str] = set()
        for source in sources:
            dirty |= self.impact.get(source, frozenset())
        return tuple(name for name in self.order if name in dirty)

    def dependents_of(self, name: str) -> frozenset[str]:
        """Everything that would change if this feature changed."""
        reverse: dict[str, set[str]] = defaultdict(set)
        for feature in self.features.values():
            for dependency in feature.dependencies:
                reverse[dependency].add(feature.name)

        seen: set[str] = set()
        stack = [name]
        while stack:
            current = stack.pop()
            for dependent in reverse.get(current, set()):
                if dependent not in seen:
                    seen.add(dependent)
                    stack.append(dependent)
        return frozenset(seen)

    def to_dot(self) -> str:
        """Graphviz rendering, for the generated documentation."""
        lines = ["digraph features {", "  rankdir=LR;", '  node [shape=box];']
        for feature in self.ordered():
            group = feature.spec.group
            lines.append(f'  "{feature.name}" [group="{group}"];')
            for dependency in sorted(feature.dependencies):
                lines.append(f'  "{dependency}" -> "{feature.name}";')
            for raw in sorted(feature.raw_inputs):
                lines.append(f'  "raw:{raw}" [shape=ellipse];')
                lines.append(f'  "raw:{raw}" -> "{feature.name}";')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Mermaid rendering, so the docs render without a Graphviz install."""
        lines = ["graph LR"]
        for feature in self.ordered():
            node = feature.name.replace("-", "_")
            for dependency in sorted(feature.dependencies):
                lines.append(f"  {dependency.replace('-', '_')} --> {node}")
            for raw in sorted(feature.raw_inputs):
                lines.append(f"  raw_{raw.replace('.', '_')}([{raw}]) --> {node}")
        return "\n".join(lines)


def build_graph(features: list[Feature]) -> FeatureGraph:
    """Validate declarations and produce a deterministic compute order."""
    by_name: dict[str, Feature] = {}
    for feature in features:
        if feature.name in by_name:
            raise ConfigError(
                f"Duplicate feature definition: {feature.name!r}",
                context={"name": feature.name},
            )
        by_name[feature.name] = feature

    # Every declared dependency must exist.
    for feature in features:
        for dependency in feature.dependencies:
            if dependency not in by_name:
                raise UnknownDependencyError(
                    f"Feature {feature.name!r} depends on {dependency!r}, "
                    "which no feature provides",
                    context={"feature": feature.name, "missing": dependency},
                )

    order = _topological_order(by_name)
    impact = _impact_map(by_name, order)
    return FeatureGraph(features=by_name, order=order, impact=impact)


def _topological_order(by_name: dict[str, Feature]) -> tuple[str, ...]:
    """Kahn's algorithm with name-sorted tie-breaking.

    Sorting the ready set keeps the order stable across runs; a set-iteration
    order would silently vary between processes and break bit-for-bit replay.
    """
    indegree: dict[str, int] = {name: 0 for name in by_name}
    dependents: dict[str, list[str]] = defaultdict(list)
    for feature in by_name.values():
        for dependency in feature.dependencies:
            indegree[feature.name] += 1
            dependents[dependency].append(feature.name)

    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        newly_ready = []
        for dependent in dependents.get(name, []):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        if newly_ready:
            ready = sorted(ready + newly_ready)

    if len(order) != len(by_name):
        stuck = sorted(set(by_name) - set(order))
        raise DependencyCycleError(
            "Feature dependency cycle detected",
            context={"involved": ", ".join(stuck), "path": _find_cycle(by_name, stuck)},
        )
    return tuple(order)


def _find_cycle(by_name: dict[str, Feature], candidates: list[str]) -> str:
    """Name one concrete cycle, so the error is actionable.

    Iterative depth-first search with an explicit stack rather than recursion
    with a closure: the graph is small, but a cycle error should never itself
    risk a RecursionError.
    """
    for start in candidates:
        path: list[str] = []
        on_path: set[str] = set()
        # (node, whether we are entering or leaving it)
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                on_path.discard(name)
                if path and path[-1] == name:
                    path.pop()
                continue
            if name in on_path:
                cycle = [*path[path.index(name) :], name]
                return " -> ".join(cycle)
            feature = by_name.get(name)
            if feature is None:
                continue
            on_path.add(name)
            path.append(name)
            stack.append((name, True))
            for dependency in sorted(feature.dependencies, reverse=True):
                stack.append((dependency, False))
    return "unknown"


def _impact_map(
    by_name: dict[str, Feature], order: tuple[str, ...]
) -> dict[str, frozenset[str]]:
    """Raw source -> every feature transitively depending on it."""
    # Walking in topological order means a feature's dependencies are already
    # resolved when we reach it, so one pass suffices.
    sources_of: dict[str, set[str]] = {}
    for name in order:
        feature = by_name[name]
        sources = set(feature.raw_inputs)
        for dependency in feature.dependencies:
            sources |= sources_of.get(dependency, set())
        sources_of[name] = sources

    impact: dict[str, set[str]] = defaultdict(set)
    for name, sources in sources_of.items():
        for source in sources:
            impact[source].add(name)
    return {source: frozenset(names) for source, names in impact.items()}
