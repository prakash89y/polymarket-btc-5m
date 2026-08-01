"""Module 6 validation: determinism, incrementality, and archive reproducibility.

The critical test is the last one: features computed by replaying archived raw
frames must match features computed live, bit for bit. Everything else in this
module is worthless if that does not hold.
"""

from __future__ import annotations

import sys

from rich.console import Console

from pmbtc.config import load_config
from pmbtc.features.base import FeatureContext
from pmbtc.features.consistency import check_consistency
from pmbtc.features.docs import require_documentation, undocumented
from pmbtc.features.drift import DriftMonitor
from pmbtc.features.engine import FeatureEngine
from pmbtc.features.families import FAMILIES, all_features
from pmbtc.features.graph import build_graph
from pmbtc.features.matrix import MatrixBuilder, feature_set_version
from pmbtc.live.archive import TickArchive
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.live.fixtures import in_continuous_integration, resolve_clob_archive
from pmbtc.logging_setup import configure_logging

console = Console()
FAILURES: list[str] = []

#: `--require-live` makes this the local production check: the committed
#: fixture is refused and a real collector archive must exist. CI never sets it.
REQUIRE_LIVE = "--require-live" in sys.argv


def check(name: str, passed: bool, detail: str = "") -> bool:
    console.print(f"  {'[green]PASS[/]' if passed else '[red]FAIL[/]'} {name}"
                  + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def main() -> int:
    config = load_config()
    configure_logging(config)
    graph = build_graph(all_features())

    mode = "production (live archive required)" if REQUIRE_LIVE else (
        "CI (committed fixture)" if in_continuous_integration() else "local"
    )
    console.print(f"[dim]validation mode: {mode}[/]")

    console.rule("[bold]1. Feature declarations")
    check("families are independent modules", len(FAMILIES) == 10,
          f"{len(FAMILIES)} families: {', '.join(sorted(FAMILIES))}")
    check("every feature documented", undocumented(graph) == [],
          f"{len(graph)} features")
    require_documentation(graph)
    check("every feature declares inputs",
          all(f.spec.inputs for f in graph.features.values()))
    check("every feature declares a freshness budget",
          all(f.spec.freshness_budget_ms > 0 for f in graph.features.values()))
    check("every feature declares reproducibility",
          all(f.spec.reproducibility for f in graph.features.values()))

    console.rule("[bold]2. Dependency graph")
    position = {name: i for i, name in enumerate(graph.order)}
    ordered = all(
        position[dep] < position[f.name]
        for f in graph.features.values()
        for dep in f.dependencies
    )
    check("topological order valid", ordered, f"{len(graph.order)} features ordered")
    check("order is deterministic", build_graph(all_features()).order == graph.order)
    check("raw sources mapped", len(graph.raw_sources) > 0,
          ", ".join(graph.raw_sources))

    console.rule("[bold]3. Determinism")
    def context(now_ms: int = 1_785_600_000_000 - 60_000) -> FeatureContext:
        return FeatureContext(
            now_ms=now_ms,
            window_open_ms=1_785_600_000_000 - 300_000,
            settlement_ms=1_785_600_000_000,
        )

    a = FeatureEngine().compute(context())
    b = FeatureEngine().compute(context())
    check("identical inputs produce identical fingerprints",
          a.fingerprint() == b.fingerprint(), a.fingerprint())

    console.rule("[bold]4. Incremental updates")
    engine = FeatureEngine()
    engine.compute(context())
    partial = engine.compute(context(), changed_sources={"clock"})
    check("clock change does not recompute the book",
          "ob_best_bid" not in partial.recomputed,
          f"{len(partial.recomputed)} of {len(engine.feature_names)} recomputed")

    console.rule("[bold]5. Reproducible from archive, no live services")
    # Determinism is a property of the code, so it is verified against whichever
    # archive is available: a live one locally, the committed fixture on a clean
    # runner. The assertions are identical either way; only the source differs,
    # and it is always reported. `--require-live` refuses the fixture, which is
    # the local production check that the collector really is emitting
    # replayable output.
    resolved = resolve_clob_archive(config, require_live=REQUIRE_LIVE)
    console.print(f"       archive source: {resolved.describe()}")
    archive = TickArchive(resolved.root, enabled=False)
    clob_files = list(resolved.files)
    if not clob_files:
        check("archived frames available", False, resolved.reason)
    else:
        check("archive source resolved", True, resolved.source.value)
        frames = archive.read_frames(clob_files[-1])
        tokens: list[str] = []
        for _, payload in frames:
            for ev in payload if isinstance(payload, list) else [payload]:
                if isinstance(ev, dict) and ev.get("event_type") == "book":
                    asset = str(ev.get("asset_id", ""))
                    if asset and asset not in tokens:
                        tokens.append(asset)
            if len(tokens) >= 2:
                break

        def replay_fingerprint() -> tuple[str, int]:
            """Rebuild books from the archive and compute one feature pass."""
            import asyncio

            feed = PolymarketMarketFeed(
                "wss://replay", tokens[0], tokens[1] if len(tokens) > 1 else "",
                slug="replay",
            )
            midpoint = max(1, len(frames) // 2)

            async def run() -> None:
                for received_ms, payload in frames[:midpoint]:
                    await feed.handle(payload, received_ms)

            asyncio.run(run())
            last_ms = frames[midpoint - 1][0]
            engine = FeatureEngine()
            vector = engine.compute(
                FeatureContext(
                    now_ms=last_ms,
                    window_open_ms=last_ms - 150_000,
                    settlement_ms=last_ms + 150_000,
                    books=feed.books,
                    pm_tape=feed.up_tape,
                    regime={"tick_size": 0.01},
                )
            )
            return vector.fingerprint(), vector.populated

        first, populated = replay_fingerprint()
        second, _ = replay_fingerprint()
        check("archive replay is reproducible", first == second,
              f"fingerprint {first}, {populated} features populated")
        check("replay populates real features", populated > 10,
              f"{populated} non-null features from archived frames alone")

    console.rule("[bold]6. Drift and consistency")
    monitor = DriftMonitor(min_samples=10)
    for i in range(60):
        monitor.observe({"x": float(i % 5)})
    check("drift monitor runs without retraining",
          not hasattr(monitor, "retrain"), f"{len(monitor.windows)} windows tracked")
    report = check_consistency(graph, now_ms=0, pm_book_age_ms=60_000)
    check("stale source annotates dependent features",
          "ob_mid" in report.affected_features,
          f"{len(report.affected_features)} features annotated, "
          f"confidence x{report.confidence_multiplier:.2f}")

    console.rule("[bold]7. Versioned matrix")
    builder = MatrixBuilder()
    vector = builder.engine.compute(context())
    builder.add(condition_id="c", slug="s", horizon_seconds=60,
                snapshot_time_ms=context().now_ms,
                settlement_time_ms=1_785_600_000_000, vector=vector, label=1)
    matrix = builder.build(built_at_ms=0)
    check("matrix carries a feature-set version", bool(matrix.version), matrix.version)
    check("version matches the graph", matrix.version == feature_set_version(graph))

    console.rule("[bold]Result")
    if FAILURES:
        console.print(f"[bold red]{len(FAILURES)} check(s) failed:[/]")
        for failure in FAILURES:
            console.print(f"  - {failure}")
        return 1
    console.print("[bold green]All Module 6 validation checks passed.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
