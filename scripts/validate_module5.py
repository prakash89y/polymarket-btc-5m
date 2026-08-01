"""Module 5 validation over live-collected data.

Checks the architectural rules agreed before Module 5, against data collected
from live sockets — not simulated inputs.
"""

from __future__ import annotations

import sys
from collections import Counter

from rich.console import Console

from pmbtc.config import load_config
from pmbtc.dataset import open_dataset_store, scan_snapshots
from pmbtc.features.registry import REGISTRY, FeatureTier, ReproducibilityPolicy
from pmbtc.live import TickArchive
from pmbtc.live.clob import PolymarketMarketFeed
from pmbtc.logging_setup import configure_logging
from pmbtc.models import readiness_for_store

console = Console()
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    console.print(f"  {'[green]PASS[/]' if passed else '[red]FAIL[/]'} {name}"
                  + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def main() -> int:
    config = load_config()
    configure_logging(config)
    store = open_dataset_store(config)
    snapshots = store.snapshots()

    console.rule("[bold]1. Live feed data reached the dataset")
    live_sources = Counter()
    for snapshot in snapshots:
        for obs in snapshot.observations:
            live_sources[obs.data_source] += 1
    check("polymarket_clob observations present", live_sources.get("polymarket_clob", 0) > 0,
          f"{live_sources.get('polymarket_clob', 0)} observations")
    check("binance_spot observations present", live_sources.get("binance_spot", 0) > 0,
          f"{live_sources.get('binance_spot', 0)} observations")
    console.print(f"       sources: {dict(live_sources)}")

    console.rule("[bold]2. Feature tiers and provenance")
    check("features registered", len(REGISTRY) > 0, f"{len(REGISTRY)} registered")
    console.print(f"       tiers: {REGISTRY.summary()}")
    check("real-time tier populated", len(REGISTRY.by_tier(FeatureTier.REAL_TIME)) > 0,
          f"{len(REGISTRY.by_tier(FeatureTier.REAL_TIME))} real-time features")

    anonymous = set()
    for snapshot in snapshots:
        for obs in snapshot.observations:
            if not REGISTRY.has(obs.name) and obs.data_source in {
                "polymarket_clob", "binance_spot"
            }:
                anonymous.add(obs.name)
    check("no anonymous live features", not anonymous, f"{sorted(anonymous)[:5]}")

    missing_provenance = [
        obs.name
        for snapshot in snapshots
        for obs in snapshot.observations
        if not obs.data_source or obs.event_time_ms <= 0 or obs.observation_time_ms <= 0
    ]
    check("every observation has full provenance", not missing_provenance,
          f"{len(missing_provenance)} incomplete")

    console.rule("[bold]3. Reproducibility policy")
    unreproducible = REGISTRY.unreproducible()
    check("no unreproducible feature is trainable", not unreproducible,
          f"{[s.name for s in unreproducible]}")
    stream_features = [
        s for s in REGISTRY.trainable()
        if s.reproducibility is ReproducibilityPolicy.ARCHIVED_STREAM
    ]
    check("stream features declared ARCHIVED_STREAM", bool(stream_features),
          f"{len(stream_features)} features depend on the tick archive")

    console.rule("[bold]4. Tick archive")
    archive = TickArchive(config.resolved_path(config.feeds.archive_dir), enabled=False)
    files = archive.files()
    total_bytes = sum(f.stat().st_size for f in files)
    check("archive written", bool(files),
          f"{len(files)} file(s), {total_bytes / 1024:.1f} KB")

    replayed = 0
    clob_frames = 0
    for path in files[:4]:
        frames = archive.read_frames(path)
        replayed += len(frames)
        if "clob" in path.name:
            clob_frames += len(frames)
    check("archived frames replayable", replayed > 0, f"{replayed} frames read back")

    console.rule("[bold]5. Archive replay reconstructs a book")
    clob_files = [f for f in files if "clob" in f.name]
    if clob_files:
        frames = archive.read_frames(clob_files[0])

        # Learn BOTH token ids from the archive itself — the file contains
        # book events for each side of the market.
        tokens: list[str] = []
        for _, payload in frames:
            for ev in payload if isinstance(payload, list) else [payload]:
                if isinstance(ev, dict) and ev.get("event_type") == "book":
                    asset = str(ev.get("asset_id", ""))
                    if asset and asset not in tokens:
                        tokens.append(asset)
            if len(tokens) >= 2:
                break

        feed = PolymarketMarketFeed(
            "wss://replay",
            tokens[0] if tokens else "",
            tokens[1] if len(tokens) > 1 else "",
            slug="replay",
        )
        # Replay to mid-file: at end-of-file the market has settled and the
        # losing side legitimately has no bids, which says nothing about
        # whether reconstruction works.
        midpoint = max(1, len(frames) // 2)

        import asyncio

        async def replay() -> None:
            for received_ms, payload in frames[:midpoint]:
                await feed.handle(payload, received_ms)

        asyncio.run(replay())
        up, down = feed.books.up, feed.books.down
        rebuilt = up.is_two_sided and down.is_two_sided
        check(
            "book rebuilt from archive alone",
            rebuilt,
            f"tokens={len(tokens)} up={up.best_bid}/{up.best_ask} "
            f"({len(up.bids)}x{len(up.asks)}) down={down.best_bid}/{down.best_ask} "
            f"({len(down.bids)}x{len(down.asks)}) implied_up={feed.books.implied_up}",
        )
        check(
            "reconstructed books are complementary",
            feed.books.complement_gap is not None
            and abs(feed.books.complement_gap) < 0.05,
            f"complement_gap={feed.books.complement_gap}",
        )
    else:
        check("book rebuilt from archive alone", False, "no clob archive files")

    console.rule("[bold]6. Leakage still clean with live features")
    findings = scan_snapshots(snapshots)
    check("no leaked observations", not findings, f"{len(findings)} finding(s)")

    console.rule("[bold]7. Data quality vs Gamma-only collection")
    live_snaps = [
        s for s in snapshots
        if any(o.data_source == "polymarket_clob" for o in s.observations)
    ]
    gamma_snaps = [
        s for s in snapshots
        if any(o.data_source == "polymarket_gamma" for o in s.observations)
    ]
    if live_snaps:
        live_q = sum(s.quality_score for s in live_snaps) / len(live_snaps)
        live_c = sum(s.coverage for s in live_snaps) / len(live_snaps)
        console.print(f"       live-feed snapshots : n={len(live_snaps)} "
                      f"quality={live_q:.3f} coverage={live_c:.3f}")
        check("live snapshot quality above the training floor",
              live_q >= config.dataset.min_snapshot_quality,
              f"{live_q:.3f} >= {config.dataset.min_snapshot_quality}")
    if gamma_snaps:
        g_q = sum(s.quality_score for s in gamma_snaps) / len(gamma_snaps)
        g_c = sum(s.coverage for s in gamma_snaps) / len(gamma_snaps)
        console.print(f"       gamma-only snapshots: n={len(gamma_snaps)} "
                      f"quality={g_q:.3f} coverage={g_c:.3f}")

    console.rule("[bold]8. Model readiness gate blocks training")
    report = readiness_for_store(config, store)
    check("gate correctly refuses training on a small dataset", not report.ready,
          f"{len(report.failures)} unmet threshold(s)")
    for c in report.checks:
        console.print(f"       {c}")

    console.rule("[bold]Result")
    if FAILURES:
        console.print(f"[bold red]{len(FAILURES)} check(s) failed:[/]")
        for failure in FAILURES:
            console.print(f"  - {failure}")
        return 1
    console.print("[bold green]All Module 5 validation checks passed.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
