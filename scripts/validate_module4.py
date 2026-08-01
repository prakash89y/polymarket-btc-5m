"""Module 4 validation over live-collected data.

Runs every acceptance check the dataset must pass before Module 4 is called
done, against data collected from the live Polymarket API -- no simulated
inputs anywhere in this script.
"""

from __future__ import annotations

import json
import sqlite3
import sys

from rich.console import Console

from pmbtc.config import load_config
from pmbtc.dataset import (
    DatasetExporter,
    LeakageGuard,
    assert_no_label_leakage,
    open_dataset_store,
    render_stats,
    scan_monotonicity,
    scan_snapshots,
)
from pmbtc.dataset.stats import stats_for_store
from pmbtc.logging_setup import configure_logging
from pmbtc.utils.timeutils import isoformat

console = Console()
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    mark = "[green]PASS[/]" if passed else "[red]FAIL[/]"
    console.print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def main() -> int:
    config = load_config()
    configure_logging(config)
    store = open_dataset_store(config)
    horizons = tuple(config.dataset.snapshot_horizons_s)

    markets = store.markets()
    snapshots = store.snapshots()

    console.rule("[bold]1. Collection")
    check("markets collected", bool(markets), f"{len(markets)} market(s)")
    check("snapshots collected", bool(snapshots), f"{len(snapshots)} snapshot(s)")

    console.rule("[bold]2. Complete timeline (T-300 through settlement)")
    complete = []
    for market in markets:
        got = set(store.timeline(market.condition_id))
        if set(horizons).issubset(got):
            complete.append(market)
    check(
        "at least one market with every scheduled snapshot",
        bool(complete),
        f"{len(complete)} complete of {len(markets)}",
    )
    for market in complete:
        got = sorted(store.timeline(market.condition_id), reverse=True)
        console.print(f"       {market.slug}: {got}")

    console.rule("[bold]3. Missing snapshots")
    missing_total = 0
    for market in markets:
        got = set(store.timeline(market.condition_id))
        gaps = sorted(set(horizons) - got, reverse=True)
        if gaps:
            missing_total += len(gaps)
            console.print(f"       {market.slug}: missing {gaps}")
    console.print(
        f"       total missing {missing_total} of {len(markets) * len(horizons)} scheduled"
    )

    console.rule("[bold]4. Leakage detection")
    guard = LeakageGuard(config.dataset.observation_tolerance_ms)
    findings = scan_snapshots(snapshots, guard)
    check("no leaked observations", not findings, f"{len(findings)} finding(s)")
    mono = scan_monotonicity(snapshots)
    check("snapshot ordering is monotone", not mono, f"{len(mono)} finding(s)")
    label_leak = assert_no_label_leakage(snapshots, {m.condition_id: m for m in markets})
    check("no observation postdates resolution", not label_leak, f"{len(label_leak)} finding(s)")
    for finding in (*findings[:5], *mono[:5], *label_leak[:5]):
        console.print(f"       [red]{finding}[/]")

    console.rule("[bold]5. Label integrity")
    labelled = [m for m in markets if m.is_labelled]
    check("official outcomes stored", bool(labelled), f"{len(labelled)} labelled")
    for market in labelled:
        console.print(
            f"       {market.slug}: outcome={market.official_outcome.value} "
            f"yes={market.yes_final_price} no={market.no_final_price} "
            f"source={market.resolution_source} resolved={isoformat(market.resolved_at_ms)}"
        )
    check(
        "labels come only from Polymarket",
        all(m.resolution_source == "polymarket_official" for m in labelled),
        "resolution_source on every labelled market",
    )

    console.rule("[bold]6. Statistics report")
    stats = stats_for_store(store, horizons)
    render_stats(stats, console)

    console.rule("[bold]7. Export formats")
    exporter = DatasetExporter(config, store)
    results = {}
    for fmt in ("parquet", "arrow", "csv", "sqlite"):
        try:
            result = exporter.export(fmt, name="validation")  # type: ignore[arg-type]
            results[fmt] = result
            check(f"{fmt} export", result.path.exists(),
                  f"{result.rows} rows -> {result.path.name}")
        except Exception as exc:
            check(f"{fmt} export", False, str(exc))

    if "parquet" in results:
        import pyarrow.parquet as pq

        table = pq.read_table(results["parquet"].path)
        check(
            "parquet readable with label column",
            "label" in table.column_names and table.num_rows > 0,
            f"{table.num_rows} rows x {table.num_columns} cols",
        )
    if "arrow" in results:
        import pyarrow as pa

        with pa.OSFile(str(results["arrow"].path), "rb") as source:
            table = pa.ipc.open_file(source).read_all()
        check("arrow readable", table.num_rows > 0, f"{table.num_rows} rows")
    if "sqlite" in results:
        with sqlite3.connect(results["sqlite"].path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        check("sqlite queryable", count > 0, f"{count} rows")

    console.rule("[bold]8. JSONL source of truth")
    root = config.resolved_path(config.dataset.root_dir)
    for name in ("markets.jsonl", "snapshots.jsonl"):
        path = root / name
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        ok = bool(lines) and all(json.loads(line) for line in lines if line.strip())
        check(f"{name} valid", ok, f"{len(lines)} line(s)")

    console.rule("[bold]9. Dataset version / reproducibility")
    if "parquet" in results:
        manifest = results["parquet"].manifest
        console.print_json(data=manifest.as_dict())
        check("manifest carries a git commit", bool(manifest.git_commit), manifest.git_commit)
        check("manifest carries a config hash", bool(manifest.config_hash), manifest.config_hash)
        check(
            "schema versions recorded",
            bool(manifest.feature_schema_version and manifest.settlement_schema_version),
            f"features={manifest.feature_schema_version} "
            f"settlement={manifest.settlement_schema_version}",
        )

    console.rule("[bold]Result")
    if FAILURES:
        console.print(f"[bold red]{len(FAILURES)} check(s) failed:[/]")
        for failure in FAILURES:
            console.print(f"  - {failure}")
        return 1
    console.print("[bold green]All Module 4 validation checks passed.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
