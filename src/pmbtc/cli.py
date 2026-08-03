"""Operator entry point.

Module 1 ships the commands that answer "is this box configured correctly?"
Later modules register their own sub-commands (``collect``, ``train``,
``backtest``, ``paper``, ``live``) against the same app object.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from pmbtc import __version__
from pmbtc.config import LIVE_INTERLOCK_ENV, BotSecrets, Config, load_config
from pmbtc.constants import RunMode
from pmbtc.exceptions import BotError
from pmbtc.gamma.discovery import DiscoveryResult
from pmbtc.gamma.runtime import DiscoveryStack
from pmbtc.logging_setup import configure_logging, get_logger
from pmbtc.utils.timeutils import current_window, isoformat, seconds_to_settlement

app = typer.Typer(
    name="pmbtc",
    help="Polymarket BTC 5-minute Up/Down trading bot.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to config YAML (default: $PMBTC_CONFIG)."),
]


def _load(path: Path | None) -> Config:
    try:
        return load_config(path)
    except BotError as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"pmbtc {__version__}")


@app.command("show-config")
def show_config(config: ConfigOption = None, section: str | None = None) -> None:
    """Dump the effective, fully-validated configuration.

    Secrets are not part of :class:`Config`, so this is always safe to paste
    into an issue.
    """
    cfg = _load(config)
    data = cfg.model_dump(mode="json")
    if section:
        if section not in data:
            console.print(f"[red]No such section:[/] {section}")
            raise typer.Exit(code=2)
        data = {section: data[section]}
    console.print_json(data=data)


@app.command()
def doctor(config: ConfigOption = None) -> None:
    """Pre-flight check: config validity, paths, credentials, safety interlocks."""
    cfg = _load(config)
    configure_logging(cfg)
    log = get_logger("pmbtc.doctor")
    cfg.ensure_directories()
    secrets = BotSecrets()

    table = Table(title="pmbtc pre-flight", show_lines=False)
    table.add_column("check")
    table.add_column("value")
    table.add_column("status")

    def row(name: str, value: object, ok: bool | None) -> None:
        mark = "[green]OK[/]" if ok else ("[yellow]--[/]" if ok is None else "[red]FAIL[/]")
        table.add_row(name, str(value), mark)

    open_ms, close_ms = current_window(window_ms=cfg.window_ms)

    row("mode", cfg.app.mode.value, True)
    row("window", f"{isoformat(open_ms)} -> {isoformat(close_ms)}", True)
    row("seconds to settlement", f"{seconds_to_settlement(window_ms=cfg.window_ms):.1f}", None)
    row("expected settlement source", cfg.settlement.expected_source.value, True)
    row("settlement verification required", cfg.settlement.require_verified,
        cfg.settlement.require_verified or cfg.app.mode is RunMode.BACKTEST)
    row("unknown source allowed", cfg.settlement.allow_unknown_source,
        not cfg.settlement.allow_unknown_source or cfg.app.mode is RunMode.BACKTEST)
    row("data dir", cfg.resolved_path(cfg.app.data_dir), True)
    row("log dir", cfg.resolved_path(cfg.app.log_dir), True)
    row("kill switch present", cfg.resolved_path(cfg.risk.kill_switch_file).exists(), None)

    # Credentials: presence only, never values.
    row("polymarket key", secrets.has("polymarket_private_key"),
        secrets.has("polymarket_private_key") or cfg.app.mode is not RunMode.LIVE)
    for field in ("binance_api_key", "coinglass_api_key", "glassnode_api_key"):
        row(field, secrets.has(field), None)

    armed = os.getenv(LIVE_INTERLOCK_ENV, "").strip().lower() in {"yes", "true", "1"}
    row(LIVE_INTERLOCK_ENV, armed, None)
    row("orders would be REAL", cfg.is_live, None)

    # Log first: the structlog sink is stderr and would otherwise interleave
    # into the middle of the rendered table.
    log.info("doctor.completed", mode=cfg.app.mode.value, is_live=cfg.is_live)
    console.print(table)
    if cfg.is_live:
        console.print("[bold red]LIVE MODE: orders placed by this process spend real funds.[/]")


@app.command("verify-settlement")
def verify_settlement(
    file: Annotated[
        Path,
        typer.Option("--file", "-f", help="JSON file: one Gamma market object, or a list."),
    ],
    config: ConfigOption = None,
    record: Annotated[
        bool, typer.Option("--record/--no-record", help="Persist specs to the spec store.")
    ] = False,
) -> None:
    """Run the settlement verification report over saved Gamma payloads.

    Live market scanning arrives with the Gamma client in Module 3; this command
    is the same engine driven from captured payloads, which is also how the
    report is regression-tested.
    """
    cfg = _load(config)
    configure_logging(cfg)
    cfg.ensure_directories()

    from pmbtc.settlement import (
        SettlementReport,
        SettlementVerifier,
        open_spec_store,
        parse_settlement_spec,
    )
    from pmbtc.settlement.parser import SettlementParseError

    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]Could not read {file}:[/] {exc}")
        raise typer.Exit(code=2) from exc

    markets = payload if isinstance(payload, list) else [payload]
    store = open_spec_store(cfg) if record else None

    results = []
    for raw in markets:
        try:
            spec = parse_settlement_spec(raw)
        except SettlementParseError as exc:
            console.print(f"[red]skipped unparseable payload:[/] {exc}")
            continue
        known = store.known_hash(spec.series_slug) if store else None
        result = SettlementVerifier(cfg, known_hash=known).verify(spec)
        if store:
            change = store.record(spec)
            if change:
                console.print(f"[bold red]SETTLEMENT CHANGE DETECTED:[/] {change}")
        results.append(result)

    report = SettlementReport(tuple(results))
    report.emit_log()
    report.render(console)

    # The exit code is the machine-readable form of "may trading begin?".
    if not report.all_passed:
        console.print(
            f"[yellow]{len(report.rejected)} of {len(report.results)} market(s) blocked.[/]"
        )
        raise typer.Exit(code=1)
    console.print("[green]All scanned markets verified — trading gate open.[/]")


@app.command()
def discover(
    config: ConfigOption = None,
    watch: Annotated[
        int, typer.Option("--watch", help="Repeat this many scans, one per poll interval.")
    ] = 1,
) -> None:
    """Discover live BTC 5-minute markets and run the full acceptance pipeline.

    Series-driven: no slug is constructed on the happy path, so a change in
    Polymarket's naming conventions cannot break discovery.
    """
    import asyncio

    from pmbtc.gamma.runtime import discovery_stack
    from pmbtc.metrics import METRICS

    cfg = _load(config)
    configure_logging(cfg)

    async def _run() -> int:
        accepted_total = 0
        async with discovery_stack(cfg) as stack:
            clock = stack.clock.describe()
            console.print(
                f"clock: [bold]{clock['status']}[/] offset={clock['offset_ms']}ms "
                f"±{clock['uncertainty_ms']}ms  sources={[s['source'] for s in clock['sources']]}"
            )
            for iteration in range(max(1, watch)):
                if iteration:
                    await asyncio.sleep(cfg.gamma.discovery_poll_seconds)
                result = await stack.discovery.scan()
                accepted_total += result.accepted_count
                _render_discovery(stack, result)
        console.print_json(data=METRICS.snapshot())
        return accepted_total

    accepted = asyncio.run(_run())
    if not accepted:
        console.print("[yellow]No tradeable market found.[/]")
        raise typer.Exit(code=1)


def _render_discovery(stack: DiscoveryStack, result: DiscoveryResult) -> None:
    """Render one discovery scan."""
    table = Table(title=f"Discovery ({result.source})", show_lines=False)
    for column in ("slug", "state", "settles (UTC)", "to settle", "bid/ask", "liq", "trade"):
        table.add_column(column, overflow="fold")

    clock = stack.clock
    for candidate in sorted(result.candidates, key=lambda c: c.settlement_ms):
        decision = clock.can_submit_order(candidate.settlement_ms)
        snapshot = candidate.snapshot
        quote = (
            f"{snapshot.best_bid:.3f}/{snapshot.best_ask:.3f}"
            if snapshot.two_sided
            else "-"
        )
        table.add_row(
            candidate.spec.slug,
            candidate.state.value,
            isoformat(candidate.settlement_ms),
            f"{decision.seconds_to_settlement:.0f}s",
            quote,
            f"{snapshot.liquidity_usdc:.0f}" if snapshot.liquidity_usdc else "-",
            "[green]YES[/]" if decision.allowed else f"[red]NO[/] ({decision.reason})",
        )
    console.print(table)

    for item in result.rejected:
        console.print(f"[red]rejected[/] {item.slug or item.condition_id[:12]}: {item.detail}")


@app.command()
def replay(
    config: ConfigOption = None,
    baseline: Annotated[
        Path, typer.Option("--baseline", help="Golden interpretation file.")
    ] = Path("tests/fixtures/replay_baseline.json"),
    update: Annotated[
        bool, typer.Option("--update", help="Rewrite the baseline (a reviewable diff).")
    ] = False,
) -> None:
    """Replay archived Gamma responses through the current parser.

    Fails if any previously-recorded market is now interpreted differently. That
    is the guarantee: a parser change cannot silently rewrite history.
    """
    from pmbtc.gamma.replay import ReplaySession

    cfg = _load(config)
    configure_logging(cfg)
    session = ReplaySession(cfg.resolved_path(cfg.gamma.archive_dir))

    if update:
        count = session.write_baseline(baseline)
        console.print(f"[yellow]Baseline rewritten[/] with {count} market(s) -> {baseline}")
        console.print("Review the diff before committing.")
        return

    report = session.compare_to_baseline(baseline)
    console.print(report.summary())
    for difference in report.differences:
        console.print(f"  [red]changed[/] {difference}")
    if report.new_markets:
        console.print(f"  [green]new[/] {len(report.new_markets)} market(s) not in baseline")
    if not report.stable:
        raise typer.Exit(code=1)


@app.command()
def collect(
    config: ConfigOption = None,
    minutes: Annotated[
        float, typer.Option("--minutes", help="Wall-clock minutes to collect for.")
    ] = 6.0,
) -> None:
    """Collect the historical dataset: discover, snapshot the countdown, label.

    Runs a precise per-market countdown (T-300 ... T-1). A snapshot that would
    be late is skipped and counted rather than backdated, because a backdated
    snapshot is leakage.
    """
    import asyncio

    from pmbtc.dataset import HistoricalCollector, open_dataset_store, render_stats
    from pmbtc.dataset.stats import stats_for_store
    from pmbtc.gamma.runtime import discovery_stack

    cfg = _load(config)
    configure_logging(cfg)

    async def _run() -> int:
        store = open_dataset_store(cfg)
        async with discovery_stack(cfg) as stack:
            collector = HistoricalCollector(
                cfg, stack.client, stack.clock, stack.discovery, store
            )
            console.print(
                f"clock: [bold]{stack.clock.status().value}[/] "
                f"offset={stack.clock.offset_ms:.0f}ms — collecting for {minutes:.1f} min"
            )
            stats = await collector.run(minutes * 60)
        console.print(
            f"markets={stats.markets_tracked} snapshots={stats.snapshots_written} "
            f"missed={stats.snapshots_missed} labels={stats.labels_written} "
            f"leakage_blocked={stats.leakage_blocked}"
        )
        render_stats(stats_for_store(store, tuple(cfg.dataset.snapshot_horizons_s)), console)
        return stats.snapshots_written

    if asyncio.run(_run()) == 0:
        console.print("[yellow]No snapshots captured.[/]")
        raise typer.Exit(code=1)


@app.command()
def run(
    config: ConfigOption = None,
    minutes: Annotated[
        float, typer.Option("--minutes", help="Run for N minutes; 0 means forever.")
    ] = 0.0,
) -> None:
    """Run continuous collection with live order-book feeds.

    This is the long-running service: it streams the CLOB book and trade tape
    for each upcoming market, captures the T-300 ... T-1 countdown from live
    data, archives every raw frame for reproducibility, and backfills official
    labels as markets settle.
    """
    import asyncio

    from pmbtc.dataset import open_dataset_store
    from pmbtc.gamma.runtime import discovery_stack
    from pmbtc.live import CollectionService

    cfg = _load(config)
    configure_logging(cfg)

    async def _run() -> None:
        store = open_dataset_store(cfg)
        async with discovery_stack(cfg) as stack:
            service = CollectionService(
                cfg, stack.client, stack.clock, stack.discovery, store
            )
            async with service:
                console.print(
                    f"clock={stack.clock.status().value} "
                    f"offset={stack.clock.offset_ms:.0f}ms — "
                    + ("running until interrupted" if minutes <= 0
                       else f"running for {minutes:.1f} min")
                )
                try:
                    stats = await service.run(minutes * 60 if minutes > 0 else None)
                except KeyboardInterrupt:  # pragma: no cover - operator action
                    stats = service.stats
                console.print(
                    f"windows={stats.windows_streamed} snapshots={stats.snapshots} "
                    f"labels={stats.labels} errors={stats.errors} "
                    f"feeds_opened={stats.feeds_opened}"
                )
                console.print_json(data=service.status())

    asyncio.run(_run())


@app.command("daily-summary")
def daily_summary(
    config: ConfigOption = None,
    write: Annotated[
        bool, typer.Option("--write/--no-write", help="Persist the summary to artifacts.")
    ] = True,
) -> None:
    """Daily dataset summary and progress toward the training thresholds."""
    from pmbtc.dataset import open_dataset_store
    from pmbtc.ops import build_summary, write_summary

    cfg = _load(config)
    configure_logging(cfg)
    summary = build_summary(cfg, open_dataset_store(cfg))
    summary.render(console)
    if write:
        console.print(f"written to {write_summary(cfg, summary)}")


@app.command()
def watch(config: ConfigOption = None) -> None:
    """Evaluate the alert conditions once and report.

    Fires only on the four conditions worth waking someone for: collection
    stopped, feed health degraded, readiness regressed, or data quality below
    the configured threshold.
    """
    from pmbtc.dataset import open_dataset_store
    from pmbtc.ops import (
        AlertState,
        alert_state_path,
        build_summary,
        dispatch,
        evaluate,
        read_heartbeat,
    )

    cfg = _load(config)
    configure_logging(cfg)
    heartbeat = read_heartbeat(cfg)
    summary = build_summary(cfg, open_dataset_store(cfg))
    state = AlertState(alert_state_path(cfg))
    alerts = evaluate(cfg, state, heartbeat=heartbeat, summary=summary)
    fired = dispatch(cfg, state, alerts)

    if heartbeat is not None:
        console.print(
            f"heartbeat {heartbeat.age_ms() / 1000:.0f}s old · pid={heartbeat.pid} · "
            f"sessions={heartbeat.sessions} · snapshots={heartbeat.snapshots}"
        )
    if not alerts:
        console.print("[green]No alert conditions active.[/]")
        return
    for alert in alerts:
        marker = "[red]" if alert.severity.value == "critical" else "[yellow]"
        suffix = " (new)" if alert in fired else " (already firing)"
        console.print(f"{marker}{alert.kind.value}[/]: {alert.message}{suffix}")
    raise typer.Exit(code=1)


@app.command()
def audit(config: ConfigOption = None) -> None:
    """Run the pre-training data audit.

    Eight checks for the ways a dataset silently becomes worthless. Training
    aborts on any failure; there is no override, because the correct response to
    a failed audit is to fix the data.
    """
    from pmbtc.dataset import open_dataset_store
    from pmbtc.dataset.audit import audit_store

    cfg = _load(config)
    configure_logging(cfg)
    report = audit_store(cfg, open_dataset_store(cfg))

    table = Table(title="Pre-training data audit", show_lines=False)
    for column in ("check", "severity", "detail"):
        table.add_column(column, overflow="fold")
    colours = {"failure": "[red]", "warning": "[yellow]", "info": "[green]"}
    for finding in report.findings:
        table.add_row(
            finding.check,
            f"{colours[finding.severity.value]}{finding.severity.value}[/]",
            finding.message,
        )
    console.print(table)
    for finding in (*report.failures, *report.warnings):
        if finding.examples:
            console.print(f"  {finding.check}: " + "; ".join(finding.examples[:5]))
    console.print(report.summary())
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def readiness(config: ConfigOption = None) -> None:
    """Check whether the dataset is ready for model training (Module 7 gate)."""
    from pmbtc.dataset import open_dataset_store
    from pmbtc.models import readiness_for_store

    cfg = _load(config)
    configure_logging(cfg)
    report = readiness_for_store(cfg, open_dataset_store(cfg))

    table = Table(title="Model readiness gate", show_lines=False)
    for column in ("check", "have", "need", "status"):
        table.add_column(column)
    for check in report.checks:
        table.add_row(
            check.name,
            f"{check.actual:g}",
            f"{check.required:g}",
            "[green]PASS[/]" if check.passed else "[red]FAIL[/]",
        )
    console.print(table)
    console.print(report.summary())
    if not report.ready:
        raise typer.Exit(code=1)


@app.command()
def baselines(
    config: ConfigOption = None,
    test_fraction: Annotated[
        float, typer.Option("--test-fraction", help="Chronological holdout share.")
    ] = 0.3,
) -> None:
    """Score the benchmark baselines on the collected dataset.

    Every future model must beat all of these on unseen data. The split is
    chronological, never random: a random split would let the model see the
    future of the same market.
    """
    import numpy as np

    from pmbtc.dataset import build_rows, open_dataset_store
    from pmbtc.models import run_baselines

    cfg = _load(config)
    configure_logging(cfg)
    store = open_dataset_store(cfg)
    markets = {m.condition_id: m for m in store.markets()}
    rows, _ = build_rows(markets, store.snapshots())
    if len(rows) < 4:
        console.print(
            f"[yellow]Only {len(rows)} labelled sample(s) — not enough to score baselines.[/]"
        )
        raise typer.Exit(code=1)

    feature_columns = sorted(
        {k for row in rows for k in row if k.startswith("f_")}
    )
    x = np.array([[_num(row.get(c)) for c in feature_columns] for row in rows], dtype=float)
    y = np.array([row["label"] for row in rows], dtype=int)
    market_prob = np.array(
        [_num(row.get("f_pm_implied_up") or row.get("f_pm_mid"), 0.5) for row in rows],
        dtype=float,
    )

    split = max(1, int(len(rows) * (1.0 - test_fraction)))
    report = run_baselines(
        x[:split], y[:split], x[split:], y[split:], market_prob[split:], cfg.model.random_seed
    )
    console.print(f"train={split} test={len(rows) - split} features={len(feature_columns)}")
    console.print(str(report))


def _num(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@app.command("dataset-label")
def dataset_label(config: ConfigOption = None) -> None:
    """Backfill official outcomes for markets that have settled since collection.

    Resolution lags settlement, so this runs as its own pass rather than being
    tied to a collection window. It only ever fills in a blank: an existing
    label is never changed.
    """
    import asyncio

    from pmbtc.dataset import HistoricalCollector, open_dataset_store
    from pmbtc.gamma.runtime import discovery_stack

    cfg = _load(config)
    configure_logging(cfg)

    async def _run() -> int:
        store = open_dataset_store(cfg)
        pending = len(store.unlabelled_markets())
        async with discovery_stack(cfg) as stack:
            collector = HistoricalCollector(
                cfg, stack.client, stack.clock, stack.discovery, store
            )
            written = await collector.resolve_pending_labels()
        console.print(
            f"labelled {written} of {pending} pending market(s); "
            f"{len(store.unlabelled_markets())} still awaiting resolution"
        )
        return written

    asyncio.run(_run())


@app.command("dataset-stats")
def dataset_stats(config: ConfigOption = None) -> None:
    """Report on the collected dataset."""
    from pmbtc.dataset import open_dataset_store, render_stats
    from pmbtc.dataset.stats import stats_for_store

    cfg = _load(config)
    configure_logging(cfg)
    store = open_dataset_store(cfg)
    if not store.markets():
        console.print("[yellow]Dataset is empty — run `pmbtc collect` first.[/]")
        raise typer.Exit(code=1)
    render_stats(stats_for_store(store, tuple(cfg.dataset.snapshot_horizons_s)), console)


@app.command("dataset-export")
def dataset_export(
    config: ConfigOption = None,
    fmt: Annotated[
        str, typer.Option("--format", help="parquet | arrow | csv | sqlite")
    ] = "",
    name: Annotated[str, typer.Option("--name", help="Base filename.")] = "dataset",
    include_unlabelled: Annotated[
        bool, typer.Option("--include-unlabelled", help="Also export markets without a label.")
    ] = False,
) -> None:
    """Export the training table with a reproducibility manifest.

    Re-runs the leakage scan over everything it is about to write, and refuses
    to export if anything is found.
    """
    from pmbtc.dataset import DatasetExporter, open_dataset_store

    cfg = _load(config)
    configure_logging(cfg)
    store = open_dataset_store(cfg)
    exporter = DatasetExporter(cfg, store)

    try:
        result = exporter.export(
            fmt or None,  # type: ignore[arg-type]
            name=name,
            labelled_only=not include_unlabelled,
        )
    except BotError as exc:
        console.print(f"[red]Export failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]{result}[/]")
    console.print_json(data=result.manifest.as_dict())


@app.command()
def train(
    config: ConfigOption = None,
    model_type: Annotated[
        str, typer.Option("--model", help="gradient_boosting | logistic | random_forest")
    ] = "gradient_boosting",
    scheme: Annotated[
        str, typer.Option("--scheme", help="expanding | rolling | walk_forward")
    ] = "expanding",
) -> None:
    """Train a model on the collected dataset.

    Blocked by the readiness gate and the data audit, in that order. Neither has
    an override: if the dataset is too small or unsound, the fix is more data or
    better data, not a flag.
    """
    import numpy as np

    from pmbtc.dataset import build_rows, open_dataset_store
    from pmbtc.models import ModelTrainer, SplitScheme, TrainingData, write_report

    cfg = _load(config)
    configure_logging(cfg)
    store = open_dataset_store(cfg)
    markets = store.markets()
    snapshots = store.snapshots()
    rows, _ = build_rows({m.condition_id: m for m in markets}, snapshots)

    if not rows:
        console.print("[yellow]No labelled rows available to train on.[/]")
        raise typer.Exit(code=1)

    feature_columns = sorted({k for row in rows for k in row if k.startswith("f_")})
    data = TrainingData(
        x=np.array([[_num(r.get(c)) for c in feature_columns] for r in rows], dtype=float),
        y=np.array([r["label"] for r in rows], dtype=int),
        timestamps=np.array([r["settlement_time_ms"] for r in rows], dtype=np.int64),
        groups=np.array([r["condition_id"] for r in rows]),
        market_prob=np.array(
            [_num(r.get("f_pm_implied_up") or r.get("f_ob_mid"), 0.5) for r in rows],
            dtype=float,
        ),
        feature_names=feature_columns,
        dataset_version=cfg.dataset.dataset_version,
        feature_schema_version="4.0",
    )

    trainer = ModelTrainer(cfg)
    result = trainer.train(
        data,
        model_type=model_type,
        markets=markets,
        snapshots=snapshots,
        scheme=SplitScheme(scheme),
        enforce_gates=True,
    )
    console.print(result.summary())
    if result.card:
        report = write_report(
            result,
            cfg.resolved_path(cfg.app.artifact_dir) / "reports" / f"{result.card.model_id}.json",
        )
        console.print(f"report: {report}")
    if not result.trained:
        raise typer.Exit(code=1)


@app.command()
def backtest(
    config: ConfigOption = None,
    model: Annotated[
        str, typer.Option("--model", help="market | logistic | gradient_boosting")
    ] = "logistic",
    train_fraction: Annotated[
        float, typer.Option("--train-fraction", help="Chronological share used to fit.")
    ] = 0.6,
    horizon: Annotated[
        int | None,
        typer.Option("--horizon", help="Decision horizon in seconds (default: latest allowed)."),
    ] = None,
    fill: Annotated[
        str, typer.Option("--fill", help="touch | mid | aggressive")
    ] = "touch",
    walk_forward: Annotated[
        bool, typer.Option("--walk-forward", help="Gate every configured lookback.")
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Strict walk-forward: refit the model at every fold."),
    ] = False,
    folds: Annotated[
        int, typer.Option("--folds", help="Folds for --strict.")
    ] = 4,
) -> None:
    """Simulate the strategy over settled windows and apply the deployment gate.

    The split is chronological and on a market boundary, so no market ever
    appears in both the fit and the evaluation. ``--model market`` is the null
    control: it forecasts what the book forecasts, so it should place no trades.
    """
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pmbtc.backtest import (
        BacktestEngine,
        EstimatorModel,
        FillModel,
        MarketProbabilityModel,
        ProbabilityModel,
        evaluate_backtest,
        run_strict_walk_forward,
        run_walk_forward,
    )
    from pmbtc.dataset import build_rows, open_dataset_store
    from pmbtc.models.baselines import GradientBoostingBaseline, LogisticBaseline
    from pmbtc.trading.costs import CostModel

    cfg = _load(config)
    configure_logging(cfg)
    store = open_dataset_store(cfg)
    rows, _ = build_rows({m.condition_id: m for m in store.markets()}, store.snapshots())
    if not rows:
        console.print("[yellow]No labelled rows to backtest.[/]")
        raise typer.Exit(code=1)

    columns = sorted({k for r in rows for k in r if k.startswith("f_")})
    seed = cfg.model.random_seed

    def fit(train_rows: Sequence[Mapping[str, object]]) -> ProbabilityModel:
        """Fit the chosen strategy on exactly the rows given, and nothing else."""
        if model == "market":
            return MarketProbabilityModel()
        x = np.array([[_num(r.get(c)) for c in columns] for r in train_rows], dtype=float)
        y = np.array([r["label"] for r in train_rows], dtype=int)
        if len(set(y.tolist())) < 2:
            # One class in the training slice: nothing to learn, so fall back to
            # the market rather than fitting a degenerate constant.
            return MarketProbabilityModel()
        estimator = (
            GradientBoostingBaseline(seed)
            if model == "gradient_boosting"
            else LogisticBaseline(seed)
        )
        estimator.fit(x, y)
        return EstimatorModel(estimator, columns, name=model)

    engine = BacktestEngine(
        cfg,
        fill_model=FillModel(
            CostModel(cfg.costs), style=fill, pessimistic=cfg.backtest.pessimistic_fill  # type: ignore[arg-type]
        ),
    )

    # Strict walk-forward refits at every fold, so it consumes the whole
    # history rather than a single hand-placed split.
    if strict:
        swf = run_strict_walk_forward(
            cfg, rows, fit, engine=engine, model_name=model, n_folds=folds,
            decision_horizon_seconds=horizon,
        )
        console.print(swf.render())
        if not swf.folds:
            console.print(
                "[yellow]Not enough settled markets for a strict walk-forward yet.[/]"
            )
        out = (
            cfg.resolved_path(cfg.app.artifact_dir)
            / "reports"
            / f"backtest-strict-{model}-{fill}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(swf.as_dict(), indent=2), encoding="utf-8")
        console.print(f"\nreport: {out}")
        if not swf.approved:
            raise typer.Exit(code=1)
        return

    # Split on a market boundary: a market whose T-240 row trained the model
    # must not have its T-60 row evaluated by it.
    market_order = list(dict.fromkeys(r["condition_id"] for r in rows))
    cut = max(1, int(len(market_order) * train_fraction))
    train_ids = set(market_order[:cut])
    train_rows = [r for r in rows if r["condition_id"] in train_ids]
    test_rows = [r for r in rows if r["condition_id"] not in train_ids]
    if not test_rows:
        console.print("[yellow]Nothing left to evaluate after the split.[/]")
        raise typer.Exit(code=1)

    strategy = fit(train_rows)
    console.print(
        f"markets: train={len(train_ids)} test={len(market_order) - len(train_ids)} "
        f"| rows: train={len(train_rows)} test={len(test_rows)} | fill={fill}"
    )

    if walk_forward:
        wf = run_walk_forward(
            cfg, test_rows, strategy, engine=engine, model_name=model,
            decision_horizon_seconds=horizon,
        )
        console.print(wf.render())
        payload: dict[str, object] = wf.as_dict()
        approved = wf.approved
    else:
        result = engine.run(
            test_rows, strategy, model_name=model, decision_horizon_seconds=horizon
        )
        report = evaluate_backtest(cfg, result)
        console.print(report.render())
        payload = report.as_dict()
        approved = report.approved

    out = cfg.resolved_path(cfg.app.artifact_dir) / "reports" / f"backtest-{model}-{fill}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\nreport: {out}")
    if not approved:
        raise typer.Exit(code=1)


@app.command("paper-report")
def paper_report(
    config: ConfigOption = None,
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit the report as JSON.")
    ] = False,
) -> None:
    """Score the paper-trading record and apply the promotion gate.

    Reads only the journal, so it is safe to run at any time, including while
    paper trading is live. Exits non-zero when the gate refuses promotion,
    which is the machine-readable form of "live mode may not arm".
    """
    from pmbtc.paper import build_report, open_journal

    cfg = _load(config)
    configure_logging(cfg)
    journal = open_journal(cfg)
    runs = journal.runs()
    if not runs:
        console.print("[yellow]No paper decisions recorded yet.[/]")
        raise typer.Exit(code=1)

    report = build_report(cfg, runs)
    if json_out:
        console.print_json(data=report.as_dict())
    else:
        console.print(report.render())

    out = cfg.resolved_path(cfg.app.artifact_dir) / "reports" / "paper.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    if not json_out:
        console.print(f"\nreport: {out}")
    if not report.promotion_approved:
        raise typer.Exit(code=1)


@app.command()
def paper(
    config: ConfigOption = None,
    minutes: Annotated[
        float, typer.Option("--minutes", help="Run for this long; 0 runs until stopped.")
    ] = 60.0,
) -> None:
    """Paper-trade live Polymarket BTC 5-minute markets. Places no orders.

    Runs the production decision path — the same
    ``evaluate_opportunity`` call the backtest makes — against the live CLOB
    book, records every window traded or skipped, and settles against
    Polymarket's own resolution.

    No wallet, no signing, no Polygon transaction. The only difference from live
    trading is that the fill is computed rather than requested.
    """
    import asyncio

    from pmbtc.gamma.runtime import discovery_stack
    from pmbtc.paper import LiveQuoteSource, PaperTradingEngine, open_journal
    from pmbtc.paper.runner import PaperRunner

    cfg = _load(config)
    configure_logging(cfg)
    cfg.ensure_directories()

    if cfg.is_live:
        console.print("[bold red]Refusing to paper-trade while mode is live.[/]")
        raise typer.Exit(code=2)

    async def _run() -> None:
        async with discovery_stack(cfg) as stack:
            quotes = LiveQuoteSource()
            engine = PaperTradingEngine(
                cfg, open_journal(cfg), quotes=quotes, clock=stack.clock
            )
            runner = PaperRunner(cfg, stack, engine, quotes)
            console.print(
                f"paper trading — bankroll {cfg.paper.initial_bankroll_usdc:.0f} USDC, "
                f"decision horizon T-{engine.decision_horizon()}s, "
                f"clock={stack.clock.status().value}"
            )
            stats = await runner.run(None if minutes <= 0 else minutes * 60.0)
            console.print(
                f"evaluated={stats.evaluated} traded={stats.traded} "
                f"skipped={stats.skipped} settled={stats.settled}"
            )

    asyncio.run(_run())


@app.command()
def supervise(
    config: ConfigOption = None,
    max_restarts: Annotated[
        int | None,
        typer.Option("--max-restarts", help="Stop after this many restarts (testing)."),
    ] = None,
) -> None:
    """Keep exactly one collector running, restarting it when it stops.

    This is what a Scheduled Task runs at boot. It holds a machine-wide lock, so
    a second invocation exits rather than starting a duplicate collector against
    the same append-only store.
    """
    from pmbtc.ops.supervisor import CollectorSupervisor

    cfg = _load(config)
    configure_logging(cfg)
    cfg.ensure_directories()
    supervisor = CollectorSupervisor(cfg, config_path=config)
    raise typer.Exit(code=supervisor.run(max_restarts=max_restarts))


@app.command("supervisor-status")
def supervisor_status(config: ConfigOption = None) -> None:
    """Report the five supervised health checks without changing anything.

    Used by ``scripts/verify_supervisor.ps1`` and safe to run at any time.
    """
    from pmbtc.ops.supervisor import InstanceLock, check_health

    cfg = _load(config)
    configure_logging(cfg)
    # If we can take the lock, nothing is supervising; if we cannot, something
    # is. Released immediately either way so this never blocks a real start.
    probe = InstanceLock(directory=cfg.resolved_path(cfg.app.data_dir))
    supervised = not probe.acquire()
    probe.release()

    report = check_health(cfg, child=None, lock_held=supervised)
    table = Table(title="collector supervision", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for check in report.checks:
        if check.name == "collector_process":
            continue  # only the supervisor itself holds the child handle
        mark = "[green]OK[/]" if check.healthy else "[red]FAIL[/]"
        table.add_row(check.name, mark, check.detail)
    table.add_row(
        "supervisor",
        "[green]OK[/]" if supervised else "[red]FAIL[/]",
        "running" if supervised else "not running",
    )
    console.print(table)
    if not supervised:
        raise typer.Exit(code=1)


@app.command("edge-scan")
def edge_scan(
    config: ConfigOption = None,
    model: Annotated[
        str, typer.Option("--model", help="market | logistic | gradient_boosting")
    ] = "logistic",
    folds: Annotated[int, typer.Option("--folds", help="Walk-forward folds per horizon.")] = 4,
    trust: Annotated[
        float | None,
        typer.Option("--trust", help="Override prediction.model_trust for this scan."),
    ] = None,
) -> None:
    """Module 8.5: is there evidence of genuine edge, at any horizon?

    Every horizon is evaluated by strict walk-forward with the model refitted
    inside each fold, so no horizon is ever chosen using data it was scored on.
    A horizon is called STABLE only if it is profitable, profitable in most
    folds, *and* beats the book's own forecast — one of those alone is a
    coincidence, a trap, or unexplained.

    Places no orders and simulates no live trading.
    """
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pmbtc.backtest import (
        BacktestEngine,
        EstimatorModel,
        MarketProbabilityModel,
        ProbabilityModel,
        scan_horizons,
    )
    from pmbtc.dataset import build_rows, open_dataset_store
    from pmbtc.models.baselines import GradientBoostingBaseline, LogisticBaseline

    cfg = _load(config)
    if trust is not None:
        cfg = _load(config).model_copy(
            update={"prediction": cfg.prediction.model_copy(update={"model_trust": trust})}
        )
    configure_logging(cfg)
    store = open_dataset_store(cfg)
    rows, _ = build_rows({m.condition_id: m for m in store.markets()}, store.snapshots())
    if not rows:
        console.print("[yellow]No labelled rows to scan.[/]")
        raise typer.Exit(code=1)

    columns = sorted({k for r in rows for k in r if k.startswith("f_")})
    seed = cfg.model.random_seed

    def fit(train_rows: Sequence[Mapping[str, object]]) -> ProbabilityModel:
        if model == "market":
            return MarketProbabilityModel()
        x = np.array([[_num(r.get(c)) for c in columns] for r in train_rows], dtype=float)
        y = np.array([r["label"] for r in train_rows], dtype=int)
        if len(set(y.tolist())) < 2:
            return MarketProbabilityModel()
        estimator = (
            GradientBoostingBaseline(seed)
            if model == "gradient_boosting"
            else LogisticBaseline(seed)
        )
        estimator.fit(x, y)
        return EstimatorModel(estimator, columns, name=model)

    markets = len({r["condition_id"] for r in rows})
    console.print(
        f"markets={markets} rows={len(rows)} model={model} folds={folds} "
        f"trust={cfg.prediction.model_trust:.2f} "
        f"ceiling={cfg.prediction.max_disagreement_logits:.2f} logits"
    )
    report = scan_horizons(
        cfg, rows, fit, engine=BacktestEngine(cfg), model_name=model, n_folds=folds
    )
    console.print(report.render())

    out = cfg.resolved_path(cfg.app.artifact_dir) / "reports" / f"edge-scan-{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    console.print(f"\nreport: {out}")
    if not report.any_evidence_of_edge:
        raise typer.Exit(code=1)


@app.command()
def paths(config: ConfigOption = None) -> None:
    """Create and list the runtime directories."""
    cfg = _load(config)
    cfg.ensure_directories()
    for label, path in (
        ("data", cfg.app.data_dir),
        ("logs", cfg.app.log_dir),
        ("artifacts", cfg.app.artifact_dir),
        ("parquet", cfg.storage.parquet_dir),
        ("raw", cfg.storage.raw_dir),
    ):
        console.print(f"{label:>10}  {cfg.resolved_path(path).resolve()}")


if __name__ == "__main__":  # pragma: no cover
    app()
