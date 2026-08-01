"""Module 8 validation: the backtest cannot flatter itself.

Every check here is a property that, if it broke, would let the system report a
profit it would not have earned. They are asserted on synthetic windows with
known answers rather than on the live dataset, so the script gives the same
verdict on a clean checkout as it does on this machine.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console

from pmbtc.backtest import (
    BacktestEngine,
    ConstantModel,
    FillModel,
    MarketProbabilityModel,
    compute_metrics,
    decompose,
    evaluate_backtest,
    run_strict_walk_forward,
)
from pmbtc.config import Config
from pmbtc.constants import Outcome, SkipReason
from pmbtc.logging_setup import configure_logging
from pmbtc.trading import CostModel, DecisionEngine, PositionSizer, Quote

console = Console()
FAILURES: list[str] = []

SETTLE0 = 1_785_600_000_000
WINDOW_MS = 300_000
HORIZONS = (300, 240, 180, 120, 60, 30, 15, 5)


def check(name: str, passed: bool, detail: str = "") -> bool:
    console.print(
        f"  {'[green]PASS[/]' if passed else '[red]FAIL[/]'} {name}"
        + (f" — {detail}" if detail else "")
    )
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def rows(n_markets: int, *, label: int | None = None, depth: float = 500.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(n_markets):
        outcome = label if label is not None else i % 2
        settlement = SETTLE0 + i * WINDOW_MS
        for horizon in HORIZONS:
            out.append(
                {
                    "condition_id": f"m{i:04d}",
                    "slug": f"btc-updown-5m-{settlement // 1000}",
                    "horizon_seconds": horizon,
                    "settlement_time_ms": settlement,
                    "label": outcome,
                    "official_outcome": "up" if outcome == 1 else "down",
                    "f_ob_best_bid": 0.49,
                    "f_ob_best_ask": 0.51,
                    "f_ob_mid": 0.50,
                    "f_lq_depth_bid_usdc": depth,
                    "f_lq_depth_ask_usdc": depth,
                    "lat_ob_best_bid": 200,
                    "lat_ob_best_ask": 200,
                }
            )
    return out


def main() -> int:
    config = Config()
    configure_logging(config)
    quote = Quote(best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0)

    console.print("\n[bold]1. The cost model charges the spread in both directions[/]")
    costs = CostModel(config.costs)
    up = costs.touch_price(Outcome.UP, quote)
    down = costs.touch_price(Outcome.DOWN, quote)
    mid = quote.mid or 0.0
    check("buying UP lifts the ask", up == 0.51, f"{up}")
    check("buying DOWN pays 1 - bid", down == 0.51, f"{down}")
    check(
        "neither direction is filled at the mid",
        up is not None and down is not None and up > mid and down > mid,
    )

    console.print("\n[bold]2. Abstention is the default[/]")
    engine = DecisionEngine(config)
    flat = engine.decide(
        model_prob_up=0.5, quote=quote, seconds_into_window=120, seconds_to_settlement=180
    )
    check("no edge means no trade", not flat.trade, str(flat.skip_reason))
    check("the refusal is named", flat.skip_reason is not None)
    strong = engine.decide(
        model_prob_up=0.80, quote=quote, seconds_into_window=120, seconds_to_settlement=180
    )
    check("a real edge does trade", strong.trade, str(strong))

    console.print("\n[bold]3. Sizing is capped below Kelly[/]")
    stake = PositionSizer(config).size(bankroll_usdc=1_000.0, model_prob=0.9, entry_price=0.5)
    check(
        "the per-trade risk cap binds before Kelly does",
        stake.capped_by == "max_risk_per_trade",
        f"kelly_full={stake.kelly_full:.3f} -> {stake.usdc:.2f} USDC",
    )
    check(
        "stake never exceeds max_risk_per_trade of bankroll",
        stake.usdc <= config.sizing.max_risk_per_trade * 1_000.0 + 1e-9,
    )

    console.print("\n[bold]4. Fills respect the book that was recorded[/]")
    empty = Quote(best_bid=0.49, best_ask=0.51, bid_depth_usdc=0.0, ask_depth_usdc=0.0)
    unfilled = FillModel(costs, pessimistic=True).execute(
        outcome=Outcome.UP, quote=empty, stake_usdc=50.0
    )
    check(
        "an unknown book is not assumed to be deep",
        not unfilled.filled and unfilled.skip_reason is SkipReason.INSUFFICIENT_LIQUIDITY,
    )
    thin = Quote(best_bid=0.49, best_ask=0.51, bid_depth_usdc=25.0, ask_depth_usdc=25.0)
    capped = FillModel(costs, pessimistic=True).execute(
        outcome=Outcome.UP, quote=thin, stake_usdc=200.0
    )
    check(
        "a stake beyond the depth is capped and flagged partial",
        capped.fill is not None and capped.fill.partial and capped.fill.notional_usdc == 25.0,
    )

    console.print("\n[bold]5. The null strategy earns nothing[/]")
    null = BacktestEngine(config).run(rows(40), MarketProbabilityModel(), model_name="market")
    check(
        "forecasting the book's own price places no trades",
        not null.trades,
        f"{len(null.trades)} trade(s)",
    )
    check(
        "bankroll is untouched",
        null.ending_bankroll_usdc == null.starting_bankroll_usdc,
    )

    console.print("\n[bold]6. No survivorship, no look-ahead[/]")
    check(
        "every window is recorded, traded or not",
        null.evaluated == 40 and sum(null.skip_histogram().values()) == 40,
        f"{null.evaluated} evaluated",
    )
    win = BacktestEngine(config).run(rows(20, label=1), ConstantModel(0.95))
    lose = BacktestEngine(config).run(rows(20, label=0), ConstantModel(0.95))
    check(
        "the label never reaches the decision",
        win.windows[0].decision.as_dict() == lose.windows[0].decision.as_dict(),
    )
    check(
        "but it does reach the P&L",
        win.windows[0].pnl_usdc != lose.windows[0].pnl_usdc,
    )

    console.print("\n[bold]7. Determinism[/]")
    data = rows(40)
    a = BacktestEngine(config).run(data, ConstantModel(0.8))
    b = BacktestEngine(config).run(list(reversed(data)), ConstantModel(0.8))
    check(
        "row order does not change the result",
        [w.as_dict() for w in a.windows] == [w.as_dict() for w in b.windows],
    )

    console.print("\n[bold]8. Risk stops a confidently wrong model[/]")
    doomed = BacktestEngine(config).run(rows(40, label=0), ConstantModel(0.95))
    check(
        "a losing streak triggers a stand-down",
        len(doomed.trades) <= config.risk.max_consecutive_losses + 1,
        f"{len(doomed.trades)} trades before standing down",
    )
    check(
        "the stand-down is recorded as a risk limit",
        SkipReason.RISK_LIMIT.value in doomed.skip_histogram(),
    )

    console.print("\n[bold]9. The gate refuses to conclude from a small sample[/]")
    lucky = BacktestEngine(config).run(rows(10, label=1), ConstantModel(0.95))
    report = evaluate_backtest(config, lucky)
    metrics = compute_metrics(lucky)
    check(
        "a profitable but tiny run is inconclusive",
        metrics.net_pnl_usdc > 0 and not report.conclusive and not report.approved,
        f"pnl {metrics.net_pnl_usdc:+.2f} on {metrics.trades} trades",
    )
    check(
        "break-even is the price paid, not 0.5",
        metrics.breakeven_win_rate > 0.5,
        f"{metrics.breakeven_win_rate:.4f}",
    )

    console.print("\n[bold]10. The edge decomposition reconciles exactly[/]")
    for label, name in ((1, "always up"), (0, "always down"), (None, "mixed")):
        run = BacktestEngine(config).run(rows(40, label=label), ConstantModel(0.9))
        decomposition = decompose(run, config.costs)
        realised = compute_metrics(run).net_pnl_usdc
        check(
            f"ladder sums to realised P&L ({name})",
            decomposition.reconciles(realised),
            f"error {decomposition.reconciliation_error_usdc:+.2e} USDC",
        )

    console.print("\n[bold]11. Costs are attributed, not assumed away[/]")
    winning = BacktestEngine(config).run(rows(40, label=1), ConstantModel(0.9))
    d = decompose(winning, config.costs)
    check(
        "crossing the spread costs money",
        d.spread_cost_usdc > 0,
        f"{d.spread_cost_usdc:.2f} USDC",
    )
    check(
        "raw forecast edge exceeds realised profit",
        d.raw_edge_usdc > d.net_profit_usdc,
        f"{d.raw_edge_usdc:.2f} -> {d.net_profit_usdc:.2f} USDC",
    )
    losing = decompose(
        BacktestEngine(config).run(rows(40, label=0), ConstantModel(0.95)), config.costs
    )
    check(
        "risk limits that block losers show as a saving",
        losing.risk_limit_usdc < 0,
        f"{losing.risk_limit_usdc:.2f} USDC (a negative cost is a benefit)",
    )

    console.print("\n[bold]12. Latency is charged against book freshness[/]")
    latent = Config(costs={"assumed_latency_ms": 500})
    aged = Quote(
        best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
        age_ms=1_800,
    )
    stale = DecisionEngine(latent).decide(
        model_prob_up=0.9, quote=aged, seconds_into_window=120, seconds_to_settlement=180
    )
    check(
        "a book that expires in flight is refused",
        not stale.trade and stale.skip_reason is SkipReason.STALE_DATA,
        stale.detail,
    )

    console.print("\n[bold]13. Strict walk-forward never trades its training data[/]")
    wf = run_strict_walk_forward(
        config, rows(120), lambda _: MarketProbabilityModel(), model_name="market", n_folds=3
    )
    check("folds were produced", bool(wf.folds), f"{len(wf.folds)} fold(s)")
    check(
        "every fold trains strictly before it tests",
        all(f.train_end_ms < f.test_start_ms for f in wf.folds),
    )
    check(
        "markets adjacent to the test period are purged",
        any(f.purged_markets > 0 for f in wf.folds),
        f"max purge {max((f.purged_markets for f in wf.folds), default=0)} market(s)",
    )
    repeat = run_strict_walk_forward(
        config, rows(120), lambda _: ConstantModel(0.85), n_folds=3
    )
    again = run_strict_walk_forward(
        config, rows(120), lambda _: ConstantModel(0.85), n_folds=3
    )
    check("strict walk-forward is deterministic", repeat.as_dict() == again.as_dict())

    console.print("\n[bold]14. A better Brier score is not a deployment reason[/]")
    doomed = BacktestEngine(config).run(rows(40, label=0), ConstantModel(0.95))
    verdict = evaluate_backtest(config, doomed)
    ev_check = next(c for c in verdict.checks if c.name == "positive_ev_after_costs")
    check("the gate demands positive EV after costs", not ev_check.passed, ev_check.detail)
    check(
        "the gate verifies its own attribution",
        any(c.name == "decomposition_reconciles" for c in verdict.checks),
    )

    console.print()
    if FAILURES:
        console.print(f"[bold red]Module 8 validation FAILED ({len(FAILURES)}):[/]")
        for failure in FAILURES:
            console.print(f"  - {failure}")
        return 1
    console.print("[bold green]Module 8 validation OK.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
