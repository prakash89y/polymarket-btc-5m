"""Module 8.5 validation: the ceilings hold, and the reports tell the truth.

Each check corresponds to a way the system could resume trading on pathological
disagreement. The reference case throughout is the trade the statistical review
found in live data: model 0.2391, book 0.9750, bought DOWN at 0.035, EV reported
as +2,074%, lost the full stake.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console

from pmbtc.backtest import disagreement_distribution, scan_horizons
from pmbtc.backtest.adapters import ConstantModel, MarketProbabilityModel
from pmbtc.config import Config
from pmbtc.constants import SkipReason
from pmbtc.logging_setup import configure_logging
from pmbtc.trading import DecisionEngine, Quote
from pmbtc.trading.validation import (
    DisagreementMonitor,
    blend_toward_market,
    disagreement_logits,
)

console = Console()
FAILURES: list[str] = []
SETTLE0 = 1_785_600_000_000
HORIZONS = (300, 240, 180, 120, 60, 30)


def check(name: str, passed: bool, detail: str = "") -> bool:
    console.print(
        f"  {'[green]PASS[/]' if passed else '[red]FAIL[/]'} {name}"
        + (f" — {detail}" if detail else "")
    )
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def rows(n: int = 120, *, bid: float = 0.49, ask: float = 0.51) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(n):
        settlement = SETTLE0 + i * 300_000
        for h in HORIZONS:
            out.append({
                "condition_id": f"m{i:04d}",
                "slug": f"btc-updown-5m-{settlement // 1000}",
                "horizon_seconds": h,
                "settlement_time_ms": settlement,
                "label": i % 2,
                "official_outcome": "up" if i % 2 else "down",
                "f_ob_best_bid": bid,
                "f_ob_best_ask": ask,
                "f_ob_mid": (bid + ask) / 2.0,
                "f_lq_depth_bid_usdc": 500.0,
                "f_lq_depth_ask_usdc": 500.0,
                "lat_ob_best_bid": 200,
                "lat_ob_best_ask": 200,
            })
    return out


def main() -> int:
    config = Config()
    configure_logging(config)

    console.print("\n[bold]1. Disagreement is measured on the scale that matters[/]")
    ordinary = disagreement_logits(0.75, 0.50)
    extraordinary = disagreement_logits(0.72, 0.97)
    check(
        "equal probability gaps are not equal claims",
        extraordinary > 2 * ordinary,
        f"0.25 from a 0.50 book = {ordinary:.2f} logits; "
        f"0.25 from a 0.97 book = {extraordinary:.2f} logits",
    )
    check(
        "the reviewed pathological trade scores enormous",
        disagreement_logits(0.2391, 0.9750) > 4.0,
        f"{disagreement_logits(0.2391, 0.9750):.2f} logits, ceiling "
        f"{config.prediction.max_disagreement_logits:.2f}",
    )

    console.print("\n[bold]2. The ceilings refuse what the floors let through[/]")
    book = Quote(
        best_bid=0.96, best_ask=0.97, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
        age_ms=200,
    )
    verdict = DecisionEngine(config).decide(
        model_prob_up=0.2391, quote=book, seconds_into_window=270, seconds_to_settlement=30
    )
    check(
        "the exact reviewed trade is blocked",
        not verdict.trade and verdict.skip_reason is SkipReason.EXCESSIVE_DISAGREEMENT,
        verdict.detail,
    )
    implausible = DecisionEngine(config).decide(
        model_prob_up=0.001,
        quote=Quote(best_bid=0.18, best_ask=0.19, bid_depth_usdc=500.0,
                    ask_depth_usdc=500.0, age_ms=200),
        seconds_into_window=270,
        seconds_to_settlement=30,
    )
    check(
        "an implausible probability is named as such",
        implausible.skip_reason is SkipReason.IMPLAUSIBLE_PROBABILITY,
        implausible.detail,
    )

    console.print("\n[bold]3. The ceilings do not close the system down[/]")
    fair = Quote(
        best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
        age_ms=200,
    )
    ok = DecisionEngine(config).decide(
        model_prob_up=0.72, quote=fair, seconds_into_window=120, seconds_to_settlement=180
    )
    check("an ordinary disagreement still trades", ok.trade, str(ok))

    console.print("\n[bold]4. Anomaly detection is robust to its own outliers[/]")
    monitor = DisagreementMonitor(min_observations=10)
    monitor.extend([0.2] * 20)
    check("a calm baseline flags a spike", (monitor.z_score(4.0) or 0) > 10)
    cold = DisagreementMonitor(min_observations=30)
    cold.extend([0.2] * 5)
    check(
        "a cold start has no opinion rather than a permissive one",
        cold.z_score(9.9) is None,
    )
    poisoned = DisagreementMonitor(min_observations=10)
    poisoned.extend([0.2] * 20 + [8.0, 9.0, 10.0])
    check(
        "median/MAD are not dragged by contamination",
        (poisoned.z_score(4.0) or 0) > 5,
        "a mean/stdev baseline would have absorbed the outliers",
    )

    console.print("\n[bold]5. Calibration-adjusted EV shrinks toward the book[/]")
    check(
        "zero trust reproduces the market exactly",
        blend_toward_market(0.99, 0.95, 0.0) == 0.95,
    )
    check(
        "shrinking a 0.99 claim against a 0.95 book stays above the book",
        blend_toward_market(0.99, 0.95, 0.5) > 0.95,
        "shrinking toward 0.5 instead would invent edge on the DOWN side",
    )
    muted = Config(prediction={"model_trust": 0.0})
    check(
        "zero trust abstains everywhere",
        not any(
            DecisionEngine(muted).decide(
                model_prob_up=p, quote=fair,
                seconds_into_window=120, seconds_to_settlement=180,
            ).trade
            for p in (0.6, 0.7, 0.8)
        ),
    )

    console.print("\n[bold]6. The disagreement report identifies a broken model[/]")
    broken = disagreement_distribution(
        config, rows(40), ConstantModel(0.999), horizon_seconds=30
    )
    check(
        "a model that fights the book everywhere is called pathological",
        not broken.healthy and broken.over_ceiling > 0.9,
        f"median {broken.median:.2f} logits, {broken.over_ceiling:.0%} over ceiling",
    )
    sane = disagreement_distribution(
        config, rows(40), MarketProbabilityModel(), horizon_seconds=30
    )
    check(
        "a model that agrees with the book is called healthy",
        sane.healthy and sane.over_ceiling == 0.0,
    )

    console.print("\n[bold]7. Horizon choice is out-of-sample and demands stability[/]")
    scan = scan_horizons(
        config, rows(120), lambda _: MarketProbabilityModel(),
        model_name="market", n_folds=3,
    )
    check("every candidate horizon was evaluated", len(scan.horizons) >= 5,
          f"{len(scan.horizons)} horizons")
    check(
        "a null strategy yields no evidence of edge",
        not scan.any_evidence_of_edge and scan.best is None,
    )
    from pmbtc.backtest.edgescan import HorizonResult

    lucky = HorizonResult(
        horizon_seconds=60, folds=4, trades=50, net_pnl_usdc=100.0,
        profitable_folds=1, model_brier=0.2, market_brier=0.3,
    )
    check(
        "profit concentrated in one fold is not stability",
        not lucky.stable,
        "1 of 4 folds profitable",
    )
    worse_than_book = HorizonResult(
        horizon_seconds=60, folds=4, trades=50, net_pnl_usdc=100.0,
        profitable_folds=3, model_brier=0.4, market_brier=0.3,
    )
    check(
        "profit without beating the book is not stability",
        not worse_than_book.stable,
        "unexplained profit is not evidence of edge",
    )

    console.print("\n[bold]8. Determinism[/]")
    data = rows(120)
    a = scan_horizons(config, data, lambda _: ConstantModel(0.62), n_folds=3)
    b = scan_horizons(config, data, lambda _: ConstantModel(0.62), n_folds=3)
    check("the edge scan is reproducible", a.as_dict() == b.as_dict())

    console.print()
    if FAILURES:
        console.print(f"[bold red]Module 8.5 validation FAILED ({len(FAILURES)}):[/]")
        for failure in FAILURES:
            console.print(f"  - {failure}")
        return 1
    console.print("[bold green]Module 8.5 validation OK.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
