"""Module 7 validation on fixture data.

The production readiness gate still blocks real training, so the framework is
exercised on a synthetic dataset with the same shape as the real one: many
snapshots per market, one shared label, chronological order.
"""

from __future__ import annotations

import sys

import numpy as np
from rich.console import Console

from pmbtc.config import load_config
from pmbtc.logging_setup import configure_logging
from pmbtc.models import (
    ModelTrainer,
    SearchMethod,
    SplitScheme,
    TimeSeriesSplitter,
    TrainingData,
    write_report,
)

console = Console()
FAILURES: list[str] = []
SETTLE0 = 1_785_600_000_000
HORIZONS = (300, 240, 180, 120, 60, 30, 15, 10, 5, 2, 1)


def check(name: str, passed: bool, detail: str = "") -> bool:
    console.print(f"  {'[green]PASS[/]' if passed else '[red]FAIL[/]'} {name}"
                  + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def fixture(n_markets: int, seed: int, signal: float) -> TrainingData:
    rng = np.random.default_rng(seed)
    xs, ys, ts, gs, ms = [], [], [], [], []
    for market in range(n_markets):
        latent = rng.normal()
        label = int(latent + rng.normal(scale=signal) > 0)
        settle = SETTLE0 + market * 300_000
        for horizon in HORIZONS:
            clarity = 1.0 - horizon / 300.0
            observed = latent * (0.3 + 0.7 * clarity) + rng.normal(scale=0.5)
            xs.append([observed, rng.normal(), rng.normal(), float(horizon)])
            ys.append(label)
            ts.append(settle)
            gs.append(f"m{market}")
            ms.append(float(np.clip(0.5 + 0.15 * observed, 0.02, 0.98)))
    return TrainingData(
        x=np.array(xs), y=np.array(ys), timestamps=np.array(ts, dtype=np.int64),
        groups=np.array(gs), market_prob=np.array(ms),
        feature_names=["signal", "noise_a", "noise_b", "horizon"],
        dataset_version="fixture-1", feature_schema_version="fs-1",
    )


def main() -> int:
    config = load_config()
    configure_logging(config)
    data = fixture(400, seed=3, signal=0.6)

    console.rule("[bold]1. Time-series validation")
    for scheme in (SplitScheme.EXPANDING, SplitScheme.ROLLING, SplitScheme.WALK_FORWARD):
        splitter = TimeSeriesSplitter(scheme=scheme, n_folds=4, min_train_markets=60)
        folds = list(splitter.split(data.timestamps, data.groups))
        ordered = all(
            max(data.timestamps[f.train]) <= min(data.timestamps[f.test]) for f in folds
        )
        disjoint = all(
            not (set(data.groups[f.train]) & set(data.groups[f.test])) for f in folds
        )
        check(f"{scheme.value}: train precedes test", ordered, f"{len(folds)} folds")
        check(f"{scheme.value}: markets never span a split", disjoint)
    check("no shuffle parameter exists",
          "shuffle" not in TimeSeriesSplitter.__dataclass_fields__)

    console.rule("[bold]2. Full training pipeline")
    trainer = ModelTrainer(config)
    result = trainer.train(
        data, model_type="gradient_boosting", search_method=SearchMethod.RANDOM,
        enforce_gates=False,
    )
    check("model trained", result.trained,
          result.card.model_id if result.card else result.blocked_reason)
    check("cross-validated", len(result.cv_metrics) > 0, f"{len(result.cv_metrics)} folds")
    if result.holdout:
        console.print(f"       holdout: {result.holdout}")
        check("calibration reported",
              all(k in result.holdout.as_dict() for k in ("brier", "log_loss", "ece", "mce")))
        console.print(result.holdout.curve.render() if result.holdout.curve else "")

    console.rule("[bold]3. Benchmark comparison")
    if result.baselines:
        console.print(str(result.baselines))
        check("all five baselines scored", len(result.baselines.scores) == 5)

    console.rule("[bold]4. Statistical testing and the promotion gate")
    if result.decision:
        for test in result.decision.tests:
            console.print(f"       {test}")
        for gate in result.decision.checks:
            console.print(f"       {gate}")
        console.print(f"       -> {result.decision.summary()}")
        check("gate reached a decision", True,
              "approved" if result.decision.approved else "blocked")
        check("gate has no override",
              "force" not in ModelTrainer.train.__code__.co_varnames)

    console.rule("[bold]5. Shadow validation")
    if result.shadow:
        console.print(f"       {result.shadow.summary()}")
        check("determinism verified", result.shadow.deterministic)
        check("latency measured", result.shadow.latency_p95_ms >= 0,
              f"p50={result.shadow.latency_p50_ms:.2f}ms p95={result.shadow.latency_p95_ms:.2f}ms")
        check("calibration stability measured", len(result.shadow.blocks) > 1,
              f"{len(result.shadow.blocks)} blocks, drift={result.shadow.calibration_drift:.4f}")

    console.rule("[bold]6. Explainability")
    if result.explanation:
        console.print(result.explanation.render(8))
        check("global importance produced", bool(result.explanation.importance))
        check("interactions computed", bool(result.explanation.interactions))
        top = result.explanation.top(1)
        check("dominant feature is the real signal", bool(top) and top[0][0] == "signal",
              f"top={top[0][0] if top else 'none'}")

    console.rule("[bold]7. Registry and reproducibility")
    if result.card:
        card = result.card
        for field_name in ("model_id", "dataset_version", "feature_schema_version",
                           "config_hash", "hyperparameter_hash", "trained_at_ms"):
            check(f"card.{field_name}", bool(getattr(card, field_name)),
                  str(getattr(card, field_name))[:48])
        reloaded = trainer.registry.load_card(card.model_id)
        check("card round-trips from disk",
              reloaded.reproducibility_key() == card.reproducibility_key())
        path = write_report(
            result,
            config.resolved_path(config.app.artifact_dir) / "reports" / f"{card.model_id}.json",
        )
        check("evaluation report written", path.exists(), str(path))

    console.rule("[bold]8. A useless model is refused")
    noise = fixture(250, seed=11, signal=50.0)
    noise_result = ModelTrainer(config).train(
        noise, search_method=SearchMethod.RANDOM, enforce_gates=False
    )
    check("noise model not promoted",
          noise_result.decision is not None and not noise_result.decision.approved,
          noise_result.decision.summary() if noise_result.decision else "")

    console.rule("[bold]Result")
    if FAILURES:
        console.print(f"[bold red]{len(FAILURES)} check(s) failed:[/]")
        for failure in FAILURES:
            console.print(f"  - {failure}")
        return 1
    console.print("[bold green]All Module 7 validation checks passed.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
