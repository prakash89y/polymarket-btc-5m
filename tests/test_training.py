"""Module 7: validation discipline, calibration, statistics, and the gate.

The tests that matter here are the ones asserting the framework *refuses* to
produce a flattering result: no shuffled splits, no market spanning train and
test, no tuning on the holdout, and no promotion on a difference that is not
statistically real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmbtc.config import load_config
from pmbtc.constants import Outcome, SettlementSource
from pmbtc.dataset.schema import MarketRecord
from pmbtc.exceptions import InsufficientDataError, ModelError
from pmbtc.models import (
    Calibrator,
    ModelRegistry,
    ModelTrainer,
    SearchMethod,
    SplitScheme,
    TimeSeriesSplitter,
    TrainingData,
    assert_no_leakage_between,
    binomial_vs_breakeven,
    calibration_curve,
    evaluate,
    make_model_id,
    mcnemar,
    paired_bootstrap,
    run_shadow_validation,
    stable_hash,
)
from pmbtc.models.registry import ModelCard
from pmbtc.models.search import HyperparameterSearch, grid_candidates, random_candidates

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
SETTLE0 = 1_785_600_000_000
HORIZONS = (300, 240, 180, 120, 60, 30, 15, 10, 5, 2, 1)


@pytest.fixture
def config(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})


def synthetic_dataset(
    n_markets: int = 400, seed: int = 0, signal: float = 0.9
) -> TrainingData:
    """A fixture dataset with a real but modest signal.

    Eleven snapshots per market share one label — the same structure as the real
    dataset, which is what makes shuffled splits so dangerous here.
    """
    rng = np.random.default_rng(seed)
    rows_x, rows_y, rows_ts, rows_g, rows_m = [], [], [], [], []
    for market in range(n_markets):
        latent = rng.normal()
        label = int(latent + rng.normal(scale=signal) > 0)
        settle = SETTLE0 + market * 300_000
        for horizon in HORIZONS:
            # Signal strengthens as settlement approaches, as it does in reality.
            clarity = 1.0 - horizon / 300.0
            observed = latent * (0.3 + 0.7 * clarity) + rng.normal(scale=0.5)
            rows_x.append([observed, rng.normal(), rng.normal(), float(horizon)])
            rows_y.append(label)
            rows_ts.append(settle)
            rows_g.append(f"m{market}")
            rows_m.append(float(np.clip(0.5 + 0.15 * observed, 0.02, 0.98)))
    return TrainingData(
        x=np.array(rows_x, dtype=float),
        y=np.array(rows_y, dtype=int),
        timestamps=np.array(rows_ts, dtype=np.int64),
        groups=np.array(rows_g),
        market_prob=np.array(rows_m, dtype=float),
        feature_names=["signal", "noise_a", "noise_b", "horizon"],
        dataset_version="synthetic-1",
        feature_schema_version="fs-1",
    )


def synthetic_markets(n: int = 400) -> list[MarketRecord]:
    markets = []
    for i in range(n):
        settle = SETTLE0 + i * 300_000
        markets.append(
            MarketRecord(
                condition_id=f"m{i}", slug=f"s{i}", question="q",
                discovery_time_ms=settle - 86_400_000,
                open_time_ms=settle - 300_000,
                lock_time_ms=settle - 20_000,
                settlement_time_ms=settle,
                settlement_provider=SettlementSource.CHAINLINK,
                settlement_spec_hash="hash1",
                official_outcome=Outcome.UP if i % 2 else Outcome.DOWN,
                resolved_at_ms=settle + 30_000,
                resolution_source="polymarket_official",
            )
        )
    return markets


# =========================================================================== #
# Time-series validation
# =========================================================================== #
class TestValidation:
    def test_no_shuffle_option_exists(self) -> None:
        # Structural: there is no parameter that could randomise a split.
        assert not hasattr(TimeSeriesSplitter, "shuffle")
        assert "shuffle" not in TimeSeriesSplitter.__dataclass_fields__

    def test_train_always_precedes_test(self) -> None:
        data = synthetic_dataset(200)
        splitter = TimeSeriesSplitter(n_folds=4, min_train_markets=40)
        folds = list(splitter.split(data.timestamps, data.groups))
        assert folds
        for fold in folds:
            assert max(data.timestamps[fold.train]) <= min(data.timestamps[fold.test])

    def test_market_never_spans_a_split(self) -> None:
        # Eleven snapshots share a label; splitting them would leak it.
        data = synthetic_dataset(200)
        for fold in TimeSeriesSplitter(n_folds=4, min_train_markets=40).split(
            data.timestamps, data.groups
        ):
            assert not (set(data.groups[fold.train]) & set(data.groups[fold.test]))

    def test_expanding_window_grows(self) -> None:
        data = synthetic_dataset(300)
        folds = list(
            TimeSeriesSplitter(
                scheme=SplitScheme.EXPANDING, n_folds=4, min_train_markets=40
            ).split(data.timestamps, data.groups)
        )
        sizes = [f.train_size for f in folds]
        assert sizes == sorted(sizes)

    def test_rolling_window_is_bounded(self) -> None:
        data = synthetic_dataset(300)
        folds = list(
            TimeSeriesSplitter(
                scheme=SplitScheme.ROLLING, n_folds=4, train_size=50,
                min_train_markets=40,
            ).split(data.timestamps, data.groups)
        )
        assert all(f.train_size <= 50 * len(HORIZONS) for f in folds)

    def test_embargo_purges_adjacent_markets(self) -> None:
        data = synthetic_dataset(200)
        folds = list(
            TimeSeriesSplitter(n_folds=3, min_train_markets=40, embargo_markets=5).split(
                data.timestamps, data.groups
            )
        )
        assert all(f.purged == 5 for f in folds)

    def test_insufficient_data_raises(self) -> None:
        data = synthetic_dataset(5)
        with pytest.raises(InsufficientDataError):
            list(TimeSeriesSplitter(n_folds=5, min_train_markets=50).split(
                data.timestamps, data.groups
            ))

    def test_holdout_is_the_chronological_tail(self) -> None:
        data = synthetic_dataset(200)
        develop, holdout = TimeSeriesSplitter().final_holdout(
            data.timestamps, data.groups, 0.2
        )
        assert max(data.timestamps[develop]) <= min(data.timestamps[holdout])
        assert not (set(data.groups[develop]) & set(data.groups[holdout]))

    def test_leakage_assertion_catches_a_bad_split(self) -> None:
        data = synthetic_dataset(50)
        everything = np.arange(len(data.y))
        with pytest.raises(InsufficientDataError):
            assert_no_leakage_between(everything, everything, data.timestamps, data.groups)


# =========================================================================== #
# Calibration
# =========================================================================== #
class TestCalibration:
    def test_perfect_calibration_has_zero_ece(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.uniform(size=20_000) < p).astype(int)
        assert evaluate(y, p).ece < 0.02

    def test_overconfident_model_is_penalised(self) -> None:
        rng = np.random.default_rng(1)
        truth = rng.uniform(0.4, 0.6, 5_000)
        y = (rng.uniform(size=5_000) < truth).astype(int)
        overconfident = np.where(truth > 0.5, 0.95, 0.05)
        assert evaluate(y, overconfident).ece > evaluate(y, truth).ece

    def test_metrics_are_reported(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.uniform(0.1, 0.9, 500)
        y = (rng.uniform(size=500) < p).astype(int)
        metrics = evaluate(y, p)
        for key in ("accuracy", "brier", "log_loss", "ece", "mce", "brier_skill"):
            assert key in metrics.as_dict()

    def test_calibration_curve_bins(self) -> None:
        curve = calibration_curve(
            np.array([0, 1, 1, 1]), np.array([0.1, 0.9, 0.85, 0.95]), bins=10
        )
        assert curve.samples == 4
        assert any(b.count for b in curve.bins)
        assert "predicted -> observed" in curve.render()

    def test_isotonic_calibrator_improves_a_skewed_model(self) -> None:
        rng = np.random.default_rng(3)
        truth = rng.uniform(0.2, 0.8, 2_000)
        y = (rng.uniform(size=2_000) < truth).astype(int)
        skewed = np.clip(truth * 0.5 + 0.25, 0.01, 0.99)  # compressed toward 0.5
        calibrator = Calibrator("isotonic").fit(skewed[:1_000], y[:1_000])
        corrected = calibrator.transform(skewed[1_000:])
        assert evaluate(y[1_000:], corrected).ece <= evaluate(y[1_000:], skewed[1_000:]).ece

    def test_none_calibrator_is_identity(self) -> None:
        p = np.array([0.2, 0.5, 0.8])
        assert np.allclose(Calibrator("none").fit(p, np.array([0, 1, 1])).transform(p), p)


# =========================================================================== #
# Statistics
# =========================================================================== #
class TestStatistics:
    def test_bootstrap_detects_a_real_difference(self) -> None:
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 2_000)
        good = np.where(y == 1, 0.8, 0.2)
        bad = np.full(2_000, 0.5)
        result = paired_bootstrap(y, good, bad, iterations=500)
        assert result.significant(0.05)
        assert result.effect > 0

    def test_bootstrap_rejects_noise(self) -> None:
        # The whole point: a trivial difference must not clear the bar.
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 400)
        a = rng.uniform(0.45, 0.55, 400)
        b = a + rng.normal(scale=1e-4, size=400)
        assert not paired_bootstrap(y, a, b, iterations=400).significant(0.05)

    def test_mcnemar_uses_discordant_pairs_only(self) -> None:
        y = np.array([1, 1, 1, 1, 0, 0])
        a = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
        b = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = mcnemar(y, a, b)
        assert result.n == 4
        assert result.p_value < 0.10

    def test_mcnemar_with_no_disagreement(self) -> None:
        y = np.array([1, 0])
        p = np.array([0.9, 0.1])
        assert mcnemar(y, p, p).p_value == 1.0

    def test_binomial_vs_breakeven(self) -> None:
        assert binomial_vs_breakeven(70, 100).significant(0.05)
        assert not binomial_vs_breakeven(52, 100).significant(0.05)


# =========================================================================== #
# Search
# =========================================================================== #
class TestSearch:
    def test_grid_is_exhaustive_and_ordered(self) -> None:
        space = {"a": [1, 2], "b": [3, 4]}
        first = list(grid_candidates(space))
        assert len(first) == 4
        assert first == list(grid_candidates(space))

    def test_random_search_is_bounded_and_unique(self) -> None:
        space = {"a": [1, 2, 3], "b": [4, 5]}
        drawn = list(random_candidates(space, 4, seed=0))
        assert len(drawn) <= 4
        assert len({stable_hash(d) for d in drawn}) == len(drawn)

    def test_search_never_sees_the_holdout(self) -> None:
        # The search is handed development rows only; assert the fit callback
        # is never invoked with more rows than were supplied.
        data = synthetic_dataset(150)
        develop, _ = TimeSeriesSplitter().final_holdout(data.timestamps, data.groups, 0.2)
        seen: list[int] = []

        def fit_predict(params, xt, yt, xv):  # type: ignore[no-untyped-def]
            seen.append(len(xt) + len(xv))
            return np.full(len(xv), 0.5)

        search = HyperparameterSearch(
            space={"k": [1, 2]},
            splitter=TimeSeriesSplitter(n_folds=3, min_train_markets=30),
        )
        search.run(
            fit_predict, data.x[develop], data.y[develop],
            data.timestamps[develop], data.groups[develop],
        )
        assert seen and max(seen) <= len(develop)

    def test_best_params_are_reproducible(self) -> None:
        data = synthetic_dataset(150)

        def fit_predict(params, xt, yt, xv):  # type: ignore[no-untyped-def]
            return np.full(len(xv), 0.5 + 0.01 * params["k"])

        def run() -> dict:
            return HyperparameterSearch(
                space={"k": [1, 2, 3]},
                splitter=TimeSeriesSplitter(n_folds=3, min_train_markets=30),
            ).run(fit_predict, data.x, data.y, data.timestamps, data.groups).best_params

        assert run() == run()


# =========================================================================== #
# Registry
# =========================================================================== #
class TestRegistry:
    def _card(self, model_id: str = "m1") -> ModelCard:
        return ModelCard(
            model_id=model_id, model_type="gradient_boosting",
            dataset_version="d1", feature_schema_version="f1",
            config_hash="c1", hyperparameter_hash="h1", trained_at_ms=SETTLE0,
        )

    def test_model_id_encodes_inputs(self) -> None:
        a = make_model_id("gb", "d1", "f1", "c1", "h1", SETTLE0)
        b = make_model_id("gb", "d1", "f1", "c1", "h2", SETTLE0)
        assert a != b
        assert a.startswith("gb-")

    def test_save_and_load(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "models")
        registry.save(self._card(), estimator={"fake": True})
        card = registry.load_card("m1")
        assert card.dataset_version == "d1"
        estimator, calibrator = registry.load_estimator("m1")
        assert estimator == {"fake": True}
        assert calibrator is None

    def test_artifacts_are_immutable(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "models")
        registry.save(self._card(), estimator={})
        with pytest.raises(ModelError, match="immutable"):
            registry.save(self._card(), estimator={})

    def test_promotion_moves_a_pointer(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "models")
        registry.save(self._card("m1"), estimator={})
        registry.save(self._card("m2"), estimator={})
        registry.promote("m1", ["ok"])
        assert registry.production_model_id() == "m1"
        registry.promote("m2", ["better"])
        assert registry.production_model_id() == "m2"
        assert registry.load_card("m1").promoted is True  # artifact unchanged

    def test_reproducibility_key(self) -> None:
        assert self._card().reproducibility_key() == self._card("other").reproducibility_key()


# =========================================================================== #
# Shadow validation
# =========================================================================== #
class TestShadow:
    def test_deterministic_model_passes(self) -> None:
        data = synthetic_dataset(60)

        def predict(x: np.ndarray) -> np.ndarray:
            return np.clip(0.5 + 0.1 * x[:, 0], 0.01, 0.99)

        result = run_shadow_validation(
            predict, data.x, data.y, data.timestamps, max_calibration_drift=1.0
        )
        assert result.deterministic
        assert result.passed
        assert result.latency_p95_ms >= 0

    def test_nondeterministic_model_fails(self) -> None:
        data = synthetic_dataset(30)
        rng = np.random.default_rng(0)

        def predict(x: np.ndarray) -> np.ndarray:
            return np.clip(0.5 + rng.normal(scale=0.1, size=len(x)), 0.01, 0.99)

        result = run_shadow_validation(predict, data.x, data.y, data.timestamps)
        assert not result.deterministic
        assert not result.passed

    def test_artifact_mismatch_is_caught(self) -> None:
        data = synthetic_dataset(30)

        def predict(x: np.ndarray) -> np.ndarray:
            return np.full(len(x), 0.6)

        def reloaded(x: np.ndarray) -> np.ndarray:
            return np.full(len(x), 0.61)

        result = run_shadow_validation(
            predict, data.x, data.y, data.timestamps, reloaded_predict=reloaded
        )
        assert not result.reproducible_from_artifact
        assert not result.passed

    def test_latency_budget_is_enforced(self) -> None:
        import time as _time

        data = synthetic_dataset(12)

        def slow(x: np.ndarray) -> np.ndarray:
            _time.sleep(0.003)
            return np.full(len(x), 0.5)

        result = run_shadow_validation(
            slow, data.x, data.y, data.timestamps, max_latency_ms=0.5
        )
        assert any("latency" in f for f in result.failures)


# =========================================================================== #
# End-to-end training with gates
# =========================================================================== #
class TestTrainerGates:
    def test_readiness_gate_blocks_training(self, config) -> None:  # type: ignore[no-untyped-def]
        trainer = ModelTrainer(config)
        result = trainer.train(
            synthetic_dataset(60), markets=synthetic_markets(60), snapshots=[],
            enforce_gates=True,
        )
        assert not result.trained
        assert "NOT ready" in result.blocked_reason

    def test_no_override_parameter_exists(self) -> None:
        # There must be no way to force promotion.
        import inspect

        signature = inspect.signature(ModelTrainer.train)
        assert "force" not in signature.parameters
        assert "override" not in signature.parameters
        from pmbtc.models.promotion import evaluate_promotion

        assert "force" not in inspect.signature(evaluate_promotion).parameters

    def test_full_pipeline_on_fixture_data(self, config) -> None:  # type: ignore[no-untyped-def]
        # Gates disabled only because the *production* dataset is still small;
        # the framework itself is exercised end to end on fixtures.
        trainer = ModelTrainer(config)
        result = trainer.train(
            synthetic_dataset(300, signal=0.6),
            model_type="gradient_boosting",
            search_method=SearchMethod.RANDOM,
            enforce_gates=False,
        )
        assert result.trained
        assert result.holdout is not None
        assert result.baselines is not None and len(result.baselines.scores) == 5
        assert result.shadow is not None
        assert result.decision is not None
        assert result.explanation is not None
        assert result.card is not None
        assert result.card.metrics["holdout"]["samples"] > 0

    def test_promoted_model_beats_every_baseline(self, config) -> None:  # type: ignore[no-untyped-def]
        trainer = ModelTrainer(config)
        result = trainer.train(
            synthetic_dataset(300, signal=0.6), enforce_gates=False,
            search_method=SearchMethod.RANDOM,
        )
        assert result.decision is not None
        if result.decision.approved:
            candidate = result.holdout.brier  # type: ignore[union-attr]
            assert all(candidate <= s.brier for s in result.baselines.scores)  # type: ignore[union-attr]

    def test_useless_model_is_not_promoted(self, config) -> None:  # type: ignore[no-untyped-def]
        # Pure noise: no signal at all, so the gate must refuse.
        trainer = ModelTrainer(config)
        data = synthetic_dataset(250, seed=9, signal=50.0)
        result = trainer.train(data, enforce_gates=False, search_method=SearchMethod.RANDOM)
        assert result.decision is not None
        assert not result.decision.approved

    def test_model_card_is_complete(self, config) -> None:  # type: ignore[no-untyped-def]
        trainer = ModelTrainer(config)
        result = trainer.train(
            synthetic_dataset(250, signal=0.6), enforce_gates=False,
            search_method=SearchMethod.RANDOM,
        )
        card = result.card
        assert card is not None
        for field_name in (
            "model_id", "dataset_version", "feature_schema_version",
            "config_hash", "hyperparameter_hash", "trained_at_ms",
        ):
            assert getattr(card, field_name)
        assert card.metrics and card.validation and card.shadow
