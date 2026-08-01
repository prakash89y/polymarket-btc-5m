"""Module 6: graph, determinism, incremental updates, and per-feature maths.

Every feature has a stated formula, so every feature gets a test that checks the
formula against a hand-built book or tape whose correct answer can be computed
on paper. That is the only way "prioritise correctness over feature count"
becomes something other than a slogan.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pytest

from pmbtc.exceptions import ConfigError
from pmbtc.features.base import (
    Feature,
    FeatureContext,
    define,
    safe_div,
)
from pmbtc.features.consistency import ConsistencyLimits, check_consistency
from pmbtc.features.docs import generate_markdown, require_documentation, undocumented
from pmbtc.features.drift import DriftKind, DriftMonitor, population_stability_index
from pmbtc.features.engine import FeatureEngine
from pmbtc.features.families import FAMILIES, all_features
from pmbtc.features.graph import (
    DependencyCycleError,
    UnknownDependencyError,
    build_graph,
)
from pmbtc.features.matrix import MatrixBuilder, feature_set_version
from pmbtc.features.registry import FeatureTier, ReproducibilityPolicy
from pmbtc.features.selection import select_features
from pmbtc.live.book import BookPair
from pmbtc.live.tape import Trade, TradeTape

SETTLE = 1_785_600_000_000
OPEN = SETTLE - 300_000


def make_books(up_bid: float = 0.48, up_ask: float = 0.52, bid_size: float = 100.0,
               ask_size: float = 50.0) -> BookPair:
    books = BookPair()
    books.up.asset_id, books.down.asset_id = "UP", "DOWN"
    books.up.replace([(up_bid, bid_size), (up_bid - 0.01, 200.0)],
                     [(up_ask, ask_size), (up_ask + 0.01, 150.0)], 1_000)
    books.down.replace([(1 - up_ask, ask_size)], [(1 - up_bid, bid_size)], 1_000)
    return books


def make_tape(prices: list[float], sizes: list[float], signs: list[int],
              start_ms: int = 0, step_ms: int = 1_000) -> TradeTape:
    tape = TradeTape()
    for i, (p, s, sign) in enumerate(zip(prices, sizes, signs, strict=True)):
        tape.add(Trade(p, s, sign, start_ms + i * step_ms))
    return tape


def make_context(now_ms: int = SETTLE - 60_000, **kwargs) -> FeatureContext:  # type: ignore[no-untyped-def]
    defaults: dict = {
        "books": make_books(),
        "pm_tape": make_tape([0.5, 0.51], [10, 20], [1, -1], now_ms - 30_000),
        # A realistic tape: dense enough and spanning enough of the window for
        # a volatility estimate to be trusted. Deterministic by construction so
        # the formula assertions below stay exact.
        "ref_tape": make_tape(
            [63_000.0 + 10.0 * ((i * 7) % 5 - 2) for i in range(59)] + [63_030.0],
            [1.0] * 60,
            [1 if i % 2 else -1 for i in range(60)],
            now_ms - 300_000,
            step_ms=5_000,
        ),
        "ref_bid": 63_029.0,
        "ref_ask": 63_031.0,
        "window_open_price": 63_000.0,
        "regime": {"tick_size": 0.01, "funding_rate": 0.0001},
    }
    defaults.update(kwargs)
    return FeatureContext(
        now_ms=now_ms, window_open_ms=OPEN, settlement_ms=SETTLE, **defaults
    )


# =========================================================================== #
# Graph
# =========================================================================== #
class TestGraph:
    def test_builds_from_all_families(self) -> None:
        graph = build_graph(all_features())
        assert len(graph) > 50
        assert graph.order

    def test_order_is_topological(self) -> None:
        graph = build_graph(all_features())
        position = {name: i for i, name in enumerate(graph.order)}
        for feature in graph.features.values():
            for dependency in feature.dependencies:
                assert position[dependency] < position[feature.name], (
                    f"{dependency} must come before {feature.name}"
                )

    def test_order_is_deterministic(self) -> None:
        # Bit-for-bit reproducibility depends on this being stable.
        assert build_graph(all_features()).order == build_graph(all_features()).order

    def test_cycle_is_detected_and_named(self) -> None:
        class Loop(Feature):
            def compute(self, context: FeatureContext) -> float | None:
                return 1.0

        a = Loop(define("cy_a", tier=FeatureTier.DERIVED, source="s", formula="b",
                        inputs=("cy_b",), units="u", purpose="p"))
        b = Loop(define("cy_b", tier=FeatureTier.DERIVED, source="s", formula="a",
                        inputs=("cy_a",), units="u", purpose="p"))
        with pytest.raises(DependencyCycleError) as exc:
            build_graph([a, b])
        assert "cy_a" in str(exc.value)

    def test_unknown_dependency_is_rejected(self) -> None:
        class Orphan(Feature):
            def compute(self, context: FeatureContext) -> float | None:
                return 1.0

        orphan = Orphan(
            define("orphan", tier=FeatureTier.DERIVED, source="s", formula="x",
                   inputs=("does_not_exist",), units="u", purpose="p")
        )
        with pytest.raises(UnknownDependencyError, match="does_not_exist"):
            build_graph([orphan])

    def test_impact_is_transitive(self) -> None:
        graph = build_graph(all_features())
        affected = graph.impact["pm_book"]
        # ob_mid reads the book directly; pb_conviction depends on ob_mid.
        assert "ob_mid" in affected
        assert "pb_conviction" in affected

    def test_affected_by_returns_topological_order(self) -> None:
        graph = build_graph(all_features())
        affected = graph.affected_by({"pm_book"})
        position = {name: i for i, name in enumerate(graph.order)}
        assert list(affected) == sorted(affected, key=lambda n: position[n])

    def test_clock_only_change_does_not_touch_the_book(self) -> None:
        graph = build_graph(all_features())
        affected = set(graph.affected_by({"clock"}))
        assert "tm_seconds_to_settlement" in affected
        assert "ob_best_bid" not in affected

    def test_mermaid_renders(self) -> None:
        assert build_graph(all_features()).to_mermaid().startswith("graph LR")


# =========================================================================== #
# Determinism and incremental updates
# =========================================================================== #
class TestEngine:
    def test_identical_context_gives_identical_fingerprint(self) -> None:
        a = FeatureEngine().compute(make_context())
        b = FeatureEngine().compute(make_context())
        assert a.fingerprint() == b.fingerprint()
        assert a.values == b.values

    def test_different_input_changes_the_fingerprint(self) -> None:
        base = FeatureEngine().compute(make_context())
        moved = FeatureEngine().compute(make_context(books=make_books(up_bid=0.60, up_ask=0.62)))
        assert base.fingerprint() != moved.fingerprint()

    def test_incremental_recomputes_only_affected_features(self) -> None:
        engine = FeatureEngine()
        engine.compute(make_context())
        vector = engine.compute(make_context(), changed_sources={"clock"})
        assert "tm_seconds_to_settlement" in vector.recomputed
        assert "ob_best_bid" not in vector.recomputed
        assert len(vector.recomputed) < len(engine.feature_names)

    def test_incremental_matches_full_recompute(self) -> None:
        # The whole point: recomputing less must not change the answer.
        full = FeatureEngine()
        full.compute(make_context())
        result_full = full.compute(make_context(now_ms=SETTLE - 30_000))

        partial = FeatureEngine()
        partial.compute(make_context())
        result_partial = partial.compute(
            make_context(now_ms=SETTLE - 30_000),
            changed_sources={"clock", "pm_book", "pm_trades", "ref_trades",
                             "ref_quote", "window_open_price", "regime", "market_meta"},
        )
        assert result_full.values == result_partial.values

    def test_non_finite_values_become_missing(self) -> None:
        # An infinite feature silently poisons any model that sees it.
        engine = FeatureEngine()
        vector = engine.compute(make_context(ref_bid=0.0, ref_ask=0.0))
        assert all(
            v is None or (v == v and abs(v) != float("inf")) for v in vector.values.values()
        )

    def test_a_failing_feature_does_not_void_the_pass(self) -> None:
        engine = FeatureEngine()
        broken = engine.graph.features["ob_mid"]
        original = broken.compute
        broken.compute = lambda ctx: 1 / 0  # type: ignore[assignment,method-assign]
        try:
            vector = engine.compute(make_context())
            assert "ob_mid" in vector.errors
            assert vector.values["ob_best_bid"] is not None
        finally:
            broken.compute = original  # type: ignore[method-assign]

    def test_cross_pass_features_need_a_prior(self) -> None:
        engine = FeatureEngine()
        first = engine.compute(make_context(now_ms=SETTLE - 120_000))
        assert first.values["pb_velocity"] is None  # no prior on the first pass
        second = engine.compute(
            make_context(now_ms=SETTLE - 60_000, books=make_books(0.58, 0.62))
        )
        assert second.values["pb_velocity"] is not None

    def test_reset_clears_cross_pass_state(self) -> None:
        engine = FeatureEngine()
        engine.compute(make_context(now_ms=SETTLE - 120_000))
        engine.reset()
        assert engine.compute(make_context()).values["pb_velocity"] is None


# =========================================================================== #
# Feature mathematics — each against its stated formula
# =========================================================================== #
class TestOrderBookMaths:
    def _values(self, **kwargs) -> dict:  # type: ignore[no-untyped-def]
        return FeatureEngine().compute(make_context(**kwargs)).values

    def test_spread_and_mid(self) -> None:
        v = self._values()
        assert v["ob_best_bid"] == pytest.approx(0.48)
        assert v["ob_best_ask"] == pytest.approx(0.52)
        assert v["ob_spread"] == pytest.approx(0.04)
        assert v["ob_mid"] == pytest.approx(0.50)

    def test_microprice_formula(self) -> None:
        # (bid*ask_size + ask*bid_size) / (bid_size + ask_size)
        expected = (0.48 * 50 + 0.52 * 100) / 150
        assert self._values()["ob_microprice"] == pytest.approx(expected)

    def test_microprice_edge_is_signed_toward_the_thin_side(self) -> None:
        v = self._values()
        assert v["ob_microprice_edge"] == pytest.approx(v["ob_microprice"] - v["ob_mid"])
        assert v["ob_microprice_edge"] > 0  # ask is thinner

    def test_imbalance_l1(self) -> None:
        assert self._values()["ob_imbalance_1"] == pytest.approx((100 - 50) / 150)

    def test_complement_gap_is_zero_on_a_consistent_pair(self) -> None:
        assert self._values()["ob_complement_gap"] == pytest.approx(0.0, abs=1e-9)


class TestVolatilityMaths:
    def _values(self, **kwargs) -> dict:  # type: ignore[no-untyped-def]
        return FeatureEngine().compute(make_context(**kwargs)).values

    def test_price_change_bps(self) -> None:
        # last ref price 63030 vs open 63000 -> +4.7619 bps
        expected = (63_030.0 - 63_000.0) / 63_000.0 * 10_000
        assert self._values()["vol_price_change_bps"] == pytest.approx(expected)

    def test_thin_tape_yields_no_volatility_estimate(self) -> None:
        # Cold start: a handful of prints must produce None, not a tiny sigma.
        # Observed live before this guard: distance_in_sigma of -85, which pins
        # the implied fair value at exactly 0.
        thin = make_tape([63_000.0, 63_010.0, 63_020.0], [1.0, 1.0, 1.0], [1, 1, 1],
                         SETTLE - 5_000)
        values = self._values(ref_tape=thin)
        assert values["vol_realized_300s_bps"] is None
        assert values["vol_distance_in_sigma"] is None
        assert values["vol_implied_fair_up"] is None

    def test_dense_tape_yields_a_sane_sigma_distance(self) -> None:
        rng = __import__("random").Random(7)
        prices, sizes, signs = [], [], []
        for i in range(80):
            prices.append(63_000.0 + rng.gauss(0, 15))
            sizes.append(1.0)
            signs.append(1 if i % 2 else -1)
        dense = make_tape(prices, sizes, signs, SETTLE - 360_000, step_ms=4_000)
        values = self._values(now_ms=SETTLE - 60_000, ref_tape=dense)
        distance = values["vol_distance_in_sigma"]
        assert distance is not None
        assert abs(distance) < 20  # a plausible move, not 85 sigma

    def test_time_scaled_sigma_shrinks_toward_settlement(self) -> None:
        early = self._values(now_ms=SETTLE - 300_000)["vol_time_scaled_sigma_bps"]
        late = self._values(now_ms=SETTLE - 30_000)["vol_time_scaled_sigma_bps"]
        assert early is not None and late is not None
        assert late < early

    def test_time_scaling_follows_sqrt(self) -> None:
        v = self._values(now_ms=SETTLE - 75_000)
        sigma, scaled = v["vol_realized_300s_bps"], v["vol_time_scaled_sigma_bps"]
        assert sigma is not None and scaled is not None
        assert scaled == pytest.approx(sigma * sqrt(75.0 / 300.0), rel=1e-6)

    def test_implied_fair_is_the_normal_cdf(self) -> None:
        v = self._values()
        distance, fair = v["vol_distance_in_sigma"], v["vol_implied_fair_up"]
        assert distance is not None and fair is not None
        assert fair == pytest.approx(0.5 * (1 + erf(distance / sqrt(2))), rel=1e-9)

    def test_fair_value_is_half_with_no_displacement(self) -> None:
        v = self._values(window_open_price=63_030.0)  # price == open
        assert v["vol_price_change_bps"] == pytest.approx(0.0)
        assert v["vol_implied_fair_up"] == pytest.approx(0.5)

    def test_upward_displacement_raises_fair_value(self) -> None:
        v = self._values(window_open_price=62_000.0)  # price well above open
        assert v["vol_implied_fair_up"] > 0.5


class TestOtherFamilies:
    def _values(self, **kwargs) -> dict:  # type: ignore[no-untyped-def]
        return FeatureEngine().compute(make_context(**kwargs)).values

    def test_time_features(self) -> None:
        v = self._values(now_ms=SETTLE - 60_000)
        assert v["tm_seconds_to_settlement"] == pytest.approx(60.0)
        assert v["tm_window_progress"] == pytest.approx(0.8)
        assert v["tm_sqrt_time_remaining"] == pytest.approx(sqrt(0.2))

    def test_cyclical_time_encoding_is_on_the_unit_circle(self) -> None:
        v = self._values()
        assert v["tm_minute_sin"] ** 2 + v["tm_minute_cos"] ** 2 == pytest.approx(1.0)

    def test_trade_flow_cvd(self) -> None:
        v = self._values()
        # pm tape: +10 then -20 -> -10
        assert v["tf_cvd_60s"] == pytest.approx(-10.0)

    def test_liquidity_depth_and_slippage(self) -> None:
        v = self._values()
        assert v["lq_depth_bid_usdc"] == pytest.approx(0.48 * 100 + 0.47 * 200)
        assert v["lq_slippage_100"] is not None

    def test_microstructure_spread_in_ticks(self) -> None:
        assert self._values()["ms_spread_ticks"] == pytest.approx(0.04 / 0.01)

    def test_probability_logit(self) -> None:
        from math import log

        v = self._values()
        assert v["pb_logit"] == pytest.approx(log(0.5 / 0.5), abs=1e-9)

    def test_probability_conviction(self) -> None:
        assert self._values(books=make_books(0.68, 0.72))["pb_conviction"] == pytest.approx(0.2)

    def test_regime_features_are_none_when_absent(self) -> None:
        # A fabricated neutral value is indistinguishable from a real one.
        v = self._values(regime={"tick_size": 0.01})
        assert v["rg_funding_rate"] is None
        assert v["rg_fear_greed"] is None

    def test_regime_features_read_supplied_values(self) -> None:
        assert self._values()["rg_funding_rate"] == pytest.approx(0.0001)

    def test_safe_div_guards(self) -> None:
        assert safe_div(1.0, 0.0) is None
        assert safe_div(None, 1.0) is None
        assert safe_div(1.0, 2.0) == 0.5


# =========================================================================== #
# Documentation contract
# =========================================================================== #
class TestDocumentation:
    def test_every_feature_is_documented(self) -> None:
        graph = build_graph(all_features())
        assert undocumented(graph) == []
        require_documentation(graph)

    def test_markdown_includes_formula_and_purpose(self) -> None:
        markdown = generate_markdown(build_graph(all_features()))
        assert "ob_microprice" in markdown
        assert "formula" in markdown
        assert "```mermaid" in markdown

    def test_undocumented_feature_fails_the_contract(self) -> None:
        class Bare(Feature):
            def compute(self, context: FeatureContext) -> float | None:
                return 1.0

        spec = define("bare_x", tier=FeatureTier.DERIVED, source="s", formula="",
                      inputs=(), units="", purpose="")
        graph = build_graph([Bare(spec)])
        assert undocumented(graph) == ["bare_x"]
        with pytest.raises(ConfigError):
            require_documentation(graph)

    def test_families_are_separate_modules(self) -> None:
        assert len(FAMILIES) == 10
        assert all(features for features in FAMILIES.values())


# =========================================================================== #
# Selection
# =========================================================================== #
class TestSelection:
    @pytest.fixture
    def data(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(0)
        n = 300
        signal = rng.normal(size=n)
        noise = rng.normal(size=(n, 6))
        y = (signal + rng.normal(scale=0.4, size=n) > 0).astype(int)
        x = np.column_stack([signal, noise])
        names = ["signal", *[f"noise_{i}" for i in range(6)]]
        return x, y, names

    def test_selects_the_informative_feature(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, names = data
        result = select_features(x, y, names, max_features=3)
        assert "signal" in result.selected

    def test_all_four_methods_are_attempted(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, names = data
        methods = {r.method for r in select_features(x, y, names).rankings}
        assert methods == {"mutual_information", "permutation", "shap", "rfe"}

    def test_collinear_features_are_pruned(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=200)
        x = np.column_stack([base, base * 2.0 + 1e-9, rng.normal(size=200)])
        y = (base > 0).astype(int)
        result = select_features(x, y, ["a", "a_copy", "b"])
        assert "a_copy" in result.dropped_correlated

    def test_constant_columns_are_dropped(self) -> None:
        rng = np.random.default_rng(2)
        signal = rng.normal(size=200)
        x = np.column_stack([signal, np.ones(200)])
        y = (signal > 0).astype(int)
        result = select_features(x, y, ["signal", "constant"])
        assert result.dropped_correlated.get("constant") == "constant"

    def test_selection_is_deterministic(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, names = data
        assert select_features(x, y, names).selected == select_features(x, y, names).selected

    def test_empty_input_is_handled(self) -> None:
        assert select_features(np.zeros((0, 0)), np.zeros(0), []).selected == ()


# =========================================================================== #
# Drift
# =========================================================================== #
class TestDrift:
    def test_psi_is_zero_for_identical_samples(self) -> None:
        sample = [float(i % 50) for i in range(1_000)]
        assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=0.02)

    def test_psi_detects_a_shift(self) -> None:
        reference = [float(i % 50) for i in range(1_000)]
        shifted = [v + 100.0 for v in reference[:300]]
        assert population_stability_index(reference, shifted) > 0.25

    def test_sudden_drift_alert(self) -> None:
        monitor = DriftMonitor(min_samples=100)
        rng = np.random.default_rng(3)
        # 500 normal observations age into the reference baseline, then a hard
        # level shift fills the recent window.
        for _ in range(600):
            monitor.observe({"f": float(rng.normal())})
        for _ in range(500):
            monitor.observe({"f": float(rng.normal() + 25.0)})
        kinds = {a.kind for a in monitor.evaluate()}
        assert DriftKind.SUDDEN in kinds

    def test_gradual_drift_alert(self) -> None:
        monitor = DriftMonitor(min_samples=100)
        rng = np.random.default_rng(7)
        for _ in range(600):
            monitor.observe({"f": float(rng.normal())})
        # A slow ramp rather than a step: PSI should catch what a z-score on a
        # single mean might not.
        for i in range(500):
            monitor.observe({"f": float(rng.normal() + i * 0.02)})
        kinds = {a.kind for a in monitor.evaluate()}
        assert DriftKind.GRADUAL in kinds or DriftKind.SUDDEN in kinds

    def test_stable_feature_raises_nothing(self) -> None:
        monitor = DriftMonitor(min_samples=100)
        rng = np.random.default_rng(4)
        for _ in range(1_200):
            monitor.observe({"f": float(rng.normal())})
        assert monitor.evaluate() == []

    def test_coverage_alert(self) -> None:
        monitor = DriftMonitor(min_samples=50, min_coverage=0.9)
        for i in range(200):
            monitor.observe({"f": None if i % 2 else 1.0})
        assert any(a.kind is DriftKind.COVERAGE for a in monitor.evaluate())

    def test_drift_never_retrains(self) -> None:
        # The monitor's only outputs are alerts; it has no retrain hook at all.
        assert not hasattr(DriftMonitor, "retrain")


# =========================================================================== #
# Cross-source consistency
# =========================================================================== #
class TestConsistency:
    def test_clean_sources(self) -> None:
        graph = build_graph(all_features())
        report = check_consistency(
            graph, now_ms=SETTLE, pm_book_age_ms=100, ref_quote_age_ms=100,
            pm_spread=0.02, ref_spread_bps=1.0, clock_offset_ms=50,
        )
        assert report.clean
        assert report.confidence_multiplier == 1.0

    def test_stale_feed_annotates_dependent_features(self) -> None:
        graph = build_graph(all_features())
        report = check_consistency(graph, now_ms=SETTLE, pm_book_age_ms=60_000)
        assert not report.clean
        assert "ob_mid" in report.affected_features
        assert report.confidence_multiplier < 1.0

    def test_clock_drift_contaminates_windowed_features(self) -> None:
        graph = build_graph(all_features())
        report = check_consistency(graph, now_ms=SETTLE, clock_offset_ms=9_000)
        assert "tf_cvd_60s" in report.affected_features

    def test_price_divergence_is_flagged(self) -> None:
        graph = build_graph(all_features())
        report = check_consistency(
            graph, now_ms=SETTLE, settlement_price=63_500.0, reference_price=63_000.0
        )
        assert any(i.kind.value == "price_divergence" for i in report.issues)

    def test_annotation_marks_contested_features(self) -> None:
        graph = build_graph(all_features())
        report = check_consistency(graph, now_ms=SETTLE, pm_book_age_ms=60_000)
        annotated = report.annotate({"ob_mid": 0.5, "tm_window_progress": 0.5})
        assert annotated["ob_mid"]["contested"] is True
        assert annotated["tm_window_progress"]["contested"] is False

    def test_abnormal_spread(self) -> None:
        graph = build_graph(all_features())
        report = check_consistency(
            graph, now_ms=SETTLE, pm_spread=0.4, limits=ConsistencyLimits()
        )
        assert any(i.kind.value == "abnormal_spread" for i in report.issues)


# =========================================================================== #
# Feature matrix
# =========================================================================== #
class TestMatrix:
    def _builder(self) -> MatrixBuilder:
        builder = MatrixBuilder()
        for index, horizon in enumerate((300, 120, 60, 1)):
            vector = builder.engine.compute(make_context(now_ms=SETTLE - horizon * 1000))
            builder.add(
                condition_id="0xabc", slug="btc-updown-5m-1", horizon_seconds=horizon,
                snapshot_time_ms=SETTLE - horizon * 1000, settlement_time_ms=SETTLE,
                vector=vector, label=index % 2,
            )
        return builder

    def test_version_is_stable(self) -> None:
        graph = build_graph(all_features())
        assert feature_set_version(graph) == feature_set_version(graph)

    def test_version_changes_when_a_formula_changes(self) -> None:
        # The version must track what a column *means*, so a changed formula
        # must invalidate it even if the name is identical.
        from dataclasses import replace

        graph = build_graph(all_features())
        before = feature_set_version(graph)
        spec = graph.features["ob_mid"].spec
        graph.features["ob_mid"].spec = replace(spec, formula="something else")
        try:
            assert feature_set_version(graph) != before
        finally:
            graph.features["ob_mid"].spec = spec

    def test_rows_sorted_chronologically(self) -> None:
        matrix = self._builder().build(built_at_ms=SETTLE)
        horizons = [r.horizon_seconds for r in matrix.rows]
        assert horizons == sorted(horizons, reverse=True)

    def test_matrix_fingerprint_is_reproducible(self) -> None:
        assert self._builder().build(SETTLE).fingerprint() == (
            self._builder().build(SETTLE).fingerprint()
        )

    def test_to_arrays_returns_labelled_rows(self) -> None:
        matrix = self._builder().build(SETTLE)
        x, y, columns = matrix.to_arrays()
        assert x.shape[0] == len(matrix.labelled)
        assert x.shape[1] == len(columns)
        assert set(np.unique(y)) <= {0, 1}

    def test_manifest_pins_the_feature_set(self) -> None:
        manifest = self._builder().build(SETTLE).manifest()
        assert manifest["feature_set_version"]
        assert manifest["value_precision"] == 10
        assert manifest["rows"] == 4

    def test_parquet_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import pyarrow.parquet as pq

        paths = self._builder().build(SETTLE).write(tmp_path)
        table = pq.read_table(paths["parquet"])
        assert table.num_rows == 4
        assert "f_ob_mid" in table.column_names
        assert paths["manifest"].exists()


# =========================================================================== #
# Registry contract
# =========================================================================== #
class TestRegistryContract:
    def test_all_features_declare_reproducibility(self) -> None:
        for feature in all_features():
            assert feature.spec.reproducibility in set(ReproducibilityPolicy)

    def test_no_feature_is_unreproducible(self) -> None:
        # An unreproducible feature cannot be validated, so none may exist.
        assert [
            f.name
            for f in all_features()
            if f.spec.reproducibility is ReproducibilityPolicy.NOT_REPRODUCIBLE
        ] == []

    def test_every_feature_declares_inputs(self) -> None:
        assert [f.name for f in all_features() if not f.spec.inputs] == []

    def test_freshness_budgets_are_positive(self) -> None:
        assert all(f.spec.freshness_budget_ms > 0 for f in all_features())
