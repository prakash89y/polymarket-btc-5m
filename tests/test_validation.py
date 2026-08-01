"""Module 8.5 — market edge validation: the ceilings.

The pathology these tests defend against is specific and was measured on real
data: the deployed logistic baseline claimed a median edge of 0.83 probability
points at T-30, bought DOWN at 0.035 against a market pricing UP at 0.975, and
reported +2,074% expected value on the trade. Every threshold in the system was
a floor, and a broken model clears floors more easily the worse it gets.
"""

from __future__ import annotations

import math

import pytest

from pmbtc.backtest import disagreement_distribution, scan_horizons
from pmbtc.backtest.adapters import ConstantModel, MarketProbabilityModel
from pmbtc.config import Config
from pmbtc.constants import SkipReason
from pmbtc.trading import DecisionEngine, Quote
from pmbtc.trading.validation import (
    DisagreementMonitor,
    MarketEdgeValidator,
    blend_toward_market,
    disagreement_logits,
    implausible,
    inv_logit,
    logit,
)

SETTLE0 = 1_785_600_000_000
HORIZONS = (300, 240, 180, 120, 60, 30)


@pytest.fixture
def config() -> Config:
    return Config()


def rows(n: int = 120, *, bid: float = 0.49, ask: float = 0.51, depth: float = 500.0):
    out = []
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
                "f_lq_depth_bid_usdc": depth,
                "f_lq_depth_ask_usdc": depth,
                "lat_ob_best_bid": 200,
                "lat_ob_best_ask": 200,
            })
    return out


# --------------------------------------------------------------------------- #
# The log-odds scale
# --------------------------------------------------------------------------- #
class TestLogOdds:
    def test_logit_roundtrips(self) -> None:
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert inv_logit(logit(p)) == pytest.approx(p, abs=1e-9)

    def test_extremes_are_finite(self) -> None:
        """logit(0) is -inf, and an infinite disagreement is not a number any
        gate or log line can reason about."""
        assert math.isfinite(logit(0.0))
        assert math.isfinite(logit(1.0))

    def test_disagreement_is_symmetric_and_non_negative(self) -> None:
        assert disagreement_logits(0.7, 0.4) == pytest.approx(disagreement_logits(0.4, 0.7))
        assert disagreement_logits(0.5, 0.5) == pytest.approx(0.0)

    def test_confident_books_are_harder_to_disagree_with(self) -> None:
        """The whole reason for log-odds. Both gaps are 0.25 in probability
        space; only one of them is an extraordinary claim."""
        ordinary = disagreement_logits(0.75, 0.50)
        extraordinary = disagreement_logits(0.72, 0.97)
        # ln(0.75/0.25) - ln(1) = 1.0986; ln(0.97/0.03) - ln(0.72/0.28) = 2.5316
        assert ordinary == pytest.approx(1.0986, abs=1e-3)
        assert extraordinary == pytest.approx(2.5316, abs=1e-3)
        assert extraordinary > 2 * ordinary

    def test_the_real_pathological_trade_scores_enormous(self) -> None:
        """Model 0.24 against a book at 0.975 — trade 1 of the review."""
        assert disagreement_logits(0.2391, 0.975) > 4.0


# --------------------------------------------------------------------------- #
# Shrinkage toward the market
# --------------------------------------------------------------------------- #
class TestBlending:
    def test_full_trust_is_the_model(self) -> None:
        assert blend_toward_market(0.8, 0.5, 1.0) == pytest.approx(0.8)

    def test_zero_trust_is_the_market(self) -> None:
        assert blend_toward_market(0.8, 0.5, 0.0) == pytest.approx(0.5)

    def test_partial_trust_lands_between(self) -> None:
        blended = blend_toward_market(0.9, 0.5, 0.5)
        assert 0.5 < blended < 0.9

    def test_zero_trust_produces_exactly_no_edge(self) -> None:
        """Deferring entirely to the book must yield the book's own price, so
        the edge is zero and no trade can follow."""
        for market in (0.05, 0.3, 0.5, 0.77, 0.96):
            assert blend_toward_market(0.99, market, 0.0) == pytest.approx(market)

    def test_shrinking_toward_the_market_not_toward_a_coin_flip(self) -> None:
        """Shrinking a 0.99 claim toward 0.5 against a 0.95 book would *invent*
        edge on the DOWN side. Shrinking toward the book cannot."""
        toward_market = blend_toward_market(0.99, 0.95, 0.5)
        assert toward_market > 0.95
        assert abs(toward_market - 0.95) < abs(0.99 - 0.95)


# --------------------------------------------------------------------------- #
# Anomaly monitor
# --------------------------------------------------------------------------- #
class TestDisagreementMonitor:
    def test_no_opinion_before_enough_history(self) -> None:
        """None, not 0.0 — a cold start must not wave everything through as
        'perfectly normal'."""
        m = DisagreementMonitor(min_observations=30)
        m.extend([0.2] * 10)
        assert m.z_score(9.9) is None
        assert not m.ready

    def test_flags_an_outlier_against_a_calm_baseline(self) -> None:
        m = DisagreementMonitor(min_observations=10)
        m.extend([0.20, 0.22, 0.18, 0.25, 0.19, 0.21, 0.23, 0.17, 0.24, 0.20])
        z = m.z_score(4.0)
        assert z is not None and z > 10

    def test_median_and_mad_resist_contamination(self) -> None:
        """Mean and stdev would be dragged by the outliers until the next one
        looks unremarkable. That is the failure mode being avoided."""
        calm = DisagreementMonitor(min_observations=10)
        calm.extend([0.2] * 20)
        poisoned = DisagreementMonitor(min_observations=10)
        poisoned.extend([0.2] * 20 + [8.0, 9.0])
        assert poisoned.z_score(4.0) is not None
        assert poisoned.z_score(4.0) > 5

    def test_capacity_is_bounded(self) -> None:
        m = DisagreementMonitor(capacity=50, min_observations=5)
        m.extend([0.1] * 200)
        assert len(m.observations) == 50

    def test_degenerate_spread_is_infinite_not_zero(self) -> None:
        m = DisagreementMonitor(min_observations=5)
        m.extend([0.3] * 10)
        assert m.z_score(0.3) == 0.0
        assert m.z_score(0.9) == math.inf


# --------------------------------------------------------------------------- #
# The validator
# --------------------------------------------------------------------------- #
class TestMarketEdgeValidator:
    def _v(self, **kw) -> MarketEdgeValidator:
        params = {
            "max_disagreement_logits": 1.5,
            "min_plausible_prob": 0.02,
            "max_disagreement_z": 4.0,
        }
        params.update(kw)
        return MarketEdgeValidator(**params)  # type: ignore[arg-type]

    def test_an_ordinary_claim_passes(self) -> None:
        assert self._v().validate(0.62, 0.50).accepted

    def test_excessive_disagreement_is_refused(self) -> None:
        v = self._v().validate(0.24, 0.975)
        assert not v.accepted
        assert v.skip_reason is SkipReason.EXCESSIVE_DISAGREEMENT

    def test_implausible_certainty_is_refused(self) -> None:
        """The logistic baseline emitted 0.001 on real data."""
        v = self._v().validate(0.001, 0.185)
        assert not v.accepted
        assert v.skip_reason is SkipReason.IMPLAUSIBLE_PROBABILITY

    def test_implausibility_is_checked_before_disagreement(self) -> None:
        """A broken probability should be named broken, not merely distant."""
        v = self._v().validate(0.0001, 0.5)
        assert v.skip_reason is SkipReason.IMPLAUSIBLE_PROBABILITY

    def test_anomalous_edge_is_refused_once_history_exists(self) -> None:
        validator = self._v(max_disagreement_logits=99.0)
        for _ in range(40):
            validator.validate(0.55, 0.50)
        v = validator.validate(0.999999, 0.5)
        assert not v.accepted
        assert v.skip_reason in {
            SkipReason.ANOMALOUS_EDGE,
            SkipReason.IMPLAUSIBLE_PROBABILITY,
        }

    def test_rejected_claims_still_feed_the_monitor(self) -> None:
        """Excluding a model's worst moments would make the baseline useless."""
        validator = self._v()
        validator.validate(0.24, 0.975)
        assert len(validator.monitor.observations) == 1

    def test_the_validator_reports_the_adjusted_probability(self) -> None:
        v = self._v(model_trust=0.0).validate(0.9, 0.5)
        assert v.adjusted_prob == pytest.approx(0.5)


def test_implausible_helper() -> None:
    assert implausible(0.001, 0.02)
    assert implausible(0.999, 0.02)
    assert not implausible(0.5, 0.02)


# --------------------------------------------------------------------------- #
# Integration with the decision gate
# --------------------------------------------------------------------------- #
class TestDecisionGateCeilings:
    def _decide(self, config: Config, quote: Quote, prob: float, **kw):
        params = {
            "model_prob_up": prob,
            "quote": quote,
            "seconds_into_window": 270.0,
            "seconds_to_settlement": 30.0,
        }
        params.update(kw)
        return DecisionEngine(config).decide(**params)  # type: ignore[arg-type]

    def test_the_exact_pathological_trade_is_now_blocked(self, config: Config) -> None:
        """Trade 1 from the statistical review: bought DOWN at 0.035 against a
        book pricing UP at 0.975, on a model probability of 0.239. It lost the
        full 20 USDC stake."""
        book = Quote(
            best_bid=0.96, best_ask=0.97, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=200,
        )
        decision = self._decide(config, book, 0.2391)
        assert not decision.trade
        assert decision.skip_reason is SkipReason.EXCESSIVE_DISAGREEMENT
        assert decision.disagreement_logits > 4.0

    def test_an_ordinary_disagreement_still_trades(self, config: Config) -> None:
        """The ceiling must not close the system down entirely."""
        book = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=200,
        )
        decision = self._decide(
            config, book, 0.72, seconds_into_window=120.0, seconds_to_settlement=180.0
        )
        assert decision.trade
        assert decision.disagreement_logits < config.prediction.max_disagreement_logits

    def test_zero_trust_abstains_everywhere(self) -> None:
        """Deferring to the book yields no edge, hence no trades, by construction."""
        cfg = Config(prediction={"model_trust": 0.0})
        book = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=200,
        )
        for prob in (0.6, 0.7, 0.8, 0.9):
            assert not self._decide(cfg, book, prob).trade

    def test_the_decision_carries_the_adjusted_probability(self) -> None:
        cfg = Config(prediction={"model_trust": 0.5})
        book = Quote(
            best_bid=0.49, best_ask=0.51, bid_depth_usdc=500.0, ask_depth_usdc=500.0,
            age_ms=200,
        )
        d = self._decide(cfg, book, 0.80, seconds_into_window=120.0,
                         seconds_to_settlement=180.0)
        assert d.raw_prob_up == pytest.approx(0.80)
        assert 0.5 < d.adjusted_prob_up < 0.80

    def test_ceilings_are_on_by_default(self, config: Config) -> None:
        assert config.prediction.max_disagreement_logits == 1.5
        assert config.prediction.min_plausible_prob == 0.02
        assert config.prediction.model_trust == 1.0  # blending stays opt-in


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
class TestReports:
    def test_disagreement_distribution_flags_a_broken_model(
        self, config: Config
    ) -> None:
        report = disagreement_distribution(
            config, rows(40), ConstantModel(0.999), horizon_seconds=30
        )
        assert report.samples == 40
        assert report.median > 5.0
        assert report.over_ceiling == pytest.approx(1.0)
        assert not report.healthy

    def test_disagreement_distribution_passes_a_sane_model(
        self, config: Config
    ) -> None:
        report = disagreement_distribution(
            config, rows(40), MarketProbabilityModel(), horizon_seconds=30
        )
        assert report.median == pytest.approx(0.0, abs=1e-6)
        assert report.over_ceiling == 0.0
        assert report.healthy

    def test_horizon_scan_finds_no_edge_in_a_null_strategy(
        self, config: Config
    ) -> None:
        report = scan_horizons(
            config, rows(120), lambda _: MarketProbabilityModel(),
            model_name="market", n_folds=3,
        )
        assert report.horizons
        assert not report.any_evidence_of_edge
        assert report.best is None
        assert "NO EVIDENCE OF EDGE" in report.summary()

    def test_horizon_scan_requires_all_three_stability_conditions(
        self, config: Config
    ) -> None:
        """Profit alone is not stability; nor is a good Brier alone."""
        from pmbtc.backtest.edgescan import HorizonResult

        profitable_but_lucky = HorizonResult(
            horizon_seconds=30, folds=4, trades=50, net_pnl_usdc=100.0,
            profitable_folds=1, model_brier=0.2, market_brier=0.3,
        )
        assert not profitable_but_lucky.stable

        good_brier_no_money = HorizonResult(
            horizon_seconds=30, folds=4, trades=50, net_pnl_usdc=-100.0,
            profitable_folds=3, model_brier=0.1, market_brier=0.3,
        )
        assert not good_brier_no_money.stable

        money_but_worse_than_market = HorizonResult(
            horizon_seconds=30, folds=4, trades=50, net_pnl_usdc=100.0,
            profitable_folds=3, model_brier=0.4, market_brier=0.3,
        )
        assert not money_but_worse_than_market.stable

        all_three = HorizonResult(
            horizon_seconds=30, folds=4, trades=50, net_pnl_usdc=100.0,
            profitable_folds=3, model_brier=0.2, market_brier=0.3,
        )
        assert all_three.stable

    def test_scan_is_deterministic(self, config: Config) -> None:
        data = rows(120)
        a = scan_horizons(config, data, lambda _: ConstantModel(0.62), n_folds=3)
        b = scan_horizons(config, data, lambda _: ConstantModel(0.62), n_folds=3)
        assert a.as_dict() == b.as_dict()
