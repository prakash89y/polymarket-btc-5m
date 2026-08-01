"""Live feeds: book, tape, protocol handling, registry, readiness, baselines.

Book and tape logic is tested against hand-built state rather than a socket, so
every microstructure metric is verified against a book whose correct answer can
be computed by hand. Protocol handling is tested against **real frame shapes**
captured from the live CLOB socket on 2026-07-31.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmbtc.config import load_config
from pmbtc.dataset.providers import SnapshotContext
from pmbtc.dataset.quality import QualityScorer
from pmbtc.exceptions import ConfigError
from pmbtc.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    FeatureTier,
    ReproducibilityPolicy,
)
from pmbtc.live import (
    BookPair,
    LivePolymarketProvider,
    OrderBook,
    PolymarketMarketFeed,
    TickArchive,
    Trade,
    TradeTape,
)
from pmbtc.models import (
    GradientBoostingBaseline,
    LogisticBaseline,
    MarketFavourite,
    MarketUnderdog,
    RandomPredictor,
    check_readiness,
    run_baselines,
    score,
)

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

UP_TOKEN = "75947675343300382612239152835580192418310954975177482443770591765733661013763"
DOWN_TOKEN = "82813676602798150736076499247275218984847413034838945124705815442226616387084"


@pytest.fixture
def config(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})


# =========================================================================== #
# Order book
# =========================================================================== #
class TestOrderBook:
    @pytest.fixture
    def book(self) -> OrderBook:
        b = OrderBook(asset_id=UP_TOKEN)
        b.replace(
            bids=[(0.48, 100.0), (0.47, 200.0), (0.46, 300.0)],
            asks=[(0.52, 50.0), (0.53, 150.0), (0.54, 250.0)],
            timestamp_ms=1_000,
        )
        return b

    def test_touch(self, book: OrderBook) -> None:
        assert book.best_bid == 0.48
        assert book.best_ask == 0.52
        assert book.spread == pytest.approx(0.04)
        assert book.mid == pytest.approx(0.50)

    def test_sides_are_sorted_regardless_of_arrival_order(self) -> None:
        # Polymarket sends levels ascending on both sides; assuming an order
        # would invert every imbalance metric if it ever changed.
        b = OrderBook()
        b.replace(bids=[(0.46, 1.0), (0.48, 1.0)], asks=[(0.54, 1.0), (0.52, 1.0)], timestamp_ms=1)
        assert b.best_bid == 0.48
        assert b.best_ask == 0.52
        assert [level.price for level in b.bids] == [0.48, 0.46]
        assert [level.price for level in b.asks] == [0.52, 0.54]

    def test_microprice_leans_to_the_thin_side(self, book: OrderBook) -> None:
        # Ask has 50, bid has 100: the ask is about to be consumed, so fair
        # value sits above the mid.
        micro = book.microprice
        assert micro is not None
        assert micro > book.mid  # type: ignore[operator]

    def test_imbalance(self, book: OrderBook) -> None:
        assert book.imbalance(1) == pytest.approx((100 - 50) / 150)
        assert book.imbalance(3) == pytest.approx((600 - 450) / 1050)

    def test_incremental_update(self, book: OrderBook) -> None:
        book.apply_change(0.49, 500.0, "BUY", 1_100)
        assert book.best_bid == 0.49
        assert book.best_bid_size == 500.0

    def test_zero_size_removes_the_level(self, book: OrderBook) -> None:
        # A zero level kept in place would put a phantom price at the touch.
        book.apply_change(0.48, 0.0, "BUY", 1_100)
        assert book.best_bid == 0.47
        assert len(book.bids) == 2

    def test_price_for_size_walks_the_book(self, book: OrderBook) -> None:
        # 100 shares: 50 @ 0.52 then 50 @ 0.53
        assert book.price_for_size(100.0, "BUY") == pytest.approx(0.525)

    def test_price_for_size_returns_none_when_book_too_thin(self, book: OrderBook) -> None:
        assert book.price_for_size(10_000.0, "BUY") is None

    def test_slippage(self, book: OrderBook) -> None:
        assert book.slippage_for_size(100.0, "BUY") == pytest.approx(0.005)

    def test_one_sided_book_has_no_mid(self) -> None:
        b = OrderBook()
        b.replace(bids=[(0.4, 10.0)], asks=[], timestamp_ms=1)
        assert b.is_two_sided is False
        assert b.mid is None
        assert b.imbalance(1) == pytest.approx(1.0)

    def test_mid_is_clamped(self) -> None:
        b = OrderBook()
        b.replace(bids=[(0.0, 10.0)], asks=[(0.0, 10.0)], timestamp_ms=1)
        assert b.mid is not None and b.mid > 0

    def test_clear_wipes_state(self, book: OrderBook) -> None:
        book.clear()
        assert book.best_bid is None
        assert book.is_two_sided is False


class TestBookPair:
    def _pair(self, up_mid: float, down_mid: float) -> BookPair:
        pair = BookPair()
        pair.up.asset_id, pair.down.asset_id = UP_TOKEN, DOWN_TOKEN
        pair.up.replace([(up_mid - 0.005, 10.0)], [(up_mid + 0.005, 10.0)], 1)
        pair.down.replace([(down_mid - 0.005, 10.0)], [(down_mid + 0.005, 10.0)], 1)
        return pair

    def test_implied_up_reconciles_both_books(self) -> None:
        pair = self._pair(0.60, 0.40)
        assert pair.implied_up == pytest.approx(0.60)

    def test_complement_gap_flags_disagreement(self) -> None:
        pair = self._pair(0.60, 0.45)
        assert pair.complement_gap == pytest.approx(0.05)
        assert pair.implied_up == pytest.approx(0.575)

    def test_book_lookup_by_asset(self) -> None:
        pair = self._pair(0.5, 0.5)
        assert pair.book_for(UP_TOKEN) is pair.up
        assert pair.book_for("unknown") is None


# =========================================================================== #
# Trade tape
# =========================================================================== #
class TestTradeTape:
    @pytest.fixture
    def tape(self) -> TradeTape:
        t = TradeTape()
        for i, (price, size, sign) in enumerate(
            [(0.50, 10, 1), (0.51, 20, 1), (0.52, 5, -1), (0.51, 15, -1)]
        ):
            t.add(Trade(price, size, sign, 1_000 + i * 1_000))
        return t

    def test_cvd(self, tape: TradeTape) -> None:
        # +10 +20 -5 -15 = +10
        assert tape.cvd(60_000, 5_000) == pytest.approx(10.0)

    def test_volume_delta_ratio_is_scale_free(self, tape: TradeTape) -> None:
        assert tape.volume_delta_ratio(60_000, 5_000) == pytest.approx(10.0 / 50.0)

    def test_window_excludes_the_future(self, tape: TradeTape) -> None:
        # The tape holds trades up to t=4000; asking as of t=2000 must not see
        # them. This is the leakage rule applied inside the tape.
        assert tape.trade_count(60_000, 2_000) == 2
        assert tape.cvd(60_000, 2_000) == pytest.approx(30.0)

    def test_window_excludes_the_distant_past(self, tape: TradeTape) -> None:
        # Trades at t=1000..4000; a 1.5s window as of t=4000 covers (2500, 4000].
        assert tape.trade_count(1_500, 4_000) == 2

    def test_vwap(self, tape: TradeTape) -> None:
        expected = (0.50 * 10 + 0.51 * 20 + 0.52 * 5 + 0.51 * 15) / 50
        assert tape.vwap(60_000, 5_000) == pytest.approx(expected)

    def test_realized_vol_needs_enough_trades(self) -> None:
        tape = TradeTape()
        tape.add(Trade(0.5, 1, 1, 1_000))
        assert tape.realized_vol(60_000, 2_000) is None

    def test_realized_vol(self, tape: TradeTape) -> None:
        assert tape.realized_vol(60_000, 5_000) is not None

    def test_trade_intensity(self, tape: TradeTape) -> None:
        assert tape.trade_intensity(4_000, 5_000) == pytest.approx(3 / 4.0)

    def test_old_trades_are_trimmed(self) -> None:
        tape = TradeTape(max_window_ms=5_000)
        tape.add(Trade(0.5, 1, 1, 1_000))
        tape.add(Trade(0.5, 1, 1, 10_000))
        assert len(tape.trades) == 1


# =========================================================================== #
# CLOB protocol — real frame shapes
# =========================================================================== #
class TestClobProtocol:
    @pytest.fixture
    def feed(self) -> PolymarketMarketFeed:
        return PolymarketMarketFeed(
            url="wss://example/market",
            up_token_id=UP_TOKEN,
            down_token_id=DOWN_TOKEN,
            slug="btc-updown-5m-1785520200",
        )

    async def test_book_event(self, feed: PolymarketMarketFeed) -> None:
        await feed.handle(
            {
                "event_type": "book",
                "asset_id": UP_TOKEN,
                "timestamp": "1785520314136",
                "hash": "abc",
                "bids": [{"price": "0.48", "size": "100"}, {"price": "0.47", "size": "50"}],
                "asks": [{"price": "0.52", "size": "80"}],
            },
            1_785_520_314_200,
        )
        assert feed.books.up.best_bid == 0.48
        assert feed.books.up.best_ask == 0.52
        assert feed.books.up.last_update_ms == 1_785_520_314_136

    async def test_price_change_is_batched_across_assets(
        self, feed: PolymarketMarketFeed
    ) -> None:
        # Real shape: one frame carrying updates for both outcome tokens.
        await feed.handle(
            {
                "event_type": "price_change",
                "timestamp": "1785520375075",
                "price_changes": [
                    {"asset_id": DOWN_TOKEN, "price": "0.01", "size": "3951.86", "side": "BUY"},
                    {"asset_id": UP_TOKEN, "price": "0.99", "size": "3951.86", "side": "SELL"},
                ],
            },
            1_785_520_375_100,
        )
        assert feed.books.down.best_bid == 0.01
        assert feed.books.up.best_ask == 0.99

    async def test_last_trade_price_signs_correctly(
        self, feed: PolymarketMarketFeed
    ) -> None:
        await feed.handle(
            {
                "event_type": "last_trade_price",
                "asset_id": DOWN_TOKEN,
                "price": "0.03",
                "size": "10",
                "side": "BUY",
                "timestamp": "1785520376130",
            },
            1_785_520_376_200,
        )
        assert feed.down_tape.total_count == 1
        assert feed.down_tape.cvd(60_000, 1_785_520_377_000) == pytest.approx(10.0)

    async def test_frames_arrive_as_a_list(self, feed: PolymarketMarketFeed) -> None:
        # The socket batches several events into one frame.
        await feed.handle(
            [
                {"event_type": "book", "asset_id": UP_TOKEN, "timestamp": "1000",
                 "bids": [{"price": "0.4", "size": "1"}], "asks": [{"price": "0.6", "size": "1"}]},
                {"event_type": "last_trade_price", "asset_id": UP_TOKEN, "price": "0.5",
                 "size": "2", "side": "SELL", "timestamp": "1001"},
            ],
            1_100,
        )
        assert feed.books.up.is_two_sided
        assert feed.up_tape.total_count == 1

    async def test_unknown_asset_is_ignored(self, feed: PolymarketMarketFeed) -> None:
        await feed.handle(
            {"event_type": "book", "asset_id": "other", "timestamp": "1",
             "bids": [{"price": "0.4", "size": "1"}], "asks": []},
            2,
        )
        assert feed.books.up.best_bid is None

    def test_disconnect_clears_books(self, feed: PolymarketMarketFeed) -> None:
        # After a gap an incrementally-maintained book is wrong, not stale, and
        # a wrong book prices trades confidently at the wrong level.
        feed.books.up.replace([(0.48, 1.0)], [(0.52, 1.0)], 1)
        feed.on_disconnect()
        assert feed.books.up.is_two_sided is False

    async def test_archive_captures_frames(
        self, feed: PolymarketMarketFeed, tmp_path: Path
    ) -> None:
        archive = TickArchive(tmp_path / "ticks", buffer_frames=1)
        feed.archive = archive
        await feed.handle({"event_type": "book", "asset_id": UP_TOKEN, "timestamp": "1",
                           "bids": [], "asks": []}, 5)
        archive.close()
        files = archive.files()
        assert files
        frames = archive.read_frames(files[0])
        assert frames and frames[0][0] == 5


# =========================================================================== #
# Feature registry — tiers, provenance, reproducibility
# =========================================================================== #
class TestRegistry:
    def test_anonymous_features_are_refused(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(ConfigError, match="not registered"):
            registry.get("mystery_column")

    def test_conflicting_declaration_is_refused(self) -> None:
        registry = FeatureRegistry()
        spec = FeatureSpec("x", FeatureTier.REAL_TIME, "a", "d",
                           ReproducibilityPolicy.REST_REPLAYABLE)
        registry.register(spec)
        with pytest.raises(ConfigError, match="already registered"):
            registry.register(
                FeatureSpec("x", FeatureTier.DELAYED, "a", "d",
                            ReproducibilityPolicy.REST_REPLAYABLE)
            )

    def test_unreproducible_features_are_not_trainable(self) -> None:
        registry = FeatureRegistry()
        registry.register(
            FeatureSpec("ghost", FeatureTier.REAL_TIME, "s", "d",
                        ReproducibilityPolicy.NOT_REPRODUCIBLE)
        )
        assert registry.trainable() == ()
        assert len(registry.unreproducible()) == 1

    def test_freshness_decays_to_zero(self) -> None:
        spec = FeatureSpec("f", FeatureTier.REAL_TIME, "s", "d",
                           ReproducibilityPolicy.ARCHIVED_STREAM, freshness_budget_ms=1_000)
        assert spec.freshness(0) == 1.0
        assert spec.freshness(500) == pytest.approx(0.5)
        assert spec.freshness(5_000) == 0.0

    def test_delayed_tier_is_not_a_tradeable_signal(self) -> None:
        # Gamma lives here: 29-72s stale is regime context, never a trigger.
        assert FeatureTier.DELAYED.is_tradeable_signal is False
        assert FeatureTier.REAL_TIME.is_tradeable_signal is True

    def test_live_features_are_registered_and_reproducible(self) -> None:
        from pmbtc.features.registry import REGISTRY

        spec = REGISTRY.get("pm_imbalance_5")
        assert spec.tier is FeatureTier.REAL_TIME
        assert spec.reproducibility is ReproducibilityPolicy.ARCHIVED_STREAM
        assert spec.usable_for_training is True


# =========================================================================== #
# Live providers
# =========================================================================== #
class TestLiveProviders:
    async def test_observations_are_bounded_by_the_snapshot_instant(self) -> None:
        feed = PolymarketMarketFeed("wss://x", UP_TOKEN, DOWN_TOKEN, slug="s")
        feed.books.up.replace([(0.48, 100.0)], [(0.52, 50.0)], 1_000)
        # A trade that lands after the snapshot instant must not be counted.
        feed.up_tape.add(Trade(0.51, 10, 1, 900))
        feed.up_tape.add(Trade(0.51, 999, 1, 5_000))

        provider = LivePolymarketProvider(QualityScorer(), feed)
        context = SnapshotContext(
            condition_id="c", slug="s", snapshot_time_ms=1_000,
            settlement_time_ms=61_000, window_open_ms=0, horizon_seconds=60,
            now_ms=5_000, market={},
        )
        observations = {o.name: o for o in await provider.collect(context)}
        assert observations["pm_cvd_10s"].value == pytest.approx(10.0)
        assert all(o.event_time_ms <= 1_000 for o in observations.values())
        assert observations["pm_book_mid"].value == pytest.approx(0.50)
        assert observations["pm_imbalance_1"].value == pytest.approx((100 - 50) / 150)


# =========================================================================== #
# Readiness gate
# =========================================================================== #
class TestReadiness:
    def test_empty_dataset_is_not_ready(self, config) -> None:  # type: ignore[no-untyped-def]
        report = check_readiness(config, [], [])
        assert report.ready is False
        assert "NOT ready" in report.summary()

    def test_failure_names_the_shortfall(self, config) -> None:  # type: ignore[no-untyped-def]
        report = check_readiness(config, [], [])
        labelled = next(c for c in report.checks if c.name == "labelled_markets")
        assert labelled.required == config.training.min_labelled_markets
        assert labelled.actual == 0

    def test_raise_gives_an_actionable_message(self, config) -> None:  # type: ignore[no-untyped-def]
        from pmbtc.exceptions import InsufficientDataError

        with pytest.raises(InsufficientDataError) as exc:
            check_readiness(config, [], []).raise_if_not_ready()
        assert "labelled_markets" in str(exc.value)

    def test_thresholds_are_configurable(self, tmp_path: Path) -> None:
        relaxed = load_config(
            SHIPPED_CONFIG,
            app={"base_dir": str(tmp_path)},
            training={
                "min_labelled_markets": 1,
                "min_complete_timelines": 1,
                "min_feature_coverage": 0.0,
                "min_class_balance": 0.01,
                "max_low_quality_fraction": 1.0,
                "min_samples_to_train": 100,
            },
        )
        assert relaxed.training.min_labelled_markets == 1
        assert relaxed.training.min_samples_to_train == 100


# =========================================================================== #
# Baselines
# =========================================================================== #
class TestBaselines:
    @pytest.fixture
    def data(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(0)
        n = 400
        signal = rng.normal(size=n)
        y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
        x = np.column_stack([signal, rng.normal(size=n)])
        # A market probability that is informative but imperfect.
        market = np.clip(0.5 + 0.18 * signal, 0.02, 0.98)
        return x, y, market

    def test_scorecard_metrics(self) -> None:
        card = score("t", np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        assert card.accuracy == 1.0
        assert card.brier < 0.05
        assert card.log_loss < 0.3

    def test_random_is_a_constant_half(self) -> None:
        probs = RandomPredictor().predict_proba(np.zeros((5, 2)))
        assert np.allclose(probs, 0.5)
        assert score("random", np.array([1, 0, 1, 0, 1]), probs).brier == pytest.approx(0.25)

    def test_market_favourite_uses_the_book(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, market = data
        card = MarketFavourite().evaluate(x, y, market)
        assert card.brier < 0.25  # better than a coin flip

    def test_underdog_is_the_inverse(self, data) -> None:  # type: ignore[no-untyped-def]
        x, _y, market = data
        fav = MarketFavourite().predict_proba(x, market)
        dog = MarketUnderdog().predict_proba(x, market)
        assert np.allclose(fav + dog, 1.0)

    def test_underdog_loses_to_favourite_on_an_informative_book(self, data) -> None:  # type: ignore[no-untyped-def]
        # If this ever inverts on real data, the labels or the market mapping
        # are wrong -- not the model.
        x, y, market = data
        assert MarketFavourite().evaluate(x, y, market).brier < (
            MarketUnderdog().evaluate(x, y, market).brier
        )

    def test_logistic_fits_and_beats_random(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, _ = data
        model = LogisticBaseline().fit(x[:300], y[:300])
        card = model.evaluate(x[300:], y[300:])
        assert card.brier < 0.25

    def test_gradient_boosting_handles_missing_values(self, data) -> None:  # type: ignore[no-untyped-def]
        # Missing features are a real, recorded state in this dataset.
        x, y, _ = data
        x = x.copy()
        x[::10, 0] = np.nan
        model = GradientBoostingBaseline().fit(x[:300], y[:300])
        assert model.evaluate(x[300:], y[300:]).brier < 0.3

    def test_full_report_and_gate(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, market = data
        report = run_baselines(x[:300], y[:300], x[300:], y[300:], market[300:])
        assert len(report.scores) == 5
        assert report.best is not None

        # A candidate must beat every baseline, not just the weakest.
        weak = score("weak", y[300:], np.full(len(y[300:]), 0.5))
        assert report.clears_all(weak) is False
        assert report.blocking(weak)

    def test_a_perfect_candidate_clears_everything(self, data) -> None:  # type: ignore[no-untyped-def]
        x, y, market = data
        report = run_baselines(x[:300], y[:300], x[300:], y[300:], market[300:])
        perfect = score("oracle", y[300:], y[300:].astype(float) * 0.98 + 0.01)
        assert report.clears_all(perfect) is True
        assert report.blocking(perfect) == []
