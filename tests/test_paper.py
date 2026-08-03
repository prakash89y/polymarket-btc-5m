"""Module 9 tests: paper trading against the live book.

The property that matters most is pinned first: paper and backtest must run the
*same* sequence. If they can diverge, paper trading cannot validate the
backtest, and the whole module is theatre.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pmbtc.config import Config
from pmbtc.constants import Outcome, SkipReason, TradeStatus
from pmbtc.paper import (
    LiveQuoteSource,
    PaperJournal,
    PaperTradingEngine,
    TapePrint,
    build_report,
    measure_execution,
    would_have_filled,
)
from pmbtc.trading.costs import Quote
from pmbtc.utils.timeutils import utc_now_ms


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(app={"base_dir": str(tmp_path)})


@pytest.fixture
def journal(tmp_path: Path) -> PaperJournal:
    return PaperJournal(tmp_path / "paper.jsonl")


def _quote(bid: float = 0.50, ask: float = 0.52, depth: float = 5_000.0) -> Quote:
    return Quote(
        best_bid=bid, best_ask=ask, bid_depth_usdc=depth, ask_depth_usdc=depth, age_ms=150
    )


class _StaticQuotes:
    def __init__(self, quote: Quote | None) -> None:
        self.quote = quote

    def quote_for(self, condition_id: str, now_ms: int) -> Quote | None:
        return self.quote


def _engine(config: Config, journal: PaperJournal, quote: Quote | None,
            prob: float = 0.5) -> PaperTradingEngine:
    return PaperTradingEngine(
        config,
        journal,
        quotes=_StaticQuotes(quote),
        model=lambda _cid, _q: prob,
        model_name="test",
    )


# --------------------------------------------------------------------------- #
class TestSharedPipeline:
    """Paper and backtest must not be able to drift."""

    def test_both_engines_call_the_same_sequence(self) -> None:
        import inspect

        from pmbtc.backtest import engine as backtest_engine
        from pmbtc.paper import engine as paper_engine

        for module in (backtest_engine, paper_engine):
            source = inspect.getsource(module)
            assert "evaluate_opportunity(" in source, f"{module.__name__} bypasses the pipeline"

    def test_paper_does_not_reimplement_decide_size_risk_fill(self) -> None:
        """A second implementation would make the comparison meaningless."""
        import inspect

        from pmbtc.paper import engine as paper_engine

        source = inspect.getsource(paper_engine)
        assert "self.decisions.decide(" not in source
        assert "self.sizer.size(" not in source
        assert "self.ledger.check(" not in source


class TestDecisionRecording:
    """Every window is recorded — traded or skipped."""

    def test_a_skip_is_recorded_with_its_reason(
        self, config: Config, journal: PaperJournal
    ) -> None:
        # A fair book against a fair model has no edge: it must abstain.
        engine = _engine(config, journal, _quote(), prob=0.5)
        record = engine.evaluate(
            condition_id="0xabc",
            slug="btc-updown-5m-1",
            settlement_time_ms=utc_now_ms() + 30_000,
        )
        assert record is not None
        assert not record.traded
        assert record.skip_reason is not None
        assert journal.summary()["skipped"] == 1

    def test_the_book_is_captured_on_every_decision(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote(bid=0.40, ask=0.43))
        record = engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        assert record is not None
        assert record.best_bid == pytest.approx(0.40)
        assert record.best_ask == pytest.approx(0.43)
        assert record.quoted_spread == pytest.approx(0.03)
        assert record.market_prob_up == pytest.approx(0.415)

    def test_a_market_is_never_evaluated_twice(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote())
        settles = utc_now_ms() + 30_000
        first = engine.evaluate(condition_id="0xabc", slug="s", settlement_time_ms=settles)
        second = engine.evaluate(condition_id="0xabc", slug="s", settlement_time_ms=settles)
        assert first is not None and second is None
        assert journal.summary()["decisions"] == 1

    def test_it_waits_for_the_decision_horizon(
        self, config: Config, journal: PaperJournal
    ) -> None:
        """Deciding early is a different, worse-informed decision."""
        engine = _engine(config, journal, _quote())
        far = utc_now_ms() + 290_000  # T-290, long before the horizon
        assert engine.evaluate(condition_id="0xabc", slug="s", settlement_time_ms=far) is None

    def test_the_late_window_bar_does_not_apply_at_the_horizon(
        self, config: Config, journal: PaperJournal
    ) -> None:
        """Regression, found in live paper trading.

        Firing just *inside* the horizon made every decision "late", applying
        the 0.12 late-window edge bar where the backtest at the same horizon
        applies 0.04 — paper was three times stricter than the backtest it
        exists to validate.
        """
        engine = _engine(config, journal, _quote(bid=0.49, ask=0.51), prob=0.60)
        horizon = engine.decision_horizon()
        record = engine.evaluate(
            condition_id="0xabc",
            slug="s",
            settlement_time_ms=utc_now_ms() + horizon * 1000 + 500,
        )
        assert record is not None
        assert "late-window" not in (record.skip_reason or ""), record.skip_reason
        detail_is_late = record.skip_reason == "edge_too_small" and record.edge < 0.12
        assert not (detail_is_late and record.edge >= 0.04), (
            "the late-window bar applied at the horizon"
        )

    def test_it_does_not_evaluate_below_the_horizon(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote())
        horizon = engine.decision_horizon()
        too_late = utc_now_ms() + (horizon * 1000) - 5_000
        assert (
            engine.evaluate(condition_id="0xabc", slug="s", settlement_time_ms=too_late)
            is None
        )

    def test_a_settled_window_is_not_evaluated(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote())
        past = utc_now_ms() - 1_000
        assert engine.evaluate(condition_id="0xabc", slug="s", settlement_time_ms=past) is None

    def test_no_quote_means_no_record(self, config: Config, journal: PaperJournal) -> None:
        engine = _engine(config, journal, None)
        assert (
            engine.evaluate(
                condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
            )
            is None
        )


class TestSettlement:
    """The label is Polymarket's outcome and nothing else."""

    def _traded(self, config: Config, journal: PaperJournal):
        # Strong but plausible: 0.78 against a 0.50 book is 1.27 logits apart,
        # inside Module 8.5's 1.50 ceiling, with edge well over min_edge.
        engine = _engine(config, journal, _quote(bid=0.49, ask=0.51), prob=0.78)
        record = engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        return engine, record

    def test_a_winning_trade_pays_one_per_share(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine, record = self._traded(config, journal)
        assert record is not None and record.traded, record.skip_reason
        before = engine.ledger.state.bankroll_usdc
        settlement = engine.settle(record, Outcome.UP)
        assert settlement.won
        assert settlement.status == TradeStatus.SETTLED_WIN.value
        assert settlement.payoff_usdc == pytest.approx(record.expected_shares)
        assert engine.ledger.state.bankroll_usdc > before

    def test_a_losing_trade_costs_the_stake(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine, record = self._traded(config, journal)
        assert record is not None and record.traded
        settlement = engine.settle(record, Outcome.DOWN)
        assert not settlement.won
        assert settlement.pnl_usdc == pytest.approx(-record.expected_cost_usdc)

    def test_an_unresolved_market_returns_the_stake(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine, record = self._traded(config, journal)
        assert record is not None and record.traded
        opened = engine.ledger.state.bankroll_usdc
        settlement = engine.settle(record, None)
        assert settlement.status == TradeStatus.VOIDED.value
        assert engine.ledger.state.bankroll_usdc == pytest.approx(
            opened + record.expected_cost_usdc
        )

    def test_settling_a_skip_books_no_money(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote(), prob=0.5)
        record = engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        assert record is not None and not record.traded
        before = engine.ledger.state.bankroll_usdc
        settlement = engine.settle(record, Outcome.UP)
        assert settlement.pnl_usdc == 0.0
        assert engine.ledger.state.bankroll_usdc == pytest.approx(before)


class TestExecutionQuality:
    """The measurement a backtest cannot make."""

    def test_a_buy_fills_when_the_tape_prints_at_or_below(self) -> None:
        assert would_have_filled(Outcome.UP, 0.55, 0.54)
        assert would_have_filled(Outcome.UP, 0.55, 0.55)
        assert not would_have_filled(Outcome.UP, 0.55, 0.56)

    def test_a_down_buy_is_the_complement(self) -> None:
        """Buying DOWN at 0.45 fills when UP prints at or above 0.55."""
        assert would_have_filled(Outcome.DOWN, 0.45, 0.56)
        assert not would_have_filled(Outcome.DOWN, 0.45, 0.54)

    def test_fill_probability_counts_only_later_prints(self) -> None:
        """Including the print that triggered the decision would let the
        measurement confirm itself."""
        prints = [
            TapePrint(0.50, 100, 1_000),  # before — must be ignored
            TapePrint(0.54, 100, 2_000),
            TapePrint(0.60, 100, 3_000),
        ]
        quality = measure_execution(
            outcome=Outcome.UP, expected_price=0.55, decided_at_ms=1_500, prints=prints
        )
        assert quality.tape_trades_observed == 2
        assert quality.filling_trades == 1
        assert quality.fill_probability == pytest.approx(0.5)

    def test_no_prints_is_no_evidence_not_zero(self) -> None:
        quality = measure_execution(
            outcome=Outcome.UP, expected_price=0.55, decided_at_ms=1_000, prints=[]
        )
        assert quality.fill_probability is None, "absence of data must not read as 'never fills'"

    def test_an_optimistic_simulation_is_flagged(self) -> None:
        """Expected 0.50 but the market only ever offered 0.57."""
        quality = measure_execution(
            outcome=Outcome.UP,
            expected_price=0.50,
            decided_at_ms=0,
            prints=[TapePrint(0.57, 10, 1), TapePrint(0.59, 10, 2)],
        )
        assert quality.price_error is not None and quality.price_error < 0
        assert quality.optimistic

    def test_a_conservative_simulation_is_not_flagged(self) -> None:
        quality = measure_execution(
            outcome=Outcome.UP,
            expected_price=0.60,
            decided_at_ms=0,
            prints=[TapePrint(0.55, 10, 1)],
        )
        assert quality.price_error is not None and quality.price_error > 0
        assert not quality.optimistic


class TestJournal:
    """Append-only, replayable, joinable."""

    def test_decisions_and_settlements_round_trip(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote(bid=0.49, ask=0.51), prob=0.78)
        record = engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        assert record is not None
        engine.settle(record, Outcome.UP)

        reloaded = PaperJournal(journal.path)
        runs = reloaded.runs()
        assert len(runs) == 1
        assert runs[0].resolved
        assert runs[0].decision.condition_id == "0xabc"
        assert runs[0].label == 1

    def test_pending_lists_unsettled_decisions(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote())
        engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        assert len(journal.pending()) == 1


class TestReport:
    """Periods, the market comparison, and the gate."""

    def _populate(self, config: Config, journal: PaperJournal, n: int = 4) -> None:
        engine = _engine(config, journal, _quote(bid=0.49, ask=0.51), prob=0.78)
        for i in range(n):
            record = engine.evaluate(
                condition_id=f"0x{i}", slug=f"s{i}",
                settlement_time_ms=utc_now_ms() + 30_000,
            )
            if record is not None:
                engine.settle(record, Outcome.UP if i % 2 == 0 else Outcome.DOWN)

    def test_report_builds_and_gates(self, config: Config, journal: PaperJournal) -> None:
        self._populate(config, journal)
        report = build_report(config, journal.runs())
        assert report.cumulative.decisions == 4
        assert report.daily and report.weekly
        assert report.checks, "the promotion gate produced no checks"

    def test_an_empty_record_cannot_be_promoted(
        self, config: Config, journal: PaperJournal
    ) -> None:
        report = build_report(config, journal.runs())
        assert not report.promotion_approved
        assert any(c.name == "min_trades" and not c.passed for c in report.checks)

    def test_the_gate_requires_beating_the_market(
        self, config: Config, journal: PaperJournal
    ) -> None:
        self._populate(config, journal)
        report = build_report(config, journal.runs())
        assert any(c.name == "beats_market_forecast" for c in report.checks)

    def test_the_gate_refuses_an_optimistic_simulation(
        self, config: Config, journal: PaperJournal
    ) -> None:
        self._populate(config, journal)
        report = build_report(config, journal.runs())
        assert any(c.name == "execution_not_optimistic" for c in report.checks)

    def test_skips_are_reported_by_reason(
        self, config: Config, journal: PaperJournal
    ) -> None:
        engine = _engine(config, journal, _quote(), prob=0.5)
        engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        report = build_report(config, journal.runs())
        assert sum(report.skips.values()) == 1


class TestSafety:
    """No gate may be weakened by the paper path."""

    def test_paper_never_places_an_order(self) -> None:
        """No wallet, no signing, no Polygon: assert the code cannot."""
        import inspect

        from pmbtc.paper import engine, execution, journal, report

        for module in (engine, execution, journal, report):
            source = inspect.getsource(module).lower()
            for forbidden in ("private_key", "sign(", "web3", "eth_", "post_order", "submit_order"):
                assert forbidden not in source, f"{module.__name__} references {forbidden}"

    def test_the_risk_ledger_is_the_shared_one(self, config: Config, journal: PaperJournal) -> None:
        from pmbtc.trading.risk import RiskLedger

        engine = _engine(config, journal, _quote())
        assert isinstance(engine.ledger, RiskLedger)
        assert engine.ledger.config is config.risk

    def test_the_kill_switch_is_honoured(self, config: Config, journal: PaperJournal) -> None:
        switch = config.resolved_path(config.risk.kill_switch_file)
        switch.parent.mkdir(parents=True, exist_ok=True)
        switch.write_text("stop", encoding="utf-8")
        try:
            engine = _engine(config, journal, _quote(bid=0.49, ask=0.51), prob=0.78)
            record = engine.evaluate(
                condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
            )
            assert record is not None and not record.traded
            assert record.skip_reason == SkipReason.KILL_SWITCH.value
        finally:
            switch.unlink()

    def test_the_default_model_produces_no_edge(
        self, config: Config, journal: PaperJournal
    ) -> None:
        """An unconfigured paper run must abstain, not trade noise."""
        engine = PaperTradingEngine(config, journal, quotes=_StaticQuotes(_quote()))
        record = engine.evaluate(
            condition_id="0xabc", slug="s", settlement_time_ms=utc_now_ms() + 30_000
        )
        assert record is not None and not record.traded


class TestLiveQuoteSource:
    def test_a_one_sided_book_yields_no_quote(self) -> None:
        class _Book:
            is_two_sided = False

        class _Feed:
            books = type("Books", (), {"up": _Book()})()

        source = LiveQuoteSource()
        source.register("0xabc", _Feed())
        assert source.quote_for("0xabc", utc_now_ms()) is None

    def test_an_unknown_market_yields_no_quote(self) -> None:
        assert LiveQuoteSource().quote_for("0xmissing", utc_now_ms()) is None
