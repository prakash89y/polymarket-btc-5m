"""The paper-trading record.

Append-only, two record types, joined on ``condition_id``:

``decision``
    Written the instant a window is evaluated — traded *or* skipped, with the
    book that was quoted, the pipeline's verdict, and the fill we expected.

``settlement``
    Written hours later when Polymarket resolves the market, carrying the
    official outcome, the realised P&L, and what the tape said about whether our
    simulated order would really have filled.

Two records rather than one mutable row, deliberately. A paper trade is decided
at T-180 and resolved long afterwards, so a single mutable record would have to
be rewritten in place — and a rewrite is exactly how a decision quietly acquires
knowledge it did not have at the time. Appending a second record makes the
before-and-after separable and the file replayable in order, which is the same
reasoning the dataset store already applies to snapshots and labels.

Skipped windows are recorded with the same weight as traded ones. A paper record
that keeps only its trades cannot answer the question that matters most on a
market this close to fair — *why did we not trade the other 500 windows* — and
would turn a handful of trades into an apparently deliberate strategy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pmbtc.config import Config
from pmbtc.constants import Outcome, TradeStatus
from pmbtc.logging_setup import get_logger
from pmbtc.trading.pipeline import PipelineOutcome, Stage

log = get_logger("pmbtc.paper.journal")

RECORD_VERSION = "1.0.0"


@dataclass
class PaperDecision:
    """One evaluated window, as it looked at the moment of decision."""

    condition_id: str
    slug: str
    settlement_time_ms: int
    horizon_seconds: int
    decided_at_ms: int

    # --- what the pipeline said ---------------------------------------- #
    stage: str = Stage.DECISION.value
    traded: bool = False
    fill_attempted: bool = False
    skip_reason: str | None = None
    outcome: str | None = None

    # --- probabilities -------------------------------------------------- #
    model_prob_up: float = 0.5
    adjusted_prob_up: float = 0.5
    market_prob_up: float = 0.5
    disagreement_logits: float = 0.0
    confidence: float = 0.5
    edge: float = 0.0
    ev_per_usdc: float = 0.0

    # --- the book we quoted against ------------------------------------- #
    best_bid: float | None = None
    best_ask: float | None = None
    bid_depth_usdc: float = 0.0
    ask_depth_usdc: float = 0.0
    quoted_spread: float = 0.0
    book_age_ms: int = 0
    mid_price: float = 0.0
    touch_price: float = 0.0

    # --- the order we would have sent ----------------------------------- #
    desired_stake_usdc: float = 0.0
    expected_price: float = 0.0
    expected_shares: float = 0.0
    expected_cost_usdc: float = 0.0
    expected_fee_usdc: float = 0.0
    expected_slippage_usdc: float = 0.0
    partial: bool = False

    # --- provenance ------------------------------------------------------ #
    model_name: str = ""
    bankroll_before_usdc: float = 0.0
    clock_status: str = "unknown"
    clock_offset_ms: float = 0.0
    decision_latency_ms: float = 0.0
    record_version: str = RECORD_VERSION

    def as_record(self) -> dict[str, Any]:
        payload = {"record": "decision", **self.__dict__}
        return payload


@dataclass
class PaperSettlement:
    """What actually happened, written when Polymarket resolves the market."""

    condition_id: str
    slug: str
    settled_at_ms: int
    settlement_time_ms: int
    official_outcome: str | None
    status: str = TradeStatus.REJECTED.value
    pnl_usdc: float = 0.0
    payoff_usdc: float = 0.0
    cost_usdc: float = 0.0
    bankroll_after_usdc: float = 0.0
    won: bool = False

    # --- execution quality, measured from the live tape ------------------ #
    #: Share of post-decision trades that occurred at or through our price.
    fill_probability: float | None = None
    #: Best price actually printed on our side after the decision.
    observed_best_price: float | None = None
    #: Realised minus expected entry price, in probability points. Positive
    #: means the simulation was optimistic.
    price_error: float | None = None
    tape_trades_observed: int = 0
    record_version: str = RECORD_VERSION

    def as_record(self) -> dict[str, Any]:
        return {"record": "settlement", **self.__dict__}


def decision_from_outcome(
    outcome: PipelineOutcome,
    *,
    condition_id: str,
    slug: str,
    settlement_time_ms: int,
    horizon_seconds: int,
    decided_at_ms: int,
    quote: Any,
    mid_price: float,
    touch_price: float,
    bankroll_usdc: float,
    model_name: str = "",
    clock_status: str = "unknown",
    clock_offset_ms: float = 0.0,
    decision_latency_ms: float = 0.0,
) -> PaperDecision:
    """Flatten a pipeline outcome into a journal record.

    Kept as a function rather than a method on ``PipelineOutcome`` so the
    trading layer stays free of any knowledge of how paper trading stores
    things — the same pipeline serves the backtest, which stores nothing.
    """
    decision = outcome.decision
    fill = outcome.fill
    stake = outcome.stake
    return PaperDecision(
        condition_id=condition_id,
        slug=slug,
        settlement_time_ms=settlement_time_ms,
        horizon_seconds=horizon_seconds,
        decided_at_ms=decided_at_ms,
        stage=outcome.stage.value,
        traded=outcome.traded,
        fill_attempted=outcome.fill_attempted,
        skip_reason=outcome.skip_reason.value if outcome.skip_reason else None,
        outcome=decision.outcome.value if decision.outcome else None,
        model_prob_up=decision.model_prob_up,
        adjusted_prob_up=decision.adjusted_prob_up,
        market_prob_up=decision.market_prob_up,
        disagreement_logits=decision.disagreement_logits,
        confidence=decision.confidence,
        edge=decision.edge,
        ev_per_usdc=decision.ev_per_usdc,
        best_bid=quote.best_bid,
        best_ask=quote.best_ask,
        bid_depth_usdc=quote.bid_depth_usdc,
        ask_depth_usdc=quote.ask_depth_usdc,
        quoted_spread=quote.spread or 0.0,
        book_age_ms=quote.age_ms,
        mid_price=mid_price,
        touch_price=touch_price,
        desired_stake_usdc=stake.usdc if stake and stake.accepted else 0.0,
        expected_price=fill.price if fill else decision.entry_price,
        expected_shares=fill.shares if fill else 0.0,
        expected_cost_usdc=fill.cost_usdc if fill else 0.0,
        expected_fee_usdc=fill.fee_usdc if fill else 0.0,
        expected_slippage_usdc=fill.slippage_usdc if fill else 0.0,
        partial=fill.partial if fill else False,
        model_name=model_name,
        bankroll_before_usdc=bankroll_usdc,
        clock_status=clock_status,
        clock_offset_ms=clock_offset_ms,
        decision_latency_ms=decision_latency_ms,
    )


@dataclass
class PaperRun:
    """A decision joined to its settlement, if one has arrived."""

    decision: PaperDecision
    settlement: PaperSettlement | None = None

    @property
    def resolved(self) -> bool:
        return self.settlement is not None

    @property
    def pnl_usdc(self) -> float:
        return self.settlement.pnl_usdc if self.settlement else 0.0

    @property
    def label(self) -> int | None:
        if self.settlement is None or self.settlement.official_outcome is None:
            return None
        return 1 if self.settlement.official_outcome == Outcome.UP.value else 0


class PaperJournal:
    """Append-only store for paper decisions and their settlements."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._decided: set[str] = set()
        self._settled: set[str] = set()
        self._load_keys()

    def _load_keys(self) -> None:
        for record in self.read_records():
            key = str(record.get("condition_id", ""))
            if record.get("record") == "decision":
                self._decided.add(self._key(key, int(record.get("horizon_seconds", 0))))
            elif record.get("record") == "settlement":
                self._settled.add(key)

    @staticmethod
    def _key(condition_id: str, horizon_seconds: int) -> str:
        return f"{condition_id}@{horizon_seconds}"

    # ------------------------------------------------------------------ #
    def has_decision(self, condition_id: str, horizon_seconds: int) -> bool:
        """One decision per market per horizon; a re-evaluation is a bug."""
        return self._key(condition_id, horizon_seconds) in self._decided

    def has_settlement(self, condition_id: str) -> bool:
        return condition_id in self._settled

    def record_decision(self, decision: PaperDecision) -> bool:
        key = self._key(decision.condition_id, decision.horizon_seconds)
        if key in self._decided:
            log.warning(
                "paper.duplicate_decision", slug=decision.slug, horizon=decision.horizon_seconds
            )
            return False
        self._append(decision.as_record())
        self._decided.add(key)
        return True

    def record_settlement(self, settlement: PaperSettlement) -> bool:
        if settlement.condition_id in self._settled:
            return False
        self._append(settlement.as_record())
        self._settled.add(settlement.condition_id)
        return True

    def _append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    # ------------------------------------------------------------------ #
    def read_records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line is expected while the file is being
                    # appended to; skip it rather than losing the journal.
                    log.debug("paper.torn_record")

    def runs(self) -> list[PaperRun]:
        """Every decision, joined to its settlement, in chronological order."""
        decisions: list[PaperDecision] = []
        settlements: dict[str, PaperSettlement] = {}
        for record in self.read_records():
            kind = record.pop("record", "")
            record.pop("record_version", None)
            if kind == "decision":
                decisions.append(PaperDecision(**record))
            elif kind == "settlement":
                settlement = PaperSettlement(**record)
                settlements[settlement.condition_id] = settlement
        decisions.sort(key=lambda d: (d.decided_at_ms, d.condition_id))
        return [PaperRun(d, settlements.get(d.condition_id)) for d in decisions]

    def pending(self) -> list[PaperDecision]:
        """Decisions whose market has not been settled in the journal yet."""
        return [run.decision for run in self.runs() if not run.resolved]

    def summary(self) -> dict[str, int]:
        runs = self.runs()
        return {
            "decisions": len(runs),
            "traded": sum(1 for r in runs if r.decision.traded),
            "skipped": sum(1 for r in runs if not r.decision.traded),
            "settled": sum(1 for r in runs if r.resolved),
        }


def open_journal(config: Config) -> PaperJournal:
    return PaperJournal(config.resolved_path(config.app.log_dir) / "paper.jsonl")


def skip_histogram(runs: list[PaperRun]) -> dict[str, int]:
    """Named refusals, most common first — the diagnostic that matters most."""
    counts: dict[str, int] = {}
    for run in runs:
        reason = run.decision.skip_reason
        if run.decision.traded or not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
