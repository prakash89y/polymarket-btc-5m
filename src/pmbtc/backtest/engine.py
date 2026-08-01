"""The simulator.

Replays settled windows in the order they happened, asks the same decision
layer the live bot will ask, fills against the book that was actually recorded,
and keeps a bankroll. One market, one pass, no revisiting.

**Where a backtest normally cheats, and why this one cannot:**

*Look-ahead through the row.* A decision taken at T-180 may only read the
snapshot captured at T-180. The engine hands the decision layer exactly one row
and never the market's later rows, so a leak would have to be a leak in the
dataset itself — which Module 4's guard already refuses at write time.

*Look-ahead through ordering.* Markets are processed in settlement order, and
the bankroll, the risk ledger and the streak counters carry forward. A strategy
cannot be sized using a bankroll it had not yet earned.

*Survivorship.* Skipped windows are recorded, not dropped. A run that trades 4
of 600 windows must show 596 named refusals, because "we only counted the ones
we traded" is how a 4-trade sample becomes a strategy.

*The label.* Used for one purpose only — settling a position that was already
opened. It is never visible to the decision, the sizing, or the fill.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pmbtc.backtest.fills import FillModel
from pmbtc.config import Config
from pmbtc.constants import Outcome, SkipReason, TradeStatus
from pmbtc.logging_setup import get_logger
from pmbtc.trading.costs import CostModel, Fill, Quote
from pmbtc.trading.decision import Decision, DecisionEngine
from pmbtc.trading.risk import RiskLedger
from pmbtc.trading.sizing import PositionSizer, Stake

log = get_logger("pmbtc.backtest.engine")

Row = Mapping[str, Any]

#: A model is any callable from one snapshot row to P(settles UP). Keeping it
#: this loose is what lets the same engine score a baseline, a trained
#: estimator, or a fixed rule without any of them knowing about each other.
ProbabilityModel = Callable[[Row], float]


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Which dataset columns carry the book. Named once, in one place."""

    best_bid: str = "f_ob_best_bid"
    best_ask: str = "f_ob_best_ask"
    bid_depth: str = "f_lq_depth_bid_usdc"
    ask_depth: str = "f_lq_depth_ask_usdc"
    #: Per-feature capture latency, written by Module 4's quality layer.
    bid_latency: str = "lat_ob_best_bid"
    ask_latency: str = "lat_ob_best_ask"
    #: Optional volatility circuit-breaker input. No column, no gate — the
    #: dataset carries no realised-vol z-score today, and inventing one from a
    #: feature that means something else would be worse than leaving it off.
    vol_zscore: str | None = None


def _f(row: Row, key: str | None) -> float | None:
    if key is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN check without numpy


def quote_from_row(row: Row, columns: ColumnMap = ColumnMap()) -> Quote:
    """Reconstruct the book as it stood at the snapshot instant."""
    latencies = [
        _f(row, columns.bid_latency) or 0.0,
        _f(row, columns.ask_latency) or 0.0,
    ]
    return Quote(
        best_bid=_f(row, columns.best_bid),
        best_ask=_f(row, columns.best_ask),
        bid_depth_usdc=_f(row, columns.bid_depth) or 0.0,
        ask_depth_usdc=_f(row, columns.ask_depth) or 0.0,
        age_ms=int(max(latencies)),
    )


@dataclass(frozen=True, slots=True)
class WindowResult:
    """One evaluated window — traded or not."""

    condition_id: str
    slug: str
    settlement_time_ms: int
    horizon_seconds: int
    decision: Decision
    settled: Outcome | None
    status: TradeStatus
    stake: Stake | None = None
    fill: Fill | None = None
    pnl_usdc: float = 0.0
    bankroll_after_usdc: float = 0.0
    skip_reason: SkipReason | None = None

    @property
    def traded(self) -> bool:
        return self.fill is not None

    @property
    def won(self) -> bool:
        if self.fill is None or self.settled is None:
            return False
        return self.settled is self.fill.outcome

    @property
    def prob_up(self) -> float:
        return self.decision.model_prob_up

    @property
    def label(self) -> int | None:
        if self.settled is None:
            return None
        return 1 if self.settled is Outcome.UP else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "settlement_time_ms": self.settlement_time_ms,
            "horizon_seconds": self.horizon_seconds,
            "status": self.status.value,
            "settled": self.settled.value if self.settled else None,
            "traded": self.traded,
            "pnl_usdc": round(self.pnl_usdc, 6),
            "bankroll_after_usdc": round(self.bankroll_after_usdc, 6),
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
            "decision": self.decision.as_dict(),
            "stake": self.stake.as_dict() if self.stake else None,
            "fill": (
                {
                    "outcome": self.fill.outcome.value,
                    "price": round(self.fill.price, 6),
                    "shares": round(self.fill.shares, 6),
                    "cost_usdc": round(self.fill.cost_usdc, 6),
                    "partial": self.fill.partial,
                }
                if self.fill
                else None
            ),
        }


@dataclass
class BacktestResult:
    """Every window the engine looked at, in order, plus how it ended."""

    windows: list[WindowResult] = field(default_factory=list)
    starting_bankroll_usdc: float = 0.0
    ending_bankroll_usdc: float = 0.0
    ledger: RiskLedger | None = None
    model_name: str = ""
    decision_horizon_seconds: int = 0
    fill_style: str = "touch"

    @property
    def trades(self) -> list[WindowResult]:
        return [w for w in self.windows if w.traded]

    @property
    def evaluated(self) -> int:
        return len(self.windows)

    def skip_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for window in self.windows:
            if window.traded or window.skip_reason is None:
                continue
            counts[window.skip_reason.value] = counts.get(window.skip_reason.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


class BacktestEngine:
    """Runs one strategy over one set of settled windows."""

    def __init__(
        self,
        config: Config,
        *,
        fill_model: FillModel | None = None,
        columns: ColumnMap | None = None,
        honour_kill_switch: bool = False,
    ) -> None:
        self.config = config
        self.costs = CostModel(config.costs)
        self.decisions = DecisionEngine(config, self.costs)
        self.sizer = PositionSizer(config)
        self.columns = columns or ColumnMap()
        self.fills = fill_model or FillModel(
            self.costs, pessimistic=config.backtest.pessimistic_fill
        )
        # A backtest has no operator, so the kill switch is off by default:
        # a stale file on a dev box must not silently empty a historical run.
        self.honour_kill_switch = honour_kill_switch

    # ------------------------------------------------------------------ #
    def run(
        self,
        rows: Iterable[Row],
        model: ProbabilityModel,
        *,
        model_name: str = "model",
        decision_horizon_seconds: int | None = None,
        starting_bankroll_usdc: float | None = None,
        calibration_error: float | None = None,
    ) -> BacktestResult:
        """Replay ``rows`` chronologically.

        ``decision_horizon_seconds`` pins which snapshot the decision is taken
        from. Left unset, the engine uses the latest horizon that clears the
        configured timing gates — the most informed moment we are still allowed
        to trade at.
        """
        bankroll = (
            starting_bankroll_usdc
            if starting_bankroll_usdc is not None
            else self.config.backtest.initial_bankroll_usdc
        )
        ledger = RiskLedger.from_config(
            self.config,
            starting_bankroll_usdc=bankroll,
            honour_kill_switch=self.honour_kill_switch,
        )
        by_market = self._group(rows)
        horizon = decision_horizon_seconds or self._default_horizon(by_market)

        result = BacktestResult(
            starting_bankroll_usdc=bankroll,
            ledger=ledger,
            model_name=model_name,
            decision_horizon_seconds=horizon,
            fill_style=self.fills.style,
        )

        for _, market_rows in by_market:
            row = self._row_at(market_rows, horizon)
            if row is None:
                continue
            result.windows.append(
                self._evaluate(row, model, ledger, calibration_error=calibration_error)
            )

        result.ending_bankroll_usdc = ledger.state.bankroll_usdc
        log.info(
            "backtest.complete",
            model=model_name,
            evaluated=result.evaluated,
            trades=len(result.trades),
            horizon=horizon,
            start_bankroll=round(bankroll, 2),
            end_bankroll=round(result.ending_bankroll_usdc, 2),
        )
        return result

    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        row: Row,
        model: ProbabilityModel,
        ledger: RiskLedger,
        *,
        calibration_error: float | None,
    ) -> WindowResult:
        horizon = int(row["horizon_seconds"])
        settlement_ms = int(row["settlement_time_ms"])
        decision_ms = settlement_ms - horizon * 1000
        quote = quote_from_row(row, self.columns)
        settled = self._settled_outcome(row)

        def result(
            decision: Decision,
            *,
            status: TradeStatus,
            skip: SkipReason | None,
            stake: Stake | None = None,
            fill: Fill | None = None,
            pnl: float = 0.0,
        ) -> WindowResult:
            return WindowResult(
                condition_id=str(row.get("condition_id", "")),
                slug=str(row.get("slug", "")),
                settlement_time_ms=settlement_ms,
                horizon_seconds=horizon,
                decision=decision,
                settled=settled,
                status=status,
                stake=stake,
                fill=fill,
                pnl_usdc=pnl,
                bankroll_after_usdc=ledger.state.bankroll_usdc,
                skip_reason=skip,
            )

        decision = self.decisions.decide(
            model_prob_up=model(row),
            quote=quote,
            seconds_into_window=self.config.app.window_seconds - horizon,
            seconds_to_settlement=horizon,
            vol_zscore=_f(row, self.columns.vol_zscore),
            calibration_error=calibration_error,
        )
        if not decision.trade or decision.outcome is None:
            return result(decision, status=TradeStatus.REJECTED, skip=decision.skip_reason)

        stake = self.sizer.size(
            bankroll_usdc=ledger.state.bankroll_usdc,
            model_prob=decision.confidence,
            entry_price=decision.entry_price,
            calibration_error=calibration_error,
        )
        if not stake.accepted:
            return result(
                decision, status=TradeStatus.REJECTED, skip=stake.skip_reason, stake=stake
            )

        verdict = ledger.check(now_ms=decision_ms, stake_usdc=stake.usdc)
        if not verdict.allowed:
            return result(
                decision, status=TradeStatus.REJECTED, skip=verdict.reason, stake=stake
            )

        filled = self.fills.execute(
            outcome=decision.outcome, quote=quote, stake_usdc=stake.usdc
        )
        if filled.fill is None:
            return result(
                decision, status=TradeStatus.REJECTED, skip=filled.skip_reason, stake=stake
            )

        fill = filled.fill
        ledger.register_open(cost_usdc=fill.cost_usdc)
        if settled is None:
            # Should not happen on a labelled dataset, but an unsettled market
            # must return the stake rather than silently count as a loss.
            ledger.register_close(
                now_ms=settlement_ms, cost_usdc=fill.cost_usdc, payoff_usdc=fill.cost_usdc
            )
            return result(
                decision, status=TradeStatus.VOIDED, skip=None, stake=stake, fill=fill
            )

        # Redemption gas is charged inside the payoff so the ledger's bankroll
        # and the reported P&L can never disagree about what a trade cost.
        payoff = fill.payoff_usdc(settled)
        net_payoff = payoff - (self.config.costs.gas_cost_usdc if payoff > 0 else 0.0)
        pnl = ledger.register_close(
            now_ms=settlement_ms, cost_usdc=fill.cost_usdc, payoff_usdc=net_payoff
        )
        status = TradeStatus.SETTLED_WIN if payoff > 0 else TradeStatus.SETTLED_LOSS
        return result(decision, status=status, skip=None, stake=stake, fill=fill, pnl=pnl)

    # ------------------------------------------------------------------ #
    # Row plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _group(rows: Iterable[Row]) -> list[tuple[str, list[Row]]]:
        """Markets in settlement order; rows inside a market by countdown.

        Sorting explicitly rather than trusting the caller is what keeps the
        run deterministic regardless of how the rows were loaded.
        """
        grouped: dict[str, list[Row]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("condition_id", "")), []).append(row)
        ordered = sorted(
            grouped.items(),
            key=lambda kv: (int(kv[1][0]["settlement_time_ms"]), kv[0]),
        )
        return [
            (cid, sorted(market_rows, key=lambda r: -int(r["horizon_seconds"])))
            for cid, market_rows in ordered
        ]

    @staticmethod
    def _row_at(market_rows: Sequence[Row], horizon_seconds: int) -> Row | None:
        for row in market_rows:
            if int(row["horizon_seconds"]) == horizon_seconds:
                return row
        return None

    def _default_horizon(self, by_market: Sequence[tuple[str, list[Row]]]) -> int:
        """The last snapshot the timing gates still permit a trade at.

        Later is better — the forecast is fresher and the book is tighter — but
        only down to ``execution.min_seconds_to_settlement``. Anything past that
        is late-window territory, which requires an exceptional edge and should
        be opted into deliberately rather than picked as a default.
        """
        horizons = sorted(
            {int(r["horizon_seconds"]) for _, rows in by_market for r in rows}, reverse=True
        )
        window = self.config.app.window_seconds
        execution = self.config.execution
        allowed = [
            h
            for h in horizons
            if window - h >= execution.min_seconds_into_window
            and h >= execution.min_seconds_to_settlement
        ]
        if allowed:
            return min(allowed)
        return horizons[-1] if horizons else 0

    @staticmethod
    def _settled_outcome(row: Row) -> Outcome | None:
        official = row.get("official_outcome")
        if isinstance(official, Outcome):
            return official
        if isinstance(official, str) and official:
            try:
                return Outcome(official)
            except ValueError:
                return None
        label = row.get("label")
        if label is None:
            return None
        return Outcome.UP if int(label) == 1 else Outcome.DOWN
