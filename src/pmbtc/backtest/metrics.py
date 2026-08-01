"""Performance metrics for a completed run.

Two families, deliberately reported side by side:

*Forecast quality* — Brier, log loss, accuracy — computed over **every**
evaluated window, including the ones we refused to trade. This is what Modules 6
and 7 already optimise.

*Money* — ROI, profit factor, Sharpe, drawdown — computed over the traded subset
only, because that is the only subset where money moved.

They can disagree, and when they do the disagreement is the finding. A model
with an excellent Brier score that trades at bad prices loses; a mediocre model
that abstains except when the book is mispriced wins. Reporting only the first
family is how a system arrives at a beautifully calibrated losing strategy.

Sharpe needs a stated convention. Returns here are per-trade returns on capital
at risk, annualised by the strategy's *realised* trading frequency over the span
covered by the run. A strategy that trades four times a week does not get to
claim the annualisation factor of one that trades hourly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pmbtc.models.calibration import CalibrationCurve, EvaluationMetrics, evaluate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pmbtc.backtest.engine import BacktestResult, WindowResult

MS_PER_YEAR = 365.25 * 24 * 60 * 60 * 1000.0
_EPS = 1e-12


@dataclass
class BacktestMetrics:
    """Everything the deployment gate and the operator need to see."""

    # Coverage
    evaluated_windows: int = 0
    trades: int = 0
    trade_rate: float = 0.0
    span_days: float = 0.0
    trades_per_day: float = 0.0

    # Money
    starting_bankroll_usdc: float = 0.0
    ending_bankroll_usdc: float = 0.0
    net_pnl_usdc: float = 0.0
    roi: float = 0.0
    gross_profit_usdc: float = 0.0
    gross_loss_usdc: float = 0.0
    profit_factor: float = 0.0
    total_staked_usdc: float = 0.0
    return_on_stake: float = 0.0
    avg_pnl_per_trade_usdc: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0

    # Trading quality
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_entry_price: float = 0.0
    #: Win rate a strategy needs just to break even at its average entry price.
    breakeven_win_rate: float = 0.0
    avg_edge: float = 0.0
    total_fees_usdc: float = 0.0
    total_slippage_usdc: float = 0.0
    partial_fills: int = 0

    # Forecast quality, over every evaluated window
    brier: float = float("nan")
    log_loss: float = float("nan")
    accuracy: float = float("nan")
    #: Expected calibration error and its worst bin — "when it says 0.9, how
    #: often is it right". Sized positions make this a money question, not a
    #: cosmetic one: Kelly is convex in the probability.
    ece: float = float("nan")
    mce: float = float("nan")
    #: The same for the market's implied probability — the benchmark that
    #: actually matters, on exactly the windows the model was asked about.
    market_brier: float = float("nan")
    market_log_loss: float = float("nan")
    market_ece: float = float("nan")
    #: Reliability diagram, kept whole so the report can render it.
    reliability: CalibrationCurve | None = None

    # Execution quality
    fill_attempts: int = 0
    fill_rate: float = 0.0
    avg_holding_seconds: float = 0.0
    #: Mean quoted spread on the windows we actually traded.
    avg_quoted_spread: float = 0.0
    #: Mean half-spread crossed per share, i.e. touch minus mid.
    avg_spread_paid: float = 0.0
    #: Mean adverse fill beyond the touch, per share.
    avg_slippage: float = 0.0

    equity_curve: list[float] = field(default_factory=list)
    skips: dict[str, int] = field(default_factory=dict)

    @property
    def brier_skill(self) -> float:
        """Fraction of the market's Brier score removed. Negative is worse."""
        if math.isnan(self.brier) or math.isnan(self.market_brier) or self.market_brier <= 0:
            return float("nan")
        return 1.0 - self.brier / self.market_brier

    def as_dict(self) -> dict[str, Any]:
        def r(value: float, places: int = 6) -> float:
            return value if math.isnan(value) else round(value, places)

        return {
            "evaluated_windows": self.evaluated_windows,
            "trades": self.trades,
            "trade_rate": r(self.trade_rate),
            "span_days": r(self.span_days, 3),
            "trades_per_day": r(self.trades_per_day, 3),
            "starting_bankroll_usdc": r(self.starting_bankroll_usdc, 2),
            "ending_bankroll_usdc": r(self.ending_bankroll_usdc, 2),
            "net_pnl_usdc": r(self.net_pnl_usdc, 4),
            "roi": r(self.roi),
            "profit_factor": r(self.profit_factor),
            "return_on_stake": r(self.return_on_stake),
            "avg_pnl_per_trade_usdc": r(self.avg_pnl_per_trade_usdc, 4),
            "max_drawdown": r(self.max_drawdown),
            "sharpe": r(self.sharpe, 4),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": r(self.win_rate),
            "avg_entry_price": r(self.avg_entry_price),
            "breakeven_win_rate": r(self.breakeven_win_rate),
            "avg_edge": r(self.avg_edge),
            "total_staked_usdc": r(self.total_staked_usdc, 2),
            "total_fees_usdc": r(self.total_fees_usdc, 4),
            "total_slippage_usdc": r(self.total_slippage_usdc, 4),
            "partial_fills": self.partial_fills,
            "brier": r(self.brier),
            "log_loss": r(self.log_loss),
            "accuracy": r(self.accuracy),
            "ece": r(self.ece),
            "mce": r(self.mce),
            "market_brier": r(self.market_brier),
            "market_log_loss": r(self.market_log_loss),
            "market_ece": r(self.market_ece),
            "brier_skill": r(self.brier_skill),
            "reliability": self.reliability.as_dict() if self.reliability else None,
            "fill_attempts": self.fill_attempts,
            "fill_rate": r(self.fill_rate),
            "avg_holding_seconds": r(self.avg_holding_seconds, 2),
            "avg_quoted_spread": r(self.avg_quoted_spread),
            "avg_spread_paid": r(self.avg_spread_paid),
            "avg_slippage": r(self.avg_slippage),
            "skips": self.skips,
        }

    def __str__(self) -> str:
        if self.trades == 0:
            return (
                f"{self.evaluated_windows} windows evaluated, 0 trades "
                f"(brier={self.brier:.5f} vs market {self.market_brier:.5f})"
            )
        return (
            f"{self.trades} trades / {self.evaluated_windows} windows "
            f"({self.trade_rate:.1%}) | pnl {self.net_pnl_usdc:+.2f} USDC "
            f"roi {self.roi:+.2%} | pf {self.profit_factor:.2f} "
            f"sharpe {self.sharpe:.2f} dd {self.max_drawdown:.2%} | "
            f"win {self.win_rate:.1%} vs breakeven {self.breakeven_win_rate:.1%} | "
            f"brier {self.brier:.5f} vs market {self.market_brier:.5f}"
        )


def _evaluate(pairs: list[tuple[int, float]]) -> EvaluationMetrics | None:
    """Score (label, probability) pairs with Module 7's metric implementation.

    Deliberately delegated rather than reimplemented: ``pmbtc.models.calibration``
    already defines Brier, log loss, accuracy, ECE, MCE and the reliability
    curve, and they are what the promotion gate uses. A second implementation
    here would eventually disagree with it, and the disagreement would surface
    as a model that passes one gate and fails the other for no visible reason.
    """
    if not pairs:
        return None
    import numpy as np

    y = np.array([label for label, _ in pairs], dtype=float)
    p = np.array([prob for _, prob in pairs], dtype=float)
    return evaluate(y, p)


def max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough fall in the equity curve, as a fraction."""
    peak, worst = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, 1.0 - value / peak)
    return worst


def sharpe_ratio(returns: list[float], *, periods_per_year: float) -> float:
    """Annualised Sharpe of per-trade returns. Zero variance means zero Sharpe.

    No risk-free rate is subtracted: capital is committed for five minutes at a
    time, so the carry is immaterial and pretending otherwise would only add a
    parameter nobody would tune correctly.
    """
    n = len(returns)
    if n < 2 or periods_per_year <= 0:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if variance <= _EPS:
        return 0.0
    return (mean / math.sqrt(variance)) * math.sqrt(periods_per_year)


def compute_metrics(result: BacktestResult) -> BacktestMetrics:
    """Summarise a completed run. Pure function of the run's windows."""
    windows: list[WindowResult] = list(result.windows)
    metrics = BacktestMetrics(
        evaluated_windows=len(windows),
        starting_bankroll_usdc=result.starting_bankroll_usdc,
        ending_bankroll_usdc=result.ending_bankroll_usdc,
        skips=result.skip_histogram(),
    )
    if not windows:
        return metrics

    # --- forecast quality, over everything we looked at ---------------- #
    model_pairs = [(w.label, w.prob_up) for w in windows if w.label is not None]
    market_pairs = [
        (w.label, w.decision.market_prob_up) for w in windows if w.label is not None
    ]
    model_eval = _evaluate(model_pairs)
    market_eval = _evaluate(market_pairs)
    if model_eval is not None:
        metrics.brier = model_eval.brier
        metrics.log_loss = model_eval.log_loss
        metrics.accuracy = model_eval.accuracy
        metrics.ece = model_eval.ece
        metrics.mce = model_eval.mce
        metrics.reliability = model_eval.curve
    if market_eval is not None:
        metrics.market_brier = market_eval.brier
        metrics.market_log_loss = market_eval.log_loss
        metrics.market_ece = market_eval.ece

    times = [w.settlement_time_ms for w in windows]
    span_ms = max(times) - min(times)
    metrics.span_days = span_ms / 86_400_000.0

    # --- money, over the traded subset --------------------------------- #
    trades = [w for w in windows if w.traded and w.fill is not None]
    metrics.trades = len(trades)
    metrics.trade_rate = len(trades) / len(windows)
    if metrics.span_days > 0:
        metrics.trades_per_day = len(trades) / metrics.span_days
    # Fill rate is measured against intents that actually reached the book, so
    # a trade the risk engine refused is not counted as a fill failure.
    metrics.fill_attempts = sum(1 for w in windows if w.fill_attempted)
    if metrics.fill_attempts:
        metrics.fill_rate = len(trades) / metrics.fill_attempts
    if not trades:
        metrics.equity_curve = [result.starting_bankroll_usdc]
        return metrics

    equity = [result.starting_bankroll_usdc]
    returns: list[float] = []
    for window in trades:
        fill = window.fill
        assert fill is not None
        equity.append(window.bankroll_after_usdc)
        metrics.total_staked_usdc += fill.cost_usdc
        metrics.total_fees_usdc += fill.fee_usdc
        metrics.total_slippage_usdc += fill.slippage_usdc
        metrics.avg_entry_price += fill.price
        metrics.avg_edge += window.decision.edge
        metrics.partial_fills += 1 if fill.partial else 0
        metrics.avg_holding_seconds += window.holding_seconds
        metrics.avg_quoted_spread += window.quoted_spread
        # Per share: what crossing to the touch cost, and what was paid beyond it.
        metrics.avg_spread_paid += max(0.0, window.touch_price - window.mid_price)
        metrics.avg_slippage += max(0.0, fill.price - window.touch_price)
        if window.pnl_usdc > 0:
            metrics.wins += 1
            metrics.gross_profit_usdc += window.pnl_usdc
        else:
            metrics.losses += 1
            metrics.gross_loss_usdc += -window.pnl_usdc
        returns.append(window.pnl_usdc / fill.cost_usdc if fill.cost_usdc > 0 else 0.0)

    n = len(trades)
    metrics.equity_curve = equity
    metrics.net_pnl_usdc = result.ending_bankroll_usdc - result.starting_bankroll_usdc
    metrics.roi = (
        metrics.net_pnl_usdc / result.starting_bankroll_usdc
        if result.starting_bankroll_usdc > 0
        else 0.0
    )
    metrics.avg_pnl_per_trade_usdc = metrics.net_pnl_usdc / n
    metrics.return_on_stake = (
        metrics.net_pnl_usdc / metrics.total_staked_usdc
        if metrics.total_staked_usdc > 0
        else 0.0
    )
    # An infinite profit factor is not a good result, it is a small sample with
    # no losses yet. Report it as the gross profit so it stays comparable.
    metrics.profit_factor = (
        metrics.gross_profit_usdc / metrics.gross_loss_usdc
        if metrics.gross_loss_usdc > _EPS
        else metrics.gross_profit_usdc
    )
    metrics.win_rate = metrics.wins / n
    for attribute in (
        "avg_entry_price",
        "avg_edge",
        "avg_holding_seconds",
        "avg_quoted_spread",
        "avg_spread_paid",
        "avg_slippage",
    ):
        setattr(metrics, attribute, getattr(metrics, attribute) / n)
    # Break-even is the average price paid, not 0.5: a contract bought at 0.62
    # must win 62% of the time to return the stake.
    metrics.breakeven_win_rate = metrics.avg_entry_price
    metrics.max_drawdown = max_drawdown(equity)

    periods_per_year = (
        n / (span_ms / MS_PER_YEAR) if span_ms > 0 else float(n)
    )
    metrics.sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    return metrics
