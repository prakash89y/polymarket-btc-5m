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
    #: The same three for the market's implied probability — the benchmark that
    #: actually matters, on exactly the windows the model was asked about.
    market_brier: float = float("nan")
    market_log_loss: float = float("nan")

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
            "market_brier": r(self.market_brier),
            "market_log_loss": r(self.market_log_loss),
            "brier_skill": r(self.brier_skill),
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


def _brier(pairs: list[tuple[int, float]]) -> float:
    if not pairs:
        return float("nan")
    return sum((p - y) ** 2 for y, p in pairs) / len(pairs)


def _log_loss(pairs: list[tuple[int, float]]) -> float:
    if not pairs:
        return float("nan")
    total = 0.0
    for y, p in pairs:
        clipped = min(1.0 - 1e-15, max(1e-15, p))
        total -= math.log(clipped) if y == 1 else math.log(1.0 - clipped)
    return total / len(pairs)


def _accuracy(pairs: list[tuple[int, float]]) -> float:
    if not pairs:
        return float("nan")
    return sum(1 for y, p in pairs if (p >= 0.5) == (y == 1)) / len(pairs)


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
    metrics.brier = _brier(model_pairs)
    metrics.log_loss = _log_loss(model_pairs)
    metrics.accuracy = _accuracy(model_pairs)
    metrics.market_brier = _brier(market_pairs)
    metrics.market_log_loss = _log_loss(market_pairs)

    times = [w.settlement_time_ms for w in windows]
    span_ms = max(times) - min(times)
    metrics.span_days = span_ms / 86_400_000.0

    # --- money, over the traded subset --------------------------------- #
    trades = [w for w in windows if w.traded and w.fill is not None]
    metrics.trades = len(trades)
    metrics.trade_rate = len(trades) / len(windows)
    if metrics.span_days > 0:
        metrics.trades_per_day = len(trades) / metrics.span_days
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
    metrics.wins = metrics.wins
    metrics.win_rate = metrics.wins / n
    metrics.avg_entry_price /= n
    metrics.avg_edge /= n
    # Break-even is the average price paid, not 0.5: a contract bought at 0.62
    # must win 62% of the time to return the stake.
    metrics.breakeven_win_rate = metrics.avg_entry_price
    metrics.max_drawdown = max_drawdown(equity)

    periods_per_year = (
        n / (span_ms / MS_PER_YEAR) if span_ms > 0 else float(n)
    )
    metrics.sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    return metrics
