"""Edge decomposition: where the money actually came from, and where it went.

A backtest that reports "ROI +3.2%" has answered the least useful question. The
useful ones are: how much edge did the forecast actually contain, how much of it
did the spread take, how much did slippage and fees take, how much did we never
collect because the book was too thin, and how much did the risk limits cost or
save. On this instrument those terms are the same order of magnitude as the edge
itself, so a strategy can be profitable in forecast and unprofitable in fact.

**How the ladder is built.** Six counterfactual worlds, in the order the runtime
applies them, each one changing exactly one thing:

===  ==================================================  ====================
P0   every intent, full desired size, filled at the mid   the raw forecast edge
P1   ...but crossing the spread to the touch              - spread cost
P2   ...and paying modelled slippage beyond the touch     - slippage
P3   ...and paying fees and redemption gas                - fees
P4   ...minus the intents the risk engine refused         - risk limits
P5   ...minus what the book's depth would not fill        - missed fills
===  ==================================================  ====================

``P5`` is the realised P&L. Because every line is the difference of two
adjacent worlds, the identity

    raw_edge - spread - slippage - fees - risk - missed_fills == net_profit

holds **exactly**, not approximately, and :meth:`EdgeDecomposition.reconciles`
asserts it against the engine's own bankroll. A decomposition that does not
reconcile is a bug in the decomposition, and it will say so rather than quietly
mis-attribute.

**Signs.** A "cost" line can be negative, and that is information rather than an
error. Risk limits that block a losing trade *save* money and show as a negative
cost; depth that prevented a bad fill does the same. Reporting only the cases
where a constraint hurt would make the risk engine look like pure overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pmbtc.config import CostConfig
from pmbtc.logging_setup import get_logger
from pmbtc.trading.costs import CostModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pmbtc.backtest.engine import BacktestResult, WindowResult

log = get_logger("pmbtc.backtest.attribution")

#: Tolerance for the reconciliation check, in USDC. Floating-point noise over a
#: few thousand trades, not a licence for the attribution to be approximately
#: right.
RECONCILE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class EdgeLine:
    """One rung of the ladder."""

    name: str
    usdc: float
    #: The same quantity per USDC actually staked, so lines stay comparable
    #: across runs of different size.
    per_usdc_staked: float
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "usdc": round(self.usdc, 6),
            "per_usdc_staked": round(self.per_usdc_staked, 6),
            "detail": self.detail,
        }


@dataclass
class EdgeDecomposition:
    """The full waterfall from raw forecast edge to realised profit."""

    raw_edge_usdc: float = 0.0
    spread_cost_usdc: float = 0.0
    slippage_cost_usdc: float = 0.0
    fee_cost_usdc: float = 0.0
    risk_limit_usdc: float = 0.0
    missed_fill_usdc: float = 0.0
    net_profit_usdc: float = 0.0

    #: Denominator for the per-USDC columns: what was actually staked.
    staked_usdc: float = 0.0
    #: Denominator for the intent-side lines: what we wanted to stake.
    intended_usdc: float = 0.0

    intents: int = 0
    filled: int = 0
    blocked_by_risk: int = 0
    blocked_by_depth: int = 0
    partial_fills: int = 0

    reconciliation_error_usdc: float = 0.0

    def _per(self, usdc: float) -> float:
        return usdc / self.staked_usdc if self.staked_usdc > 0 else 0.0

    @property
    def lines(self) -> list[EdgeLine]:
        """The waterfall, in the order the runtime applies it."""
        return [
            EdgeLine(
                "raw_edge",
                self.raw_edge_usdc,
                self._per(self.raw_edge_usdc),
                f"{self.intents} intent(s) filled at the mid, full desired size",
            ),
            EdgeLine(
                "spread_cost",
                -self.spread_cost_usdc,
                self._per(-self.spread_cost_usdc),
                "crossing to the touch",
            ),
            EdgeLine(
                "slippage_cost",
                -self.slippage_cost_usdc,
                self._per(-self.slippage_cost_usdc),
                "modelled adverse fill beyond the touch",
            ),
            EdgeLine(
                "fee_cost",
                -self.fee_cost_usdc,
                self._per(-self.fee_cost_usdc),
                "taker fees and redemption gas",
            ),
            EdgeLine(
                "risk_limits",
                -self.risk_limit_usdc,
                self._per(-self.risk_limit_usdc),
                f"{self.blocked_by_risk} intent(s) refused by the risk engine",
            ),
            EdgeLine(
                "missed_fills",
                -self.missed_fill_usdc,
                self._per(-self.missed_fill_usdc),
                f"{self.blocked_by_depth} unfilled, {self.partial_fills} partial",
            ),
            EdgeLine(
                "net_profit",
                self.net_profit_usdc,
                self._per(self.net_profit_usdc),
                f"{self.filled} trade(s) actually filled",
            ),
        ]

    def reconciles(self, net_pnl_usdc: float | None = None) -> bool:
        """Does the ladder add up to the realised P&L?

        Checked against the engine's own bankroll movement when given, so the
        decomposition cannot drift away from the accounting it describes.
        """
        target = self.net_profit_usdc if net_pnl_usdc is None else net_pnl_usdc
        walked = (
            self.raw_edge_usdc
            - self.spread_cost_usdc
            - self.slippage_cost_usdc
            - self.fee_cost_usdc
            - self.risk_limit_usdc
            - self.missed_fill_usdc
        )
        return abs(walked - target) <= RECONCILE_TOLERANCE

    @property
    def fill_rate(self) -> float:
        """Filled intents over intents that reached the book."""
        attempted = self.intents - self.blocked_by_risk
        return self.filled / attempted if attempted > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lines": [line.as_dict() for line in self.lines],
            "intents": self.intents,
            "filled": self.filled,
            "blocked_by_risk": self.blocked_by_risk,
            "blocked_by_depth": self.blocked_by_depth,
            "partial_fills": self.partial_fills,
            "fill_rate": round(self.fill_rate, 6),
            "staked_usdc": round(self.staked_usdc, 4),
            "intended_usdc": round(self.intended_usdc, 4),
            "reconciles": self.reconciles(),
            "reconciliation_error_usdc": round(self.reconciliation_error_usdc, 9),
        }

    def render(self) -> str:
        """A waterfall a human can read in a terminal."""
        width = max(len(line.name) for line in self.lines)
        out = [
            "edge decomposition (USDC, and per USDC staked)",
            f"{'line':<{width}}  {'usdc':>12}  {'per_usdc':>10}  detail",
        ]
        for line in self.lines:
            if line.name == "net_profit":
                out.append("-" * (width + 30))
            out.append(
                f"{line.name:<{width}}  {line.usdc:>+12.4f}  "
                f"{line.per_usdc_staked:>+10.4f}  {line.detail}"
            )
        if not self.reconciles():
            out.append(
                f"!! decomposition does not reconcile: "
                f"error {self.reconciliation_error_usdc:+.9f} USDC"
            )
        return "\n".join(out)


def _won(window: WindowResult) -> float:
    """1.0 if the outcome we intended to buy is the one that settled."""
    outcome = window.decision.outcome
    if outcome is None or window.settled is None:
        return 0.0
    return 1.0 if window.settled is outcome else 0.0


def decompose(
    result: BacktestResult, costs: CostConfig | CostModel
) -> EdgeDecomposition:
    """Attribute a completed run's P&L to its causes.

    Considers every **intent** — every window where the decision gate said yes
    and the sizer produced a stake — not only the trades that were filled. The
    intents that never became trades are exactly where the risk-limit and
    missed-fill lines come from, and dropping them would make those costs
    invisible.
    """
    model = costs if isinstance(costs, CostModel) else CostModel(costs)
    gas = model.config.gas_cost_usdc
    out = EdgeDecomposition()

    p0 = p1 = p2 = p3 = p4 = p5 = 0.0

    for window in result.windows:
        if not window.intended:
            continue
        stake = window.desired_stake_usdc
        mid = window.mid_price
        touch = window.touch_price
        entry = (
            window.fill.price if window.fill is not None else window.decision.entry_price
        )
        win = _won(window)
        if mid <= 0.0 or touch <= 0.0 or entry <= 0.0:
            continue

        out.intents += 1
        out.intended_usdc += stake

        # Fees the full-size trade would have paid, plus gas if it would win.
        full_fee = model.fee(stake) + (gas if win > 0 else 0.0)

        p0 += stake / mid * win - stake
        p1 += stake / touch * win - stake
        p2 += stake / entry * win - stake
        p3 += stake / entry * win - stake - full_fee

        blocked_by_risk = not window.fill_attempted
        if blocked_by_risk:
            out.blocked_by_risk += 1
            continue

        # Survived risk: it reached the book.
        p4 += stake / entry * win - stake - full_fee

        if window.fill is None:
            out.blocked_by_depth += 1
            continue

        fill = window.fill
        out.filled += 1
        out.staked_usdc += fill.cost_usdc
        if fill.partial:
            out.partial_fills += 1
        realised_fee = fill.fee_usdc + (gas if win > 0 else 0.0)
        p5 += fill.shares * win - fill.notional_usdc - realised_fee

    out.raw_edge_usdc = p0
    out.spread_cost_usdc = p0 - p1
    out.slippage_cost_usdc = p1 - p2
    out.fee_cost_usdc = p2 - p3
    out.risk_limit_usdc = p3 - p4
    out.missed_fill_usdc = p4 - p5
    out.net_profit_usdc = p5

    walked = (
        out.raw_edge_usdc
        - out.spread_cost_usdc
        - out.slippage_cost_usdc
        - out.fee_cost_usdc
        - out.risk_limit_usdc
        - out.missed_fill_usdc
    )
    out.reconciliation_error_usdc = walked - out.net_profit_usdc

    if not out.reconciles():
        log.error(
            "attribution.reconciliation_failed",
            error_usdc=out.reconciliation_error_usdc,
            intents=out.intents,
            filled=out.filled,
        )
    return out
