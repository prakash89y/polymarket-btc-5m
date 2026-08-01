"""What the book would actually have given us.

The fill model is where a backtest most easily lies to itself. Three assumptions
do almost all of the damage, so each is refused explicitly here:

1. *Filling at the mid.* We are not the maker. Taking liquidity on a two-sided
   book means paying the ask, and on a 2-cent spread that single assumption is
   worth about 2% of notional per trade — larger than most of the edge being
   claimed.
2. *Infinite depth.* Reconstructed books flatter size. A stake larger than the
   resting depth would have walked the book or gone unfilled; here it is capped
   at the depth that was actually there.
3. *Missing data meaning "fine".* Under ``pessimistic_fill`` an unknown depth is
   treated as no depth. A backtest that cannot see the book does not get to
   assume it was deep.

``mid`` and ``aggressive`` exist as sensitivity analyses, not as options to be
chosen because they produce a better number. The gap between ``touch`` and
``mid`` is the honest measure of how much of a strategy is really a bet on
getting maker fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pmbtc.config import BacktestConfig, CostConfig
from pmbtc.constants import Outcome, SkipReason
from pmbtc.logging_setup import get_logger
from pmbtc.trading.costs import CostModel, Fill, Quote

log = get_logger("pmbtc.backtest.fills")

FillStyle = Literal["touch", "mid", "aggressive"]


@dataclass(frozen=True, slots=True)
class FillOutcome:
    """A fill, or a named reason there wasn't one."""

    fill: Fill | None
    skip_reason: SkipReason | None = None
    detail: str = ""

    @property
    def filled(self) -> bool:
        return self.fill is not None


class FillModel:
    """Prices and sizes an entry against the book that was recorded."""

    def __init__(
        self,
        costs: CostModel | CostConfig,
        *,
        style: FillStyle = "touch",
        pessimistic: bool = True,
        allow_partial: bool = True,
    ) -> None:
        self.costs = costs if isinstance(costs, CostModel) else CostModel(costs)
        self.style = style
        self.pessimistic = pessimistic
        self.allow_partial = allow_partial

    @classmethod
    def from_config(
        cls,
        costs: CostConfig,
        backtest: BacktestConfig,
        *,
        style: FillStyle = "touch",
    ) -> FillModel:
        return cls(CostModel(costs), style=style, pessimistic=backtest.pessimistic_fill)

    # ------------------------------------------------------------------ #
    def price_for(self, outcome: Outcome, quote: Quote) -> float | None:
        """The modelled fill price before depth is considered."""
        touch = self.costs.touch_price(outcome, quote)
        if touch is None:
            return None
        if self.style == "mid":
            # Optimistic reference only: assumes we captured the spread.
            mid = quote.mid
            return mid if outcome is Outcome.UP else (1.0 - mid if mid is not None else None)
        if self.style == "aggressive":
            # Assume we cleared the touch and paid into the next level. The
            # spread is the best available estimate of that level's distance.
            spread = quote.spread or 0.0
            return touch + self.costs.config.slippage + spread
        return touch + self.costs.config.slippage

    def execute(self, *, outcome: Outcome, quote: Quote, stake_usdc: float) -> FillOutcome:
        """Fill ``stake_usdc`` of ``outcome``, or explain why not."""
        if not quote.valid:
            return FillOutcome(None, SkipReason.STALE_DATA, "unusable quote")
        price = self.price_for(outcome, quote)
        if price is None:
            return FillOutcome(None, SkipReason.STALE_DATA, "no fill price")

        depth = quote.depth_for(outcome)
        if self.pessimistic and depth <= 0.0:
            return FillOutcome(
                None,
                SkipReason.INSUFFICIENT_LIQUIDITY,
                "no recorded depth on the side we must lift",
            )

        requested = stake_usdc
        available = depth if self.pessimistic else max(depth, stake_usdc)
        if stake_usdc > available:
            if not self.allow_partial:
                return FillOutcome(
                    None,
                    SkipReason.INSUFFICIENT_LIQUIDITY,
                    f"stake {stake_usdc:.2f} exceeds depth {available:.2f}",
                )
            stake_usdc = available

        fill = self.costs.fill(
            outcome, quote, stake_usdc, price_override=price, requested_usdc=requested
        )
        if fill is None:
            return FillOutcome(None, SkipReason.STALE_DATA, "cost model returned no fill")
        return FillOutcome(fill)
