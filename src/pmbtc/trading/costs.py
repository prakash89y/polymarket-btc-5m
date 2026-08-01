"""The cost model.

On Polymarket the dominant cost of a 5-minute trade is not a fee — fees are
currently zero. It is crossing a two-sided book that is routinely 1-3 cents wide
on a contract worth about 50 cents. A one-cent half-spread on a 0.50 contract is
two percent of notional, paid on entry, on an instrument whose whole edge is
measured in single percentage points. Model it wrong and a backtest will report
a profitable strategy that loses money on contact with the book.

So the arithmetic is written once, here, and used identically by the backtest,
the paper engine, and live execution.

**Binary payoff.** A share of an outcome token costs ``c`` USDC and pays exactly
1 USDC if that outcome settles true, 0 otherwise. Buying ``n`` shares therefore
risks ``n * c`` to win ``n * (1 - c)``.

**Which price we pay.** Polymarket lists paired complementary tokens. Buying UP
lifts the UP ask. Buying DOWN lifts the DOWN ask, and under no-arbitrage between
the pair ``ask_down = 1 - bid_up`` — so a single two-sided UP quote prices both
directions. This is the conservative reading in both directions: it crosses the
spread whichever way we trade, and it never assumes we captured the mid.
"""

from __future__ import annotations

from dataclasses import dataclass

from pmbtc.config import CostConfig
from pmbtc.constants import Outcome

#: Prices are probabilities; a contract is never worth less than nothing or
#: more than the dollar it pays. Fills are clamped into the open interval
#: because a fill at exactly 0.0 or 1.0 implies infinite or zero return.
MIN_PRICE = 0.001
MAX_PRICE = 0.999


def clamp_price(price: float) -> float:
    return min(MAX_PRICE, max(MIN_PRICE, price))


@dataclass(frozen=True, slots=True)
class Quote:
    """A two-sided quote for the UP token, with the depth standing behind it.

    ``age_ms`` is how stale the book was when the decision was taken. It is
    carried here rather than checked here: pricing and gating are separate
    concerns, and the gate lives in :mod:`pmbtc.trading.decision`.
    """

    best_bid: float | None
    best_ask: float | None
    bid_depth_usdc: float = 0.0
    ask_depth_usdc: float = 0.0
    age_ms: int = 0

    @property
    def valid(self) -> bool:
        """Two-sided, ordered, and inside the price bounds.

        A one-sided book is not tradeable at a known price, and a crossed book
        means the snapshot is inconsistent — neither is a thing to trade on.
        """
        if self.best_bid is None or self.best_ask is None:
            return False
        if not (0.0 < self.best_bid < 1.0 and 0.0 < self.best_ask < 1.0):
            return False
        return self.best_bid < self.best_ask

    @property
    def mid(self) -> float | None:
        if not self.valid:
            return None
        assert self.best_bid is not None and self.best_ask is not None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if not self.valid:
            return None
        assert self.best_bid is not None and self.best_ask is not None
        return self.best_ask - self.best_bid

    def depth_for(self, outcome: Outcome) -> float:
        """USDC resting on the side we would have to lift to buy ``outcome``.

        Buying DOWN is filled by the resting UP bid (the complement), so the
        depth that matters for a DOWN entry is the bid depth.
        """
        return self.ask_depth_usdc if outcome is Outcome.UP else self.bid_depth_usdc


@dataclass(frozen=True, slots=True)
class Fill:
    """The realised economics of one entry."""

    outcome: Outcome
    price: float
    shares: float
    notional_usdc: float
    fee_usdc: float
    slippage_usdc: float
    requested_usdc: float

    @property
    def cost_usdc(self) -> float:
        """Everything that leaves the bankroll to open this position."""
        return self.notional_usdc + self.fee_usdc

    @property
    def partial(self) -> bool:
        return self.notional_usdc + 1e-9 < self.requested_usdc

    def payoff_usdc(self, settled: Outcome) -> float:
        """USDC returned at settlement: 1 per share if right, 0 if wrong."""
        return self.shares if settled is self.outcome else 0.0

    def pnl_usdc(self, settled: Outcome, *, gas_usdc: float = 0.0) -> float:
        redemption_gas = gas_usdc if settled is self.outcome else 0.0
        return self.payoff_usdc(settled) - self.cost_usdc - redemption_gas


class CostModel:
    """Prices an entry against a quote. Pure: no clock, no network, no state."""

    def __init__(self, config: CostConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Prices
    # ------------------------------------------------------------------ #
    def touch_price(self, outcome: Outcome, quote: Quote) -> float | None:
        """The price to buy ``outcome`` at the touch, before slippage.

        UP lifts the UP ask. DOWN lifts the DOWN ask, which no-arbitrage puts
        at ``1 - bid_up``.
        """
        if not quote.valid:
            return None
        assert quote.best_bid is not None and quote.best_ask is not None
        return quote.best_ask if outcome is Outcome.UP else 1.0 - quote.best_bid

    def entry_price(self, outcome: Outcome, quote: Quote) -> float | None:
        """Touch plus the modelled adverse fill."""
        touch = self.touch_price(outcome, quote)
        if touch is None:
            return None
        return clamp_price(touch + self.config.slippage)

    def implied_probability(self, quote: Quote) -> float | None:
        """The market's own forecast of UP — the number a model must beat."""
        return quote.mid

    # ------------------------------------------------------------------ #
    # Charges
    # ------------------------------------------------------------------ #
    def fee(self, notional_usdc: float, *, maker: bool = False) -> float:
        bps = self.config.maker_fee_bps if maker else self.config.taker_fee_bps
        return notional_usdc * bps / 10_000.0

    def round_trip_cost(self, quote: Quote) -> float | None:
        """Total spread-and-slippage cost of getting in and back out again.

        With ``assume_hold_to_settlement`` the position is redeemed at its
        terminal value rather than sold, so only the entry leg is charged. That
        is not a rounding detail: it roughly halves the cost of a trade and is
        the reason holding to settlement is the default exit policy.
        """
        spread = quote.spread
        if spread is None:
            return None
        entry = spread / 2.0 + self.config.slippage
        if self.config.assume_hold_to_settlement:
            return entry
        return 2.0 * entry

    # ------------------------------------------------------------------ #
    # Fills
    # ------------------------------------------------------------------ #
    def fill(
        self,
        outcome: Outcome,
        quote: Quote,
        stake_usdc: float,
        *,
        price_override: float | None = None,
        extra_slippage: float = 0.0,
        requested_usdc: float | None = None,
    ) -> Fill | None:
        """Convert a stake in USDC into a position at a modelled price.

        ``price_override`` lets a fill model price the trade differently (see
        :mod:`pmbtc.backtest.fills`) while keeping fee and share arithmetic in
        one place. ``requested_usdc`` preserves what was *asked* for when the
        caller has already capped ``stake_usdc`` to the available depth —
        without it a depth-capped fill would look complete.
        """
        price = price_override if price_override is not None else self.entry_price(outcome, quote)
        if price is None or stake_usdc <= 0.0:
            return None
        price = clamp_price(price + extra_slippage)

        shares = stake_usdc / price
        notional = shares * price
        touch = self.touch_price(outcome, quote)
        slippage_usdc = shares * (price - touch) if touch is not None else 0.0
        return Fill(
            outcome=outcome,
            price=price,
            shares=shares,
            notional_usdc=notional,
            fee_usdc=self.fee(notional),
            slippage_usdc=max(0.0, slippage_usdc),
            requested_usdc=stake_usdc if requested_usdc is None else requested_usdc,
        )
