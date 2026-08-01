"""The decision gate: probability in, position or a named refusal out.

Abstention is the default. A 5-minute BTC up/down market is close to fair, and
a liquid book's own forecast is a strong opponent, so the interesting question
is not "which way" but "is this one of the rare windows worth touching at all".
Every gate below can only ever *stop* a trade; none of them can start one.

Two properties are load-bearing:

*Every refusal is named.* The result carries a :class:`~pmbtc.constants.SkipReason`,
never a bare ``False``. "The bot stopped trading" must always be answerable from
the logs, and the distribution of reasons over a backtest is the fastest way to
see that a threshold is mis-set — a run that skips 90% of windows on
``spread_too_wide`` is a different problem from one that skips on
``edge_too_small``.

*Gates are ordered cheapest-and-most-fundamental first.* Data validity before
market conditions, market conditions before model opinion. A stale book should
report ``stale_data``, not a misleading ``edge_too_small`` computed from prices
that were never real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pmbtc.config import Config
from pmbtc.constants import Outcome, SkipReason
from pmbtc.logging_setup import get_logger
from pmbtc.trading.costs import CostModel, Quote

log = get_logger("pmbtc.trading.decision")


@dataclass(frozen=True, slots=True)
class Decision:
    """What to do about one window, and why."""

    trade: bool
    outcome: Outcome | None = None
    skip_reason: SkipReason | None = None
    detail: str = ""
    #: Model probability that the window settles UP.
    model_prob_up: float = 0.5
    #: The market's implied probability of UP (the book mid).
    market_prob_up: float = 0.5
    #: Modelled all-in entry price for ``outcome``.
    entry_price: float = 0.0
    #: Model probability of the chosen outcome minus its entry price, in
    #: probability points. This is the edge that has to survive costs.
    edge: float = 0.0
    #: Expected profit per USDC staked, after the entry price is paid.
    ev_per_usdc: float = 0.0
    #: Probability assigned to the chosen outcome, after any shrinkage.
    confidence: float = 0.5
    #: The raw model probability before confidence shrinkage, kept so a report
    #: can show how much of an abstention was the model and how much was us
    #: distrusting it.
    raw_prob_up: float = 0.5

    @property
    def expected_roi(self) -> float:
        """Alias for :attr:`ev_per_usdc` — return on the capital at risk."""
        return self.ev_per_usdc

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade": self.trade,
            "outcome": self.outcome.value if self.outcome else None,
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
            "detail": self.detail,
            "model_prob_up": round(self.model_prob_up, 6),
            "raw_prob_up": round(self.raw_prob_up, 6),
            "market_prob_up": round(self.market_prob_up, 6),
            "entry_price": round(self.entry_price, 6),
            "edge": round(self.edge, 6),
            "ev_per_usdc": round(self.ev_per_usdc, 6),
            "confidence": round(self.confidence, 6),
        }

    def __str__(self) -> str:
        if not self.trade:
            reason = self.skip_reason.value if self.skip_reason else "unknown"
            return f"SKIP[{reason}] {self.detail}"
        outcome = self.outcome.value if self.outcome else "?"
        return (
            f"TRADE {outcome} @ {self.entry_price:.4f} "
            f"p={self.confidence:.4f} edge={self.edge:+.4f} ev={self.ev_per_usdc:+.4f}"
        )


def _skip(reason: SkipReason, detail: str, **fields: Any) -> Decision:
    return Decision(trade=False, skip_reason=reason, detail=detail, **fields)


def shrink_toward_fair(probability: float, shrinkage: float) -> float:
    """Pull a probability toward 0.5 by ``shrinkage``.

    ``shrinkage = 0`` leaves it untouched; ``1`` collapses it to a coin flip.
    Applied before any gate, because an overconfident probability is dangerous
    at exactly two points — it decides whether we trade, and Kelly sizing is
    convex in it, so the same 0.05 of overconfidence costs more the larger the
    stated edge.
    """
    if shrinkage <= 0.0:
        return probability
    factor = 1.0 - min(1.0, shrinkage)
    return 0.5 + (probability - 0.5) * factor


class DecisionEngine:
    """Applies the abstention gates to one window."""

    def __init__(self, config: Config, costs: CostModel | None = None) -> None:
        self.config = config
        self.costs = costs or CostModel(config.costs)

    def decide(
        self,
        *,
        model_prob_up: float,
        quote: Quote,
        seconds_into_window: float,
        seconds_to_settlement: float,
        vol_zscore: float | None = None,
        calibration_error: float | None = None,
        model_agreement: float | None = None,
    ) -> Decision:
        """Trade or abstain on this window.

        ``model_prob_up`` is the calibrated probability that the window settles
        UP. ``vol_zscore``, ``calibration_error`` and ``model_agreement`` are
        optional: when unavailable the corresponding gate does not fire, because
        a missing diagnostic is not evidence of a problem — but the feed and
        book gates below are never optional.
        """
        execution = self.config.execution
        prediction = self.config.prediction

        # --- 0. Distrust the model in proportion to its recent error ---- #
        # Done first so every downstream gate sees the probability we are
        # actually willing to act on, not the one the model asserted.
        raw_prob_up = model_prob_up
        if prediction.confidence_shrinkage > 0.0:
            scale = 1.0
            if calibration_error is not None and prediction.max_calibration_error > 0:
                scale = min(1.0, calibration_error / prediction.max_calibration_error)
            model_prob_up = shrink_toward_fair(
                model_prob_up, prediction.confidence_shrinkage * scale
            )

        # --- 1. Is the data real? ------------------------------------- #
        if not quote.valid:
            return _skip(
                SkipReason.STALE_DATA,
                f"unusable quote bid={quote.best_bid} ask={quote.best_ask}",
                raw_prob_up=raw_prob_up,
            )
        # The book must still be fresh when the order *lands*, not when we
        # looked at it. Ignoring the round trip is how a backtest fills against
        # a book that had already moved.
        effective_age = quote.age_ms + self.costs.config.assumed_latency_ms
        if effective_age > execution.max_book_age_ms:
            return _skip(
                SkipReason.STALE_DATA,
                f"book {quote.age_ms}ms old + {self.costs.config.assumed_latency_ms}ms "
                f"latency = {effective_age}ms on arrival, budget "
                f"{execution.max_book_age_ms}ms",
                raw_prob_up=raw_prob_up,
            )

        market_prob = self.costs.implied_probability(quote)
        assert market_prob is not None  # quote.valid implies a mid
        spread = quote.spread
        assert spread is not None
        base = {
            "model_prob_up": model_prob_up,
            "raw_prob_up": raw_prob_up,
            "market_prob_up": market_prob,
        }

        # --- 2. Is the market tradeable? ------------------------------ #
        if spread > execution.max_spread:
            return _skip(
                SkipReason.SPREAD_TOO_WIDE,
                f"spread {spread:.4f} > {execution.max_spread:.4f}",
                **base,
            )

        # --- 3. Are we inside the tradeable part of the window? -------- #
        if seconds_into_window < execution.min_seconds_into_window:
            return _skip(
                SkipReason.TOO_EARLY_IN_WINDOW,
                f"{seconds_into_window:.0f}s into window, need "
                f"{execution.min_seconds_into_window}s",
                **base,
            )
        # The clock's hard no-submit zone is not overridable by any edge: an
        # order that cannot be acknowledged before settlement is not a trade,
        # it is a coin flip on transport latency.
        safety = self.config.clock.order_safety_window_seconds
        if seconds_to_settlement < safety:
            return _skip(
                SkipReason.TOO_CLOSE_TO_SETTLEMENT,
                f"{seconds_to_settlement:.0f}s to settlement, inside the "
                f"{safety}s no-submit window",
                **base,
            )

        # --- 4. Is the regime one we model? --------------------------- #
        if vol_zscore is not None and abs(vol_zscore) > execution.max_vol_zscore:
            return _skip(
                SkipReason.ABNORMAL_VOLATILITY,
                f"vol z-score {vol_zscore:+.2f} beyond ±{execution.max_vol_zscore}",
                **base,
            )
        if (
            calibration_error is not None
            and calibration_error > prediction.max_calibration_error
        ):
            return _skip(
                SkipReason.MODEL_UNCALIBRATED,
                f"recent calibration error {calibration_error:.4f} > "
                f"{prediction.max_calibration_error:.4f}",
                **base,
            )
        if model_agreement is not None and model_agreement < prediction.min_model_agreement:
            return _skip(
                SkipReason.LOW_CONFIDENCE,
                f"ensemble agreement {model_agreement:.2f} < "
                f"{prediction.min_model_agreement:.2f}",
                **base,
            )

        # --- 5. Does the model have an opinion worth acting on? -------- #
        outcome = Outcome.UP if model_prob_up >= 0.5 else Outcome.DOWN
        confidence = model_prob_up if outcome is Outcome.UP else 1.0 - model_prob_up
        if confidence < prediction.min_confidence:
            return _skip(
                SkipReason.LOW_CONFIDENCE,
                f"confidence {confidence:.4f} < {prediction.min_confidence:.4f}",
                **base,
            )

        entry = self.costs.entry_price(outcome, quote)
        if entry is None:
            return _skip(SkipReason.STALE_DATA, "no entry price for a valid quote", **base)

        # Edge is measured against the price we would actually pay, not the
        # mid. Beating the mid by 3 cents is worth nothing if the spread is 4.
        edge = confidence - entry
        ev_per_usdc = edge / entry
        priced = {
            **base,
            "outcome": outcome,
            "entry_price": entry,
            "edge": edge,
            "ev_per_usdc": ev_per_usdc,
            "confidence": confidence,
        }

        # --- 6. Depth on the side we must lift ------------------------ #
        depth = quote.depth_for(outcome)
        if depth < execution.min_book_depth_usdc:
            return _skip(
                SkipReason.INSUFFICIENT_LIQUIDITY,
                f"{depth:.0f} USDC resting, need {execution.min_book_depth_usdc:.0f}",
                **priced,
            )

        # --- 7. Is the edge big enough for when we are? --------------- #
        # Past ``min_seconds_to_settlement`` the window is not closed, but the
        # bar is raised: less time for the forecast to be right, and less room
        # to be wrong about the fill.
        late = seconds_to_settlement < execution.min_seconds_to_settlement
        required_edge = execution.late_entry_min_edge if late else prediction.min_edge
        if edge < required_edge:
            return _skip(
                SkipReason.EDGE_TOO_SMALL,
                f"edge {edge:+.4f} < {required_edge:.4f}"
                + (" (late-window bar)" if late else ""),
                **priced,
            )

        # --- 8. Does it survive as expected value? -------------------- #
        if ev_per_usdc < prediction.min_ev:
            return _skip(
                SkipReason.NEGATIVE_EV,
                f"EV {ev_per_usdc:+.4f}/USDC < {prediction.min_ev:.4f}",
                **priced,
            )
        if ev_per_usdc < prediction.min_expected_roi:
            return _skip(
                SkipReason.NEGATIVE_EV,
                f"expected ROI {ev_per_usdc:+.4f} < {prediction.min_expected_roi:.4f}",
                **priced,
            )

        return Decision(
            trade=True,
            detail="all gates passed",
            model_prob_up=model_prob_up,
            raw_prob_up=raw_prob_up,
            market_prob_up=market_prob,
            outcome=outcome,
            entry_price=entry,
            edge=edge,
            ev_per_usdc=ev_per_usdc,
            confidence=confidence,
        )
