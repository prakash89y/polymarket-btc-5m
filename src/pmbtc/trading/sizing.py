"""Position sizing: fractional Kelly, with the caps that matter more than Kelly.

For a binary contract bought at price ``c`` that pays 1, with true probability
``p``, the growth-optimal fraction of bankroll to stake is::

    f* = (p - c) / (1 - c)

That formula is exactly right and almost never usable, because it assumes ``p``
is the truth. Here ``p`` is a model output with its own error, and Kelly is
brutally asymmetric about that error: overestimating ``p`` by a little costs far
more growth than underestimating it by the same amount, and full Kelly on a
biased estimate is a reliable route to ruin. Hence quarter-Kelly by default,
hard caps above it, and a further haircut when the model's recent calibration is
poor.

The caps are not decoration. On a near-fair 5-minute market a model that briefly
believes ``p = 0.95`` at ``c = 0.50`` computes ``f* = 0.90`` — ninety percent of
bankroll on one coin flip. ``max_risk_per_trade`` is what stands between that
belief and the account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pmbtc.config import Config, SizingConfig
from pmbtc.constants import SkipReason
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.trading.sizing")


def kelly_fraction(prob: float, price: float) -> float:
    """Full-Kelly bankroll fraction for a binary contract. Never negative.

    A non-positive edge returns 0: Kelly's answer to a bad bet is not to bet it
    smaller, it is not to bet it.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    edge = prob - price
    if edge <= 0.0:
        return 0.0
    return edge / (1.0 - price)


@dataclass(frozen=True, slots=True)
class Stake:
    """How much to stake, and what decided it."""

    usdc: float
    shares: float
    fraction_of_bankroll: float
    #: Full-Kelly fraction before the safety factor and the caps.
    kelly_full: float
    #: Which constraint set the final size — invaluable when a backtest's
    #: returns turn out to be a story about one cap rather than about the model.
    capped_by: str
    accepted: bool
    skip_reason: SkipReason | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "usdc": round(self.usdc, 4),
            "shares": round(self.shares, 4),
            "fraction_of_bankroll": round(self.fraction_of_bankroll, 6),
            "kelly_full": round(self.kelly_full, 6),
            "capped_by": self.capped_by,
            "accepted": self.accepted,
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        if not self.accepted:
            return f"NO STAKE ({self.detail})"
        return (
            f"{self.usdc:.2f} USDC ({self.fraction_of_bankroll:.4f} of bankroll, "
            f"capped_by={self.capped_by})"
        )


class PositionSizer:
    """Turns a decision into an amount of money. Pure and deterministic."""

    def __init__(self, config: Config | SizingConfig) -> None:
        self.config = config.sizing if isinstance(config, Config) else config
        self._max_calibration_error = (
            config.prediction.max_calibration_error if isinstance(config, Config) else 0.04
        )

    # ------------------------------------------------------------------ #
    def size(
        self,
        *,
        bankroll_usdc: float,
        model_prob: float,
        entry_price: float,
        calibration_error: float | None = None,
    ) -> Stake:
        """Stake for one entry.

        ``model_prob`` is the probability of the outcome being bought (not of
        UP), and ``entry_price`` is its all-in modelled price, so the two are
        directly comparable.
        """
        cfg = self.config
        if bankroll_usdc <= 0.0:
            return self._reject("bankroll exhausted", SkipReason.RISK_LIMIT)
        if not (0.0 < entry_price < 1.0):
            return self._reject(f"unusable entry price {entry_price}", SkipReason.STALE_DATA)

        full = kelly_fraction(model_prob, entry_price)

        if cfg.mode == "kelly":
            fraction = full * cfg.kelly_fraction
            capped_by = "kelly"
            if fraction <= 0.0:
                return self._reject(
                    f"no Kelly edge at p={model_prob:.4f} c={entry_price:.4f}",
                    SkipReason.NEGATIVE_EV,
                    kelly_full=full,
                )
        elif cfg.mode == "fixed_fraction":
            fraction = cfg.fixed_fraction
            capped_by = "fixed_fraction"
        else:  # fixed_usdc
            fraction = min(1.0, cfg.fixed_usdc / bankroll_usdc)
            capped_by = "fixed_usdc"

        # Shrink toward zero when recent calibration is poor. A model whose
        # stated probabilities are drifting has not earned full size, even when
        # its ranking is still good.
        if cfg.calibration_scaling and calibration_error is not None:
            limit = self._max_calibration_error
            haircut = 1.0 - (calibration_error / limit if limit > 0 else 0.0)
            haircut = min(1.0, max(0.0, haircut))
            if haircut < 1.0:
                fraction *= haircut
                capped_by = "calibration"
            if fraction <= 0.0:
                return self._reject(
                    f"calibration error {calibration_error:.4f} zeroed the size",
                    SkipReason.MODEL_UNCALIBRATED,
                    kelly_full=full,
                )

        # Hard caps, applied in ascending order of authority.
        if fraction > cfg.max_risk_per_trade:
            fraction, capped_by = cfg.max_risk_per_trade, "max_risk_per_trade"

        usdc = fraction * bankroll_usdc
        if usdc > cfg.max_position_usdc:
            usdc, capped_by = cfg.max_position_usdc, "max_position_usdc"
        if usdc > bankroll_usdc:
            usdc, capped_by = bankroll_usdc, "bankroll"

        # Round down, never up: rounding up would breach the cap we just applied.
        step = cfg.stake_rounding_usdc
        rounded = math.floor(usdc / step) * step if step > 0 else usdc

        if rounded < cfg.min_position_usdc:
            return self._reject(
                f"stake {rounded:.2f} below minimum {cfg.min_position_usdc:.2f}",
                SkipReason.SIZE_BELOW_MINIMUM,
                kelly_full=full,
            )

        return Stake(
            usdc=rounded,
            shares=rounded / entry_price,
            fraction_of_bankroll=rounded / bankroll_usdc,
            kelly_full=full,
            capped_by=capped_by,
            accepted=True,
            detail=f"mode={cfg.mode}",
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reject(detail: str, reason: SkipReason, *, kelly_full: float = 0.0) -> Stake:
        return Stake(
            usdc=0.0,
            shares=0.0,
            fraction_of_bankroll=0.0,
            kelly_full=kelly_full,
            capped_by="none",
            accepted=False,
            skip_reason=reason,
            detail=detail,
        )
