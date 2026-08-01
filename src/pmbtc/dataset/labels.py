"""Label resolution -- from the official Polymarket outcome, and nothing else.

The rule, without exception: **the label is what Polymarket paid out.** Not what
Chainlink printed, not what our recorded tape says, not what the settlement
price implies. Those are audit inputs; disagreeing with them is information, but
they never become the target.

The reason is not deference to the venue. It is that the bot's PnL is decided by
the venue's resolution, so a model trained on a reconstructed label is optimising
a different objective than the one that pays. Where the venue's outcome and an
external feed disagree, the *disagreement* is the signal worth recording -- and
we record it, without touching the label.

A resolved Gamma market carries ``outcomePrices`` as a JSON-encoded string that
settles to ``["1", "0"]`` (Up won) or ``["0", "1"]`` (Down won), paired with
``outcomes`` of ``["Up", "Down"]``. Anything else -- a 50-50 void, a still-open
market, a malformed pair -- yields no label rather than a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pmbtc.constants import Outcome
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.dataset.labels")

#: How close a settled price must be to 1.0 (or 0.0) to count as a clean payout.
_SETTLED_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class LabelResolution:
    """The outcome of trying to label one market."""

    outcome: Outcome | None
    settlement_probability: float | None
    yes_final_price: float | None
    no_final_price: float | None
    resolved: bool
    reason: str = ""

    @property
    def is_void(self) -> bool:
        """A 50-50 resolution: real, and not a training example."""
        return self.resolved and self.outcome is None


def _parse_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_label(market: dict[str, Any]) -> LabelResolution:
    """Extract the official outcome from a resolved Gamma market payload."""
    outcomes = _parse_json_list(market.get("outcomes"))
    prices = _parse_json_list(market.get("outcomePrices"))

    if not market.get("closed"):
        return LabelResolution(None, None, None, None, False, "market is not closed")
    if len(outcomes) != 2 or len(prices) != 2:
        return LabelResolution(
            None, None, None, None, False,
            f"expected two outcomes and two prices, got {len(outcomes)}/{len(prices)}",
        )

    values = [_as_float(p) for p in prices]
    if any(v is None for v in values):
        return LabelResolution(None, None, None, None, False, f"unparseable prices {prices}")

    # Index by outcome name rather than position: the ordering is a convention,
    # and conventions change.
    named = {name.strip().lower(): value for name, value in zip(outcomes, values, strict=True)}
    up_price = named.get("up")
    down_price = named.get("down")
    if up_price is None or down_price is None:
        return LabelResolution(
            None, None, None, None, False, f"outcomes are not Up/Down: {outcomes}"
        )

    total = up_price + down_price
    if abs(total - 1.0) > 0.05:
        # Prices that do not sum to ~1 mean the market has not truly settled.
        return LabelResolution(
            None, None, up_price, down_price, False,
            f"outcome prices sum to {total:.3f}, not 1.0",
        )

    if abs(up_price - 1.0) <= _SETTLED_TOLERANCE:
        return LabelResolution(Outcome.UP, up_price, up_price, down_price, True, "settled Up")
    if abs(down_price - 1.0) <= _SETTLED_TOLERANCE:
        return LabelResolution(Outcome.DOWN, down_price, up_price, down_price, True, "settled Down")
    if abs(up_price - 0.5) <= _SETTLED_TOLERANCE:
        # A genuine 50-50 void. Resolved, but not a classification example.
        return LabelResolution(None, 0.5, up_price, down_price, True, "resolved 50-50 (void)")

    return LabelResolution(
        None, None, up_price, down_price, False,
        f"prices {up_price}/{down_price} are not a settled payout",
    )


@dataclass(frozen=True, slots=True)
class LabelAudit:
    """Comparison of the official outcome against an external reference.

    Never used to change a label. Its purpose is to surface the case where our
    understanding of settlement and the venue's payout diverge -- which is the
    single most important alarm this system can raise, because it means either
    the settlement spec is wrong or the reference feed is.
    """

    condition_id: str
    official: Outcome | None
    reference: Outcome | None
    reference_source: str
    open_price: float | None = None
    close_price: float | None = None

    @property
    def agrees(self) -> bool | None:
        if self.official is None or self.reference is None:
            return None
        return self.official is self.reference

    def __str__(self) -> str:
        verdict = {True: "agree", False: "DISAGREE", None: "incomparable"}[self.agrees]
        return (
            f"{self.condition_id[:14]}: official={self.official} "
            f"{self.reference_source}={self.reference} -> {verdict}"
        )


def audit_against_reference(
    condition_id: str,
    official: Outcome | None,
    open_price: float | None,
    close_price: float | None,
    reference_source: str,
    tie_resolves_up: bool = True,
) -> LabelAudit:
    """Derive what an external feed *would* have said, for comparison only."""
    reference: Outcome | None = None
    if open_price is not None and close_price is not None:
        if close_price > open_price:
            reference = Outcome.UP
        elif close_price < open_price:
            reference = Outcome.DOWN
        else:
            reference = Outcome.UP if tie_resolves_up else Outcome.DOWN

    audit = LabelAudit(
        condition_id, official, reference, reference_source, open_price, close_price
    )
    if audit.agrees is False:
        log.error("label.audit_disagreement", detail=str(audit))
    return audit
