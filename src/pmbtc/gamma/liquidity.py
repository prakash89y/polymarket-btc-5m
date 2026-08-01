"""Liquidity snapshots.

Captured for **every** discovered market, including rejected ones. That is
deliberate: the book state of markets the bot declined is exactly the control
group needed later to tell "the model was wrong" apart from "the book was
untradeable", and it costs nothing to record.

Gamma publishes top-of-book and aggregate figures (``bestBid``, ``bestAsk``,
``spread``, ``lastTradePrice``, ``volume``, ``liquidity``). Full ladder depth
lives on the CLOB book endpoint and arrives with Module 5; the model here has
optional ``bids``/``asks`` ladders so that filling them later changes nothing
downstream.

Snapshots are appended as JSONL keyed by ``(condition_id, captured_at_ms)`` so
the feature pipeline can join them to any decision timestamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

from pmbtc.config import Config
from pmbtc.constants import PRICE_MAX, PRICE_MIN


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


class BookLevel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    price: float
    size: float


class LiquiditySnapshot(BaseModel):
    """Point-in-time view of one market's tradeability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_id: str
    slug: str = ""
    captured_at_ms: int
    #: Prices are probabilities in [0, 1]; None means the side was unquoted.
    best_bid: float | None = None
    best_ask: float | None = None
    last_trade_price: float | None = None
    volume_usdc: float | None = None
    liquidity_usdc: float | None = None
    open_interest_usdc: float | None = None
    #: Populated by the CLOB collector in Module 5.
    bids: tuple[BookLevel, ...] = Field(default_factory=tuple)
    asks: tuple[BookLevel, ...] = Field(default_factory=tuple)
    #: Spread as published by Gamma, kept alongside the computed one so a
    #: disagreement between them is visible rather than averaged away.
    reported_spread: float | None = None

    # ------------------------------------------------------------------ #
    @property
    def two_sided(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None

    @property
    def spread(self) -> float | None:
        """Computed from the touch, falling back to the reported value."""
        if self.two_sided:
            assert self.best_ask is not None and self.best_bid is not None
            return self.best_ask - self.best_bid
        return self.reported_spread

    @property
    def mid(self) -> float | None:
        """Mid-price -- the market's implied probability of the Up outcome.

        Clamped away from 0 and 1: this value is divided by in the edge and
        Kelly calculations, and a book quoting 0.00/0.01 must not produce an
        infinite position size.
        """
        if not self.two_sided:
            return None
        assert self.best_ask is not None and self.best_bid is not None
        raw = (self.best_bid + self.best_ask) / 2.0
        return min(max(raw, PRICE_MIN), PRICE_MAX)

    @property
    def depth_bid_usdc(self) -> float:
        return sum(level.price * level.size for level in self.bids)

    @property
    def depth_ask_usdc(self) -> float:
        return sum((1.0 - level.price) * level.size for level in self.asks)

    def as_record(self) -> dict[str, Any]:
        record = self.model_dump(mode="json")
        record["spread"] = self.spread
        record["mid"] = self.mid
        record["two_sided"] = self.two_sided
        return record


def snapshot_from_market(
    market: dict[str, Any], *, captured_at_ms: int, open_interest: float | None = None
) -> LiquiditySnapshot:
    """Build a snapshot from a Gamma market payload."""
    event = (market.get("events") or [{}])[0]
    if open_interest is None and isinstance(event, dict):
        open_interest = _as_float(event.get("openInterest"))
    return LiquiditySnapshot(
        condition_id=str(market.get("conditionId") or ""),
        slug=str(market.get("slug") or ""),
        captured_at_ms=captured_at_ms,
        best_bid=_as_float(market.get("bestBid")),
        best_ask=_as_float(market.get("bestAsk")),
        last_trade_price=_as_float(market.get("lastTradePrice")),
        volume_usdc=_as_float(market.get("volume")),
        liquidity_usdc=_as_float(market.get("liquidity")),
        open_interest_usdc=open_interest,
        reported_spread=_as_float(market.get("spread")),
    )


class LiquidityStore:
    """Append-only JSONL sink for snapshots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: LiquiditySnapshot) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(orjson.dumps(snapshot.as_record(), default=str).decode("utf-8") + "\n")
            fh.flush()

    def write_many(self, snapshots: list[LiquiditySnapshot]) -> None:
        if not snapshots:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for snapshot in snapshots:
                fh.write(orjson.dumps(snapshot.as_record(), default=str).decode("utf-8") + "\n")
            fh.flush()

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(orjson.loads(line))
                except orjson.JSONDecodeError:
                    continue
        return rows


def open_liquidity_store(config: Config) -> LiquidityStore:
    return LiquidityStore(
        config.resolved_path(config.app.data_dir) / "liquidity_snapshots.jsonl"
    )
