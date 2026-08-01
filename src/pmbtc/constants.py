"""Domain vocabulary shared by every module.

Keeping these in one place means a typo like ``"UP"`` vs ``"up"`` is a failed
enum lookup at import time instead of a silently mismatched dictionary key three
modules downstream.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

PACKAGE_NAME: Final = "pmbtc"

# --------------------------------------------------------------------------- #
# Market structure
# --------------------------------------------------------------------------- #
#: Polymarket binary outcome shares are quoted in USDC in [0, 1] and pay 1 on
#: the winning side. Prices are clamped into this open interval before any
#: division (Kelly and log-loss both explode at the boundaries).
PRICE_MIN: Final = 0.001
PRICE_MAX: Final = 0.999

#: Polygon mainnet — the chain Polymarket's CTF exchange settles on.
POLYGON_CHAIN_ID: Final = 137

#: USDC.e on Polygon, 6 decimals. Size arithmetic is done in integer micro-USDC
#: wherever it touches the chain, never in float.
USDC_DECIMALS: Final = 6


class Outcome(StrEnum):
    """The two sides of a Bitcoin Up/Down market."""

    UP = "up"
    DOWN = "down"

    @property
    def opposite(self) -> Outcome:
        return Outcome.DOWN if self is Outcome.UP else Outcome.UP


class Side(StrEnum):
    """Order direction against an outcome token."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Polymarket CLOB order lifetimes.

    ``FOK`` is the default for entries: a partial fill on a 5-minute binary
    leaves us with an unhedged stub and no time to work the remainder.
    """

    GTC = "gtc"
    GTD = "gtd"
    FOK = "fok"
    FAK = "fak"


class RunMode(StrEnum):
    """Top-level operating mode. ``LIVE`` is gated by two independent switches."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class SettlementSource(StrEnum):
    """Price feed a market declares as its resolution authority.

    Polymarket has used different feeds across crypto market families. The bot
    models exactly one of these and refuses any market whose rules text does not
    match — see :class:`pmbtc.exceptions.SettlementSourceMismatch`.
    """

    BINANCE_SPOT = "binance_spot"
    CHAINLINK = "chainlink"
    PYTH = "pyth"
    COINBASE_SPOT = "coinbase_spot"
    UNKNOWN = "unknown"


class WindowPhase(StrEnum):
    """Where we are inside a 5-minute window.

    The information content of the order flow and the shape of the fair-value
    curve differ sharply by phase, so this is both a feature and a gate:

    OPENING
        First ~60s. Price is near the strike, probability is near 0.50, and the
        Polymarket book is usually widest and least informed.
    MID
        The workhorse window. Realized displacement is meaningful relative to
        remaining volatility, and the book has usually repriced.
    LATE
        Displacement dominates remaining volatility; probabilities go convex.
        Small model errors become large EV errors here.
    FINAL
        Last seconds. Fill risk, cancel risk, and settlement-feed latency make
        this hostile. Blocked unless the edge is still large.
    """

    OPENING = "opening"
    MID = "mid"
    LATE = "late"
    FINAL = "final"


class TradeStatus(StrEnum):
    """Lifecycle of a bot-originated position."""

    PENDING = "pending"
    OPEN = "open"
    CLOSED_EARLY = "closed_early"
    SETTLED_WIN = "settled_win"
    SETTLED_LOSS = "settled_loss"
    VOIDED = "voided"
    REJECTED = "rejected"


class SkipReason(StrEnum):
    """Why an opportunity was not traded.

    Every evaluation is logged with one of these even when no order is sent.
    The distribution of skip reasons over time is the fastest diagnostic for
    "why has the bot stopped trading?" and feeds the dashboard directly.
    """

    NO_MARKET = "no_market"
    SETTLEMENT_MISMATCH = "settlement_mismatch"
    LOW_CONFIDENCE = "low_confidence"
    NEGATIVE_EV = "negative_ev"
    EDGE_TOO_SMALL = "edge_too_small"
    SPREAD_TOO_WIDE = "spread_too_wide"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    ABNORMAL_VOLATILITY = "abnormal_volatility"
    STALE_DATA = "stale_data"
    TOO_CLOSE_TO_SETTLEMENT = "too_close_to_settlement"
    TOO_EARLY_IN_WINDOW = "too_early_in_window"
    RISK_LIMIT = "risk_limit"
    KILL_SWITCH = "kill_switch"
    MODEL_UNCALIBRATED = "model_uncalibrated"
    DUPLICATE_POSITION = "duplicate_position"
    SIZE_BELOW_MINIMUM = "size_below_minimum"


class ModelName(StrEnum):
    """Learners the ensemble may contain."""

    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    TFT = "tft"
    TABNET = "tabnet"
    LOGISTIC = "logistic"
    BASELINE_VOL = "baseline_vol"


class Venue(StrEnum):
    """Market-data venues used for features (not for execution)."""

    BINANCE_FUTURES = "binance_futures"
    BINANCE_SPOT = "binance_spot"
    COINBASE = "coinbase"
    BYBIT = "bybit"
    HYPERLIQUID = "hyperliquid"


#: Session labels double as a categorical feature and as a regime key for
#: per-session performance attribution.
class Session(StrEnum):
    ASIA = "asia"
    LONDON = "london"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    NEW_YORK = "new_york"
    LATE_US = "late_us"
