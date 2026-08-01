"""Typed exception hierarchy.

Design decision: every failure mode gets its own class rather than a bare
``Exception``. The trading loop must distinguish "the venue is rate limiting me,
back off and retry" from "the market's resolution source is not the one I model,
never touch it" — without string-matching error messages.

Two properties are encoded on the class so callers can branch structurally:

``retryable``
    Safe for the transport layer to retry with backoff.
``halts_trading``
    Serious enough that the supervisor should flatten and stand down rather
    than continue. Anything touching settlement correctness, look-ahead
    leakage, or risk limits sets this.
"""

from __future__ import annotations


class BotError(Exception):
    """Base class for every error raised by this package."""

    retryable: bool = False
    halts_trading: bool = False

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        ctx = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({ctx})"


# --------------------------------------------------------------------------- #
# Configuration / startup
# --------------------------------------------------------------------------- #
class ConfigError(BotError):
    """Malformed, missing, or internally inconsistent configuration."""

    halts_trading = True


class MissingCredentialError(ConfigError):
    """A required API credential was not supplied via the environment."""


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class TransportError(BotError):
    """Base for anything that goes wrong talking to an external service."""


class NetworkError(TransportError):
    """Connection reset, DNS failure, TLS error, timeout."""

    retryable = True


class HttpStatusError(TransportError):
    """Non-2xx HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        url: str,
        body: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message, context={"status_code": status_code, "url": url})
        self.status_code = status_code
        self.url = url
        self.body = body
        self.retryable = retryable


class RateLimitError(TransportError):
    """HTTP 429, or a local token-bucket rejection."""

    retryable = True

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message, context={"retry_after_s": retry_after_s})
        self.retry_after_s = retry_after_s


class WebsocketError(TransportError):
    """A streaming connection dropped or produced an unparseable frame."""

    retryable = True


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #
class DataError(BotError):
    """Base for data-quality problems."""


class DataGapError(DataError):
    """Missing rows in a range that must be contiguous."""


class StaleDataError(DataError):
    """The freshest datapoint is older than the allowed staleness budget.

    Not fatal by itself: the feature is emitted as NaN and the model decides
    whether it can still trade. Trading on a stale order book, however, is
    gated separately in the execution filters.
    """


class LookaheadError(DataError):
    """A feature or label referenced information unavailable at decision time.

    Deliberately fatal. Leaked information produces a beautiful backtest and a
    losing account, so it must never be swallowed.
    """

    halts_trading = True


# --------------------------------------------------------------------------- #
# Market / settlement
# --------------------------------------------------------------------------- #
class MarketError(BotError):
    """Base for problems with a specific Polymarket market."""


class MarketNotFoundError(MarketError):
    """No open market matched the configured slug pattern for this window."""


class MarketClosedError(MarketError):
    """The market stopped accepting orders before we acted on it."""


class SettlementSourceMismatch(MarketError):
    """The market's stated resolution source differs from the configured one.

    This is the most dangerous silent failure in the whole system: the model
    would be forecasting one price series while the payout is decided by
    another. Always fatal for that market, never downgraded to a warning.
    """

    halts_trading = True


class UnresolvedMarketError(MarketError):
    """Settlement was requested for a market that has not resolved yet."""


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
class ModelError(BotError):
    """Base for training / inference failures."""


class ModelNotFittedError(ModelError):
    """Inference requested before the model was trained or loaded."""


class InsufficientDataError(ModelError):
    """Not enough samples to train, or to fill a feature window."""


class CalibrationError(ModelError):
    """Predicted probabilities failed their calibration gate.

    An uncalibrated probability is unusable here: position size, expected
    value, and the edge test are all functions of ``p``, not of the argmax.
    """


# --------------------------------------------------------------------------- #
# Execution / risk
# --------------------------------------------------------------------------- #
class ExecutionError(BotError):
    """Order placement, amendment, or cancellation failed."""


class InsufficientLiquidityError(ExecutionError):
    """The book cannot absorb the intended size within the slippage budget."""


class RiskLimitBreached(BotError):
    """A hard risk limit (daily loss, exposure, concurrency) was hit."""

    halts_trading = True


class TradingHaltedError(BotError):
    """The kill switch is engaged; no new orders may be sent."""

    halts_trading = True


class LiveModeNotArmedError(BotError):
    """Live trading was requested without clearing the promotion gate."""

    halts_trading = True
