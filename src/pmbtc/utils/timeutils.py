"""UTC millisecond time and 5-minute-window arithmetic.

Design decisions
----------------
1. **One internal time type: ``int`` milliseconds since the Unix epoch, UTC.**
   Every venue in scope speaks epoch millis. Converting to ``datetime`` only at
   the boundaries removes a whole class of naive/aware bugs and keeps window
   arithmetic exact — no float drift over a year-long backfill.

2. **Naive datetimes are rejected, never assumed UTC.** Silently treating a
   local-time value as UTC would shift a label by hours and quietly poison the
   training set.

3. **A window is identified by its OPEN time**, and boundaries are half-open
   ``[open, close)``. Polymarket's 5-minute Up/Down markets compare the
   settlement price at ``close`` against the reference price at ``open``, so
   these two instants are the only ones that matter for the label — everything
   else in the pipeline is a feature about the path between them.

4. **"Is this bar closed?" is always explicit.** The bar currently forming may
   never be read by a feature. That single rule prevents the most common form of
   look-ahead leakage in intraday models.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from pmbtc.constants import Session, WindowPhase
from pmbtc.exceptions import ConfigError

MS: int = 1
SECOND_MS: int = 1_000
MINUTE_MS: int = 60 * SECOND_MS
HOUR_MS: int = 60 * MINUTE_MS
DAY_MS: int = 24 * HOUR_MS
WEEK_MS: int = 7 * DAY_MS

#: The instrument this bot exists for.
WINDOW_MS: int = 5 * MINUTE_MS

_UNIT_MS: dict[str, int] = {
    "ms": MS,
    "s": SECOND_MS,
    "m": MINUTE_MS,
    "h": HOUR_MS,
    "d": DAY_MS,
    "w": WEEK_MS,
}

_INTERVAL_RE = re.compile(r"^(?P<value>\d+)(?P<unit>ms|s|m|h|d|w)$")


# --------------------------------------------------------------------------- #
# Parsing / conversion
# --------------------------------------------------------------------------- #
def interval_to_ms(interval: str) -> int:
    """Convert an exchange-style interval string (``"5m"``, ``"1h"``) to millis."""
    match = _INTERVAL_RE.match(interval.strip().lower())
    if match is None:
        raise ConfigError(
            f"Unrecognised interval {interval!r}; expected e.g. '1s', '1m', '5m', '1h', '1d'"
        )
    value = int(match.group("value"))
    if value <= 0:
        raise ConfigError(f"Interval must be positive, got {interval!r}")
    return value * _UNIT_MS[match.group("unit")]


def utc_now_ms() -> int:
    """Current wall-clock time in epoch millis (UTC)."""
    return int(time.time() * 1000)


def to_ms(value: datetime | int | float | str) -> int:
    """Coerce a datetime / epoch number / ISO-8601 string to epoch millis.

    Naive ``datetime`` objects raise: see the module docstring. ISO strings
    without an offset are the one exception — Polymarket's Gamma API emits
    ``2026-07-31T12:05:00Z`` and occasionally bare ``...T12:05:00``, both of
    which are documented UTC.
    """
    if isinstance(value, bool):  # bool is an int subclass; always a bug here
        raise ConfigError("bool is not a valid timestamp")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ConfigError("Naive datetime rejected; attach tzinfo (datetime.UTC) explicitly")
        return int(value.astimezone(UTC).timestamp() * 1000)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ConfigError(f"Unparseable timestamp {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)
    # Heuristic: values below ~1e11 are seconds, above are already millis.
    numeric = float(value)
    return int(numeric * 1000) if abs(numeric) < 1e11 else int(numeric)


def to_datetime(ms: int) -> datetime:
    """Convert epoch millis to a tz-aware UTC ``datetime``."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def isoformat(ms: int) -> str:
    """Log-friendly ISO-8601 rendering of epoch millis."""
    return to_datetime(ms).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Generic interval boundaries (candles, resample buckets)
# --------------------------------------------------------------------------- #
def floor_to_interval(ms: int, interval_ms: int) -> int:
    """Snap a timestamp down to the open time of the interval containing it."""
    if interval_ms <= 0:
        raise ConfigError("interval_ms must be positive")
    return (ms // interval_ms) * interval_ms  # floor division: correct for negatives too


def ceil_to_interval(ms: int, interval_ms: int) -> int:
    """Snap a timestamp up to the next interval open (identity if aligned)."""
    floored = floor_to_interval(ms, interval_ms)
    return floored if floored == ms else floored + interval_ms


def is_bar_closed(open_ms: int, interval_ms: int, now_ms: int | None = None) -> bool:
    """True once the bar opening at ``open_ms`` has fully elapsed."""
    now = utc_now_ms() if now_ms is None else now_ms
    return now >= open_ms + interval_ms


def last_closed_bar_open(interval_ms: int, now_ms: int | None = None) -> int:
    """Open time of the most recent fully closed bar.

    The only bar a feature pipeline may read. ``floor(now)`` is still forming.
    """
    now = utc_now_ms() if now_ms is None else now_ms
    return floor_to_interval(now, interval_ms) - interval_ms


def iter_interval_opens(start_ms: int, end_ms: int, interval_ms: int) -> Iterator[int]:
    """Yield every aligned interval open in the half-open range ``[start, end)``."""
    cursor = ceil_to_interval(start_ms, interval_ms)
    while cursor < end_ms:
        yield cursor
        cursor += interval_ms


def bar_count(start_ms: int, end_ms: int, interval_ms: int) -> int:
    """Number of aligned interval opens in ``[start_ms, end_ms)``."""
    if end_ms <= start_ms:
        return 0
    first = ceil_to_interval(start_ms, interval_ms)
    if first >= end_ms:
        return 0
    return (end_ms - 1 - first) // interval_ms + 1


def chunk_range(
    start_ms: int, end_ms: int, interval_ms: int, limit: int
) -> Iterator[tuple[int, int]]:
    """Split ``[start, end)`` into venue-sized backfill windows.

    REST kline endpoints cap at 500-1500 rows per call, so every historical
    loader needs this. Yields half-open ``(chunk_start, chunk_end)`` pairs.
    """
    if limit <= 0:
        raise ConfigError("limit must be positive")
    if interval_ms <= 0:
        raise ConfigError("interval_ms must be positive")
    span = interval_ms * limit
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + span, end_ms)
        yield cursor, chunk_end
        cursor = chunk_end


# --------------------------------------------------------------------------- #
# Polymarket 5-minute window arithmetic
# --------------------------------------------------------------------------- #
def window_open(ms: int, window_ms: int = WINDOW_MS) -> int:
    """Open time of the settlement window containing ``ms``."""
    return floor_to_interval(ms, window_ms)


def window_close(open_ms: int, window_ms: int = WINDOW_MS) -> int:
    """Exclusive close (== settlement instant) of the window opening at ``open_ms``.

    Exclusive boundaries make window arithmetic associative; convert to a venue's
    inclusive ``closeTime`` (``close - 1ms``) only when talking to that venue.
    """
    return open_ms + window_ms


def current_window(ms: int | None = None, window_ms: int = WINDOW_MS) -> tuple[int, int]:
    """``(open_ms, close_ms)`` of the window in progress."""
    now = utc_now_ms() if ms is None else ms
    open_ms = window_open(now, window_ms)
    return open_ms, window_close(open_ms, window_ms)


def next_window_open(ms: int | None = None, window_ms: int = WINDOW_MS) -> int:
    """Open time of the next window to start (strictly in the future)."""
    now = utc_now_ms() if ms is None else ms
    return window_open(now, window_ms) + window_ms


def ms_into_window(ms: int | None = None, window_ms: int = WINDOW_MS) -> int:
    """Elapsed millis since the current window opened."""
    now = utc_now_ms() if ms is None else ms
    return now - window_open(now, window_ms)


def ms_to_settlement(ms: int | None = None, window_ms: int = WINDOW_MS) -> int:
    """Millis remaining until the current window settles."""
    now = utc_now_ms() if ms is None else ms
    return window_close(window_open(now, window_ms), window_ms) - now


def seconds_to_settlement(ms: int | None = None, window_ms: int = WINDOW_MS) -> float:
    """Seconds remaining until the current window settles."""
    return ms_to_settlement(ms, window_ms) / 1000.0


def window_progress(ms: int | None = None, window_ms: int = WINDOW_MS) -> float:
    """Fraction of the window elapsed, in ``[0, 1)``.

    Used directly as a feature and as the ``t`` in the time-scaling of realized
    volatility: remaining uncertainty scales with ``sqrt(1 - progress)``.
    """
    if window_ms <= 0:
        raise ConfigError("window_ms must be positive")
    return ms_into_window(ms, window_ms) / window_ms


def phase_of_window(
    ms: int | None = None,
    window_ms: int = WINDOW_MS,
    *,
    opening_s: float = 60.0,
    final_s: float = 30.0,
    late_s: float = 120.0,
) -> WindowPhase:
    """Classify where we are inside the window.

    Thresholds are seconds measured from the relevant edge: ``opening_s`` after
    the open, ``final_s`` and ``late_s`` before the close. Defaults are sane for
    a 300-second window; the live loop passes the configured values.
    """
    elapsed_s = ms_into_window(ms, window_ms) / 1000.0
    remaining_s = (window_ms / 1000.0) - elapsed_s
    if remaining_s <= final_s:
        return WindowPhase.FINAL
    if elapsed_s < opening_s:
        return WindowPhase.OPENING
    if remaining_s <= late_s:
        return WindowPhase.LATE
    return WindowPhase.MID


def align_windows(start_ms: int, end_ms: int, window_ms: int = WINDOW_MS) -> Iterator[int]:
    """Yield every window open in ``[start, end)`` — the backtest's outer loop."""
    yield from iter_interval_opens(start_ms, end_ms, window_ms)


# --------------------------------------------------------------------------- #
# Calendar helpers (risk resets, regime features)
# --------------------------------------------------------------------------- #
def utc_day_start(ms: int) -> int:
    """Epoch millis of 00:00:00 UTC on the day containing ``ms``."""
    return floor_to_interval(ms, DAY_MS)


def utc_week_start(ms: int) -> int:
    """Epoch millis of Monday 00:00:00 UTC for the ISO week containing ``ms``."""
    day = to_datetime(utc_day_start(ms))
    monday = day - timedelta(days=day.weekday())
    return to_ms(monday)


def is_weekend(ms: int) -> bool:
    """Saturday or Sunday in UTC.

    Crypto never closes, but weekend liquidity is thinner and the realized-vol
    regime is measurably different — worth its own feature.
    """
    return to_datetime(ms).weekday() >= 5


def session_of(ms: int) -> Session:
    """Coarse trading-session label (UTC hour buckets)."""
    hour = to_datetime(ms).hour
    if hour < 7:
        return Session.ASIA
    if hour < 12:
        return Session.LONDON
    if hour < 16:
        return Session.LONDON_NY_OVERLAP
    if hour < 21:
        return Session.NEW_YORK
    return Session.LATE_US


def minute_of_day(ms: int) -> int:
    """Minutes since 00:00 UTC — the raw input for cyclical time encodings."""
    dt = to_datetime(ms)
    return dt.hour * 60 + dt.minute
