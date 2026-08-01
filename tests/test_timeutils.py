"""Window arithmetic is load-bearing: a one-window offset mislabels the dataset."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

import pytest

from pmbtc.constants import Session, WindowPhase
from pmbtc.exceptions import ConfigError
from pmbtc.utils import timeutils as tu

# 2026-07-31 12:07:30 UTC — 2m30s into the 12:05 window.
MID_WINDOW = tu.to_ms(datetime(2026, 7, 31, 12, 7, 30, tzinfo=UTC))
WINDOW_OPEN = tu.to_ms(datetime(2026, 7, 31, 12, 5, 0, tzinfo=UTC))
WINDOW_CLOSE = tu.to_ms(datetime(2026, 7, 31, 12, 10, 0, tzinfo=UTC))


class TestConversion:
    def test_interval_parsing(self) -> None:
        assert tu.interval_to_ms("5m") == 300_000
        assert tu.interval_to_ms("1s") == 1_000
        assert tu.interval_to_ms("1h") == 3_600_000

    @pytest.mark.parametrize("bad", ["", "5", "m", "0m", "-5m", "5min", "5M5"])
    def test_bad_intervals_raise(self, bad: str) -> None:
        with pytest.raises(ConfigError):
            tu.interval_to_ms(bad)

    def test_naive_datetime_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            tu.to_ms(datetime(2026, 7, 31, 12, 0, 0))

    def test_bool_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            tu.to_ms(True)

    def test_iso_z_suffix(self) -> None:
        assert tu.to_ms("2026-07-31T12:05:00Z") == WINDOW_OPEN

    def test_bare_iso_treated_as_utc(self) -> None:
        # Gamma occasionally omits the offset; the docs say UTC.
        assert tu.to_ms("2026-07-31T12:05:00") == WINDOW_OPEN

    def test_seconds_vs_millis_heuristic(self) -> None:
        assert tu.to_ms(WINDOW_OPEN // 1000) == WINDOW_OPEN
        assert tu.to_ms(WINDOW_OPEN) == WINDOW_OPEN

    def test_roundtrip(self) -> None:
        assert tu.to_ms(tu.to_datetime(MID_WINDOW)) == MID_WINDOW

    def test_isoformat(self) -> None:
        assert tu.isoformat(WINDOW_OPEN) == "2026-07-31T12:05:00.000Z"


class TestWindows:
    def test_open_and_close(self) -> None:
        assert tu.window_open(MID_WINDOW) == WINDOW_OPEN
        assert tu.window_close(WINDOW_OPEN) == WINDOW_CLOSE

    def test_open_is_idempotent_on_boundary(self) -> None:
        assert tu.window_open(WINDOW_OPEN) == WINDOW_OPEN

    def test_close_boundary_belongs_to_next_window(self) -> None:
        # Half-open [open, close): the settlement instant starts the next window.
        assert tu.window_open(WINDOW_CLOSE) == WINDOW_CLOSE

    def test_current_window(self) -> None:
        assert tu.current_window(MID_WINDOW) == (WINDOW_OPEN, WINDOW_CLOSE)

    def test_time_accounting_sums_to_window(self) -> None:
        assert tu.ms_into_window(MID_WINDOW) == 150_000
        assert tu.ms_to_settlement(MID_WINDOW) == 150_000
        assert tu.ms_into_window(MID_WINDOW) + tu.ms_to_settlement(MID_WINDOW) == tu.WINDOW_MS

    def test_seconds_to_settlement(self) -> None:
        assert tu.seconds_to_settlement(MID_WINDOW) == 150.0

    def test_progress(self) -> None:
        assert tu.window_progress(MID_WINDOW) == pytest.approx(0.5)
        assert tu.window_progress(WINDOW_OPEN) == 0.0

    def test_next_window_open_is_strictly_future(self) -> None:
        assert tu.next_window_open(WINDOW_OPEN) == WINDOW_CLOSE
        assert tu.next_window_open(MID_WINDOW) == WINDOW_CLOSE

    def test_align_windows(self) -> None:
        opens = list(tu.align_windows(WINDOW_OPEN, WINDOW_OPEN + 3 * tu.WINDOW_MS))
        assert opens == [
            WINDOW_OPEN,
            WINDOW_OPEN + tu.WINDOW_MS,
            WINDOW_OPEN + 2 * tu.WINDOW_MS,
        ]

    def test_custom_window_length(self) -> None:
        one_minute = 60_000
        assert tu.window_open(MID_WINDOW, one_minute) == tu.to_ms(
            datetime(2026, 7, 31, 12, 7, 0, tzinfo=UTC)
        )


class TestPhase:
    @pytest.mark.parametrize(
        ("offset_s", "expected"),
        [
            (0, WindowPhase.OPENING),
            (59, WindowPhase.OPENING),
            (60, WindowPhase.MID),
            (179, WindowPhase.MID),
            (180, WindowPhase.LATE),
            (269, WindowPhase.LATE),
            (270, WindowPhase.FINAL),
            (299, WindowPhase.FINAL),
        ],
    )
    def test_phase_boundaries(self, offset_s: int, expected: WindowPhase) -> None:
        assert tu.phase_of_window(WINDOW_OPEN + offset_s * 1000) is expected

    def test_final_beats_opening_on_degenerate_config(self) -> None:
        # With a 30s window and a 60s "opening", FINAL must still win: the
        # execution gate depends on it and the cost of being wrong is a fill
        # that cannot be managed.
        ms = WINDOW_OPEN + 5_000
        assert tu.phase_of_window(ms, 30_000, opening_s=60.0, final_s=30.0) is WindowPhase.FINAL


class TestBars:
    def test_last_closed_bar_excludes_forming_bar(self) -> None:
        # The bar containing MID_WINDOW is still forming and must not be read.
        assert tu.last_closed_bar_open(60_000, MID_WINDOW) == tu.to_ms(
            datetime(2026, 7, 31, 12, 6, 0, tzinfo=UTC)
        )

    def test_is_bar_closed(self) -> None:
        assert not tu.is_bar_closed(WINDOW_OPEN, tu.WINDOW_MS, MID_WINDOW)
        assert tu.is_bar_closed(WINDOW_OPEN, tu.WINDOW_MS, WINDOW_CLOSE)

    def test_bar_count(self) -> None:
        assert tu.bar_count(WINDOW_OPEN, WINDOW_CLOSE, 60_000) == 5
        assert tu.bar_count(WINDOW_CLOSE, WINDOW_OPEN, 60_000) == 0

    def test_chunk_range_covers_without_overlap(self) -> None:
        start, end = WINDOW_OPEN, WINDOW_OPEN + 10 * 60_000
        chunks = list(tu.chunk_range(start, end, 60_000, limit=4))
        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        # Contiguous and non-overlapping: a gap silently drops candles from a
        # backfill, an overlap double-counts them.
        for (_, prev_end), (next_start, _) in pairwise(chunks):
            assert prev_end == next_start

    def test_chunk_range_rejects_bad_limit(self) -> None:
        with pytest.raises(ConfigError):
            list(tu.chunk_range(0, 10, 1, limit=0))


class TestCalendar:
    def test_session_labels(self) -> None:
        def at(hour: int) -> int:
            return tu.to_ms(datetime(2026, 7, 31, hour, 0, 0, tzinfo=UTC))

        assert tu.session_of(at(3)) is Session.ASIA
        assert tu.session_of(at(9)) is Session.LONDON
        assert tu.session_of(at(14)) is Session.LONDON_NY_OVERLAP
        assert tu.session_of(at(18)) is Session.NEW_YORK
        assert tu.session_of(at(23)) is Session.LATE_US

    def test_weekend(self) -> None:
        friday = tu.to_ms(datetime(2026, 7, 31, 12, 0, tzinfo=UTC))
        saturday = tu.to_ms(datetime(2026, 8, 1, 12, 0, tzinfo=UTC))
        assert not tu.is_weekend(friday)
        assert tu.is_weekend(saturday)

    def test_week_start_is_monday(self) -> None:
        monday = tu.to_datetime(tu.utc_week_start(MID_WINDOW))
        assert monday.weekday() == 0
        assert (monday.hour, monday.minute, monday.second) == (0, 0, 0)

    def test_day_start(self) -> None:
        assert tu.to_datetime(tu.utc_day_start(MID_WINDOW)).hour == 0

    def test_minute_of_day(self) -> None:
        assert tu.minute_of_day(MID_WINDOW) == 12 * 60 + 7
