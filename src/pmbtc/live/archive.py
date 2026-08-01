"""Raw stream archive — the price of using live features at all.

Rule 5 of the agreed architecture: a feature used in live prediction must be
reproducible from archived data, or excluded from training. Order-book imbalance
and CVD exist only for the instant they are computed; nothing re-serves them
later. So the raw frames are archived as they arrive, and every live-derived
feature is declared ``ARCHIVED_STREAM`` — reproducible exactly as long as this
archive is kept.

Written as gzipped JSONL, one file per feed per UTC hour. At the observed
~155 frames/second a naive uncompressed archive would be tens of gigabytes a
day; these frames compress extremely well because consecutive book snapshots are
nearly identical.

Writes are buffered and flushed on a size threshold rather than per frame: at
this rate a flush per frame is the difference between negligible and dominant
overhead. The buffer is flushed on close and on rotation, so a clean shutdown
loses nothing and a crash loses at most one buffer.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, TextIO

import orjson

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.utils.timeutils import isoformat, utc_now_ms

log = get_logger("pmbtc.live.archive")

frames_archived = METRICS.counter(
    "pmbtc_frames_archived_total", "Raw stream frames written to the archive."
)
archive_bytes = METRICS.counter("pmbtc_archive_bytes_total", "Bytes written to the archive.")


class TickArchive:
    """Append-only gzipped archive of raw feed frames."""

    def __init__(self, root: Path, *, enabled: bool = True, buffer_frames: int = 200) -> None:
        self.root = root
        self.enabled = enabled
        self.buffer_frames = max(1, buffer_frames)
        self._handles: dict[str, TextIO] = {}
        self._hours: dict[str, str] = {}
        self._buffers: dict[str, list[str]] = {}
        #: Timestamp of the newest buffered frame, per feed. Flushing must
        #: partition by the *data's* time, never by wall-clock time — otherwise
        #: a flush at an hour boundary drops the tail of the buffer into the
        #: wrong file and archive replay loses chronological order.
        self._last_ms: dict[str, int] = {}
        self.frames_written = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _hour_key(self, ms: int) -> str:
        return isoformat(ms)[:13].replace("T", "-")  # 2026-07-31-17

    def _handle(self, feed: str, ms: int) -> TextIO:
        hour = self._hour_key(ms)
        if self._hours.get(feed) != hour:
            self._close_feed(feed)
            safe = feed.replace(":", "_").replace("/", "_")
            directory = self.root / hour[:10]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{safe}-{hour}.jsonl.gz"
            # SIM115: the handle is deliberately long-lived and owned by this
            # object; it is closed on rotation and in close().
            self._handles[feed] = gzip.open(  # noqa: SIM115
                path, "at", encoding="utf-8", compresslevel=6
            )
            self._hours[feed] = hour
        return self._handles[feed]

    def write(self, feed: str, payload: Any, received_ms: int | None = None) -> None:
        if not self.enabled:
            return
        received = received_ms if received_ms is not None else utc_now_ms()
        line = orjson.dumps(
            {"t": received, "p": payload}, default=str
        ).decode("utf-8")
        buffer = self._buffers.setdefault(feed, [])
        buffer.append(line)
        self._last_ms[feed] = received
        self.frames_written += 1
        frames_archived.inc(feed=feed)
        if len(buffer) >= self.buffer_frames:
            self._flush_feed(feed, received)

    def _flush_feed(self, feed: str, ms: int | None = None) -> None:
        buffer = self._buffers.get(feed)
        if not buffer:
            return
        # Partition by the newest buffered frame's own timestamp, falling back
        # to wall clock only when nothing has ever been written for this feed.
        when = ms if ms is not None else self._last_ms.get(feed, utc_now_ms())
        handle = self._handle(feed, when)
        blob = "\n".join(buffer) + "\n"
        handle.write(blob)
        handle.flush()
        archive_bytes.inc(len(blob), feed=feed)
        buffer.clear()

    def flush(self) -> None:
        for feed in list(self._buffers):
            self._flush_feed(feed)

    def _close_feed(self, feed: str) -> None:
        handle = self._handles.pop(feed, None)
        if handle is not None:
            handle.close()
        self._hours.pop(feed, None)

    def close(self) -> None:
        self.flush()
        for feed in list(self._handles):
            self._close_feed(feed)

    def __enter__(self) -> TickArchive:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def files(self, day: str | None = None) -> list[Path]:
        if not self.root.exists():
            return []
        directories = (
            [self.root / day] if day else [p for p in self.root.iterdir() if p.is_dir()]
        )
        found: list[Path] = []
        for directory in directories:
            if directory.exists():
                found.extend(sorted(directory.glob("*.jsonl.gz")))
        return found

    def read_frames(self, path: Path) -> list[tuple[int, Any]]:
        """Load an archived file back as ``(received_ms, payload)`` pairs.

        This is what makes a live feature reproducible: the same frames, in the
        same order, replayed through the same book and tape code.
        """
        frames: list[tuple[int, Any]] = []
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        # A torn final line is expected while a file is still
                        # being appended to; skip it rather than losing the file.
                        continue
                    frames.append((int(record["t"]), record["p"]))
        except EOFError:
            # The current hour's file has no end-of-stream marker until the
            # collector rotates it. Reading it live is normal, not an error —
            # return the frames decoded so far.
            log.debug("archive.partial", path=str(path), frames=len(frames))
        except OSError as exc:
            log.warning("archive.unreadable", path=str(path), error=str(exc))
        return frames


def open_tick_archive(config: Config) -> TickArchive:
    return TickArchive(
        config.resolved_path(config.feeds.archive_dir),
        enabled=config.feeds.archive_ticks,
        buffer_frames=config.feeds.archive_buffer_frames,
    )
