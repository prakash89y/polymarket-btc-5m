"""Build the committed CLOB tick fixture from a real live archive.

Run on a machine that has collected data. The output is a small, real slice of
the CLOB stream, committed so that replay determinism can be verified on any
clean checkout — including a CI runner that has never collected anything.

Kept deliberately small (under the 512 KB repository hygiene limit) while still
containing every event type the parser handles: a full `book` snapshot for both
outcome tokens, incremental `price_change` batches, and executed
`last_trade_price` prints. A fixture that exercises only one code path proves
only one code path.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "raw" / "ticks"
TARGET = ROOT / "tests" / "fixtures" / "ticks"
MAX_BYTES = 400 * 1024


def build() -> int:
    sources = sorted(LIVE.rglob("*clob*.jsonl.gz"), key=lambda p: p.stat().st_size)
    if not sources:
        print("No live CLOB archive found. Run the collector first.")
        return 1

    source = sources[0]
    print(f"source: {source.relative_to(ROOT)} ({source.stat().st_size / 1024:.0f} KB)")

    frames: list[bytes] = []
    seen_types: set[str] = set()
    tokens: set[str] = set()
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            payload = record.get("p")
            for event in payload if isinstance(payload, list) else [payload]:
                if isinstance(event, dict):
                    seen_types.add(str(event.get("event_type", "")))
                    if event.get("event_type") == "book":
                        tokens.add(str(event.get("asset_id", "")))
            frames.append(line.rstrip("\n").encode("utf-8"))
            # Stop once the fixture is big enough *and* covers every event type
            # for both tokens — size alone would risk a fixture that never sees
            # a trade print.
            body = sum(len(f) + 1 for f in frames)
            if body > MAX_BYTES * 4 and len(tokens) >= 2 and len(seen_types) >= 3:
                break

    TARGET.mkdir(parents=True, exist_ok=True)
    out = TARGET / "clob_sample.jsonl.gz"
    # mtime=0 so the gzip header is byte-identical on every rebuild; otherwise
    # regenerating the fixture would show as a spurious diff.
    with gzip.GzipFile(filename="", mode="wb", fileobj=out.open("wb"),
                       compresslevel=9, mtime=0) as gz:
        gz.write(b"\n".join(frames) + b"\n")

    size = out.stat().st_size
    print(f"wrote {out.relative_to(ROOT)} ({size / 1024:.0f} KB, {len(frames)} frames)")
    print(f"  event types: {sorted(t for t in seen_types if t)}")
    print(f"  tokens: {len(tokens)}")
    if size > MAX_BYTES:
        print(f"  WARNING: exceeds {MAX_BYTES / 1024:.0f} KB budget")
        return 1
    if len(tokens) < 2 or len(seen_types) < 3:
        print("  ERROR: fixture does not cover both tokens and all event types")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(build())
