"""Structured logging, secret redaction, and the append-only decision log.

Design decisions
----------------
1. **structlog with a stdlib backend.** Third-party libraries log through the
   stdlib; routing everything into one structlog pipeline means those records
   get the same timestamps, redaction, and formatting as ours.

2. **Redaction is a processor, not a caller responsibility.** Anything named
   like a secret is replaced before the record reaches a handler, recursively
   through nested dicts and lists, and inside repeated arguments. Relying on
   every call site to remember is how keys end up in log files.

3. **Two sinks with different lifetimes.** ``pmbtc.log`` is operational and
   rotates. ``decisions.jsonl`` is the *training set* — every evaluation, traded
   or skipped, with its features hash, probabilities, and eventual settlement —
   and is never rotated or purged. Losing it means losing the bot's memory.

4. **Skipped opportunities are logged as loudly as trades.** A model that stops
   trading is indistinguishable from a broken data feed unless the abstentions
   are recorded with their reason.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import orjson
import structlog

from pmbtc.config import Config

_DEFAULT_REDACTED = "***REDACTED***"
_configured = False


# --------------------------------------------------------------------------- #
# Processors
# --------------------------------------------------------------------------- #
def make_redactor(
    redact_keys: tuple[str, ...], placeholder: str = _DEFAULT_REDACTED
) -> Any:
    """Build a structlog processor that scrubs secret-looking keys, recursively.

    Matching is substring-based and case-insensitive, so ``BINANCE_API_KEY``,
    ``headers.Authorization``, and ``poly_api_key`` are all caught by the short
    key list in :class:`~pmbtc.config.LoggingConfig`.
    """
    needles = tuple(k.lower() for k in redact_keys)

    def _is_secret(key: str) -> bool:
        lowered = key.lower()
        return any(needle in lowered for needle in needles)

    def _scrub(value: Any, depth: int = 0) -> Any:
        if depth > 6:  # cycle / pathological nesting guard
            return value
        if isinstance(value, dict):
            return {
                k: (placeholder if isinstance(k, str) and _is_secret(k) else _scrub(v, depth + 1))
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            scrubbed = [_scrub(v, depth + 1) for v in value]
            return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
        return value

    def processor(
        _logger: Any, _method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return _scrub(event_dict)

    return processor


def _orjson_dumps(obj: Any, default: Any = None, **_: Any) -> str:
    """Fast, deterministic JSON for log records.

    ``default=str`` is the catch-all: a log line must never raise because a
    Decimal, Path, or enum wandered into the event dict.
    """
    return orjson.dumps(obj, default=default or str).decode("utf-8")


def _drop_color_message(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """uvicorn duplicates its message under ``color_message``; drop the copy."""
    event_dict.pop("color_message", None)
    return event_dict


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def configure_logging(config: Config, *, force: bool = False) -> None:
    """Install the logging pipeline. Idempotent unless ``force=True``."""
    global _configured
    if _configured and not force:
        return

    log_cfg = config.logging
    level = getattr(logging, log_cfg.level)
    redactor = make_redactor(log_cfg.redact_keys)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _drop_color_message,
        # ExtraAdder sits *before* the redactor so `logging.info(..., extra=...)`
        # from a third-party library is scrubbed on the same terms as our own
        # events. It is a no-op for native structlog calls.
        structlog.stdlib.ExtraAdder(),
        redactor,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer(serializer=_orjson_dumps)
        if log_cfg.json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_cfg.file_enabled:
        log_dir = config.resolved_path(config.app.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / log_cfg.file_name,
            maxBytes=log_cfg.rotate_mb * 1024 * 1024,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        # The file sink is always JSON: it is read by tooling, not by humans.
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(serializer=_orjson_dumps),
                ],
            )
        )
        root.addHandler(file_handler)

    root.setLevel(level)
    _quiet_noisy_libraries()
    _configured = True


def _quiet_noisy_libraries() -> None:
    """Third-party DEBUG output would bury our own at 4 predictions/minute."""
    for name in ("httpx", "httpcore", "websockets", "urllib3", "asyncio", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str = "pmbtc") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.stdlib.get_logger(name)


# --------------------------------------------------------------------------- #
# Decision log
# --------------------------------------------------------------------------- #
class DecisionLog:
    """Append-only JSONL record of every decision the bot makes.

    One line per evaluated market window, whether or not it was traded. Later
    modules append the settlement outcome by market id, so the file is both the
    audit trail and the raw material for retraining. It is deliberately not a
    rotating handler: this file is the bot's memory.

    Writes are line-buffered and flushed per record. At one record every few
    seconds the cost is irrelevant, and a crash must not lose the decision that
    preceded it.
    """

    def __init__(self, path: Path, *, redact_keys: tuple[str, ...] = ()) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._redactor = make_redactor(redact_keys) if redact_keys else None
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        payload = self._redactor(None, "info", dict(record)) if self._redactor else record
        self._fh.write(orjson.dumps(payload, default=str).decode("utf-8") + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> DecisionLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_decision_log(config: Config) -> DecisionLog:
    """Open the configured decision log."""
    log_dir = config.resolved_path(config.app.log_dir)
    return DecisionLog(
        log_dir / config.logging.decision_log_name,
        redact_keys=config.logging.redact_keys,
    )


def reset_logging() -> None:
    """Test helper: allow :func:`configure_logging` to run again."""
    global _configured
    _configured = False
    structlog.reset_defaults()
