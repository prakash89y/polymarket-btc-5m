"""Redaction and the decision log.

A leaked private key and a lost decision log are both unrecoverable, so both
paths get tests rather than trust.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import structlog

from pmbtc.logging_setup import (
    DecisionLog,
    configure_logging,
    get_logger,
    make_redactor,
    open_decision_log,
)

REDACT = ("api_key", "secret", "private_key", "authorization", "token")


class TestRedaction:
    def test_top_level_key(self) -> None:
        out = make_redactor(REDACT)(None, "info", {"api_key": "abc123", "n": 1})
        assert out["api_key"] == "***REDACTED***"
        assert out["n"] == 1

    def test_case_and_substring_insensitive(self) -> None:
        out = make_redactor(REDACT)(
            None, "info", {"BINANCE_API_KEY": "x", "Authorization": "Bearer y"}
        )
        assert out["BINANCE_API_KEY"] == "***REDACTED***"
        assert out["Authorization"] == "***REDACTED***"

    def test_nested_dict(self) -> None:
        out = make_redactor(REDACT)(
            None, "info", {"req": {"headers": {"authorization": "Bearer z"}, "url": "/x"}}
        )
        assert out["req"]["headers"]["authorization"] == "***REDACTED***"
        assert out["req"]["url"] == "/x"

    def test_inside_list(self) -> None:
        out = make_redactor(REDACT)(None, "info", {"items": [{"secret": "s"}, {"ok": 1}]})
        assert out["items"][0]["secret"] == "***REDACTED***"
        assert out["items"][1]["ok"] == 1

    def test_non_secret_values_untouched(self) -> None:
        payload = {"price": 0.53, "outcome": "up", "market_id": "0xabc"}
        assert make_redactor(REDACT)(None, "info", dict(payload)) == payload

    def test_deep_nesting_terminates(self) -> None:
        # Guard against pathological structures rather than recursing forever.
        deep: dict[str, object] = {"k": "v"}
        for _ in range(30):
            deep = {"nested": deep}
        make_redactor(REDACT)(None, "info", deep)


class TestConfigureLogging:
    def test_writes_json_file(self, tmp_config, capsys) -> None:  # type: ignore[no-untyped-def]
        configure_logging(tmp_config, force=True)
        get_logger("pmbtc.test").info("hello", api_key="leak-me", market="btc-5m")

        log_file = tmp_config.resolved_path(tmp_config.app.log_dir) / tmp_config.logging.file_name
        assert log_file.exists()
        record = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert record["event"] == "hello"
        assert record["market"] == "btc-5m"
        assert record["api_key"] == "***REDACTED***"
        assert "leak-me" not in log_file.read_text(encoding="utf-8")

    def test_stdlib_records_are_redacted_too(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        # Third-party libraries log through the stdlib; they must not bypass
        # the redactor.
        configure_logging(tmp_config, force=True)
        logging.getLogger("some.library").warning("boom", extra={"api_secret": "leak-me"})

        log_file = tmp_config.resolved_path(tmp_config.app.log_dir) / tmp_config.logging.file_name
        assert "leak-me" not in log_file.read_text(encoding="utf-8")

    def test_level_is_applied(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        configure_logging(tmp_config, force=True)
        assert logging.getLogger().level == logging.INFO

    def test_noisy_libraries_are_quieted(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        configure_logging(tmp_config, force=True)
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_idempotent(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        configure_logging(tmp_config, force=True)
        before = len(logging.getLogger().handlers)
        configure_logging(tmp_config)
        assert len(logging.getLogger().handlers) == before

    def test_bound_context_is_included(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        configure_logging(tmp_config, force=True)
        structlog.contextvars.bind_contextvars(window_open=1234)
        try:
            get_logger("pmbtc.test").info("ctx")
        finally:
            structlog.contextvars.clear_contextvars()

        log_file = tmp_config.resolved_path(tmp_config.app.log_dir) / tmp_config.logging.file_name
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[-1])["window_open"] == 1234


class TestDecisionLog:
    def test_appends_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        with DecisionLog(path) as log:
            log.write({"window_open": 1, "action": "skip", "reason": "low_confidence"})
            log.write({"window_open": 2, "action": "buy", "outcome": "up", "size_usdc": 12.0})

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["reason"] == "low_confidence"
        assert json.loads(lines[1])["size_usdc"] == 12.0

    def test_survives_non_serialisable_values(self, tmp_path: Path) -> None:
        # A decision record must never be lost because an exotic type crept in.
        from decimal import Decimal

        path = tmp_path / "decisions.jsonl"
        with DecisionLog(path) as log:
            log.write({"price": Decimal("0.53"), "path": tmp_path})
        assert json.loads(path.read_text(encoding="utf-8").strip())["price"] == "0.53"

    def test_redacts(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        with DecisionLog(path, redact_keys=("api_key",)) as log:
            log.write({"api_key": "leak-me", "edge": 0.05})
        assert "leak-me" not in path.read_text(encoding="utf-8")

    def test_reopening_appends_rather_than_truncates(self, tmp_path: Path) -> None:
        # This file is the training set: opening it in "w" would erase history.
        path = tmp_path / "decisions.jsonl"
        with DecisionLog(path) as log:
            log.write({"i": 1})
        with DecisionLog(path) as log:
            log.write({"i": 2})
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_open_from_config(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        tmp_config.ensure_directories()
        with open_decision_log(tmp_config) as log:
            log.write({"ok": True})
        expected = (
            tmp_config.resolved_path(tmp_config.app.log_dir)
            / tmp_config.logging.decision_log_name
        )
        assert expected.exists()
