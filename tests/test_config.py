"""Configuration is the last line of defence for several unrecoverable mistakes.

The tests that matter most here are the ones asserting that a *plausible-looking
edit* to config.yaml is rejected: disabled settlement verification, an edge
threshold below trading costs, or live mode without the interlocks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from pmbtc.config import (
    LIVE_INTERLOCK_ENV,
    BotSecrets,
    load_config,
    load_yaml,
)
from pmbtc.constants import RunMode, SettlementSource
from pmbtc.exceptions import ConfigError, MissingCredentialError

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


class TestShippedConfig:
    def test_shipped_config_is_valid(self, shipped_config: object) -> None:
        assert shipped_config is not None

    def test_defaults_are_safe(self, shipped_config) -> None:  # type: ignore[no-untyped-def]
        assert shipped_config.app.mode is RunMode.PAPER
        assert shipped_config.live.enabled is False
        assert shipped_config.live.dry_run is True
        assert shipped_config.is_live is False
        assert shipped_config.settlement.require_verified is True
        assert shipped_config.settlement.allow_unknown_source is False

    def test_window_length_matches_the_instrument(self, shipped_config) -> None:  # type: ignore[no-untyped-def]
        assert shipped_config.app.window_seconds == 300
        assert shipped_config.window_ms == 300_000

    def test_yaml_sections_do_not_silently_drop_urls(self, shipped_config) -> None:  # type: ignore[no-untyped-def]
        # A YAML section replaces the code default wholesale, so an entry that
        # omits its base_url would leave an empty string and fail at runtime.
        assert shipped_config.venues.binance_spot.rest_base_url.startswith("https://")
        assert shipped_config.sources.fear_greed.base_url.startswith("https://")
        assert shipped_config.polymarket.gamma_base_url.startswith("https://")


class TestPrecedence:
    def test_env_beats_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PMBTC_PREDICTION__MIN_EDGE", "0.08")
        cfg = load_config(SHIPPED_CONFIG)
        assert cfg.prediction.min_edge == pytest.approx(0.08)

    def test_yaml_beats_defaults(self) -> None:
        cfg = load_config(SHIPPED_CONFIG)
        assert cfg.sizing.kelly_fraction == pytest.approx(0.25)

    def test_programmatic_override(self) -> None:
        cfg = load_config(SHIPPED_CONFIG, app={"mode": "backtest"})
        assert cfg.app.mode is RunMode.BACKTEST


class TestYamlLoading:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert load_yaml(tmp_path / "nope.yaml") == {}

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("app: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_yaml(bad)

    def test_non_mapping_root_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_yaml(bad)

    def test_env_interpolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_DB", "sqlite+aiosqlite:///data/x.db")
        path = tmp_path / "c.yaml"
        path.write_text("storage:\n  db_url: ${SOME_DB}\n", encoding="utf-8")
        assert load_config(path).storage.db_url == "sqlite+aiosqlite:///data/x.db"

    def test_env_interpolation_default(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("storage:\n  db_url: ${NOT_SET_ANYWHERE:-fallback}\n", encoding="utf-8")
        assert load_config(path).storage.db_url == "fallback"

    def test_undefined_env_reference_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("storage:\n  db_url: ${DEFINITELY_NOT_SET}\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        # extra="forbid": a typo must fail loudly, not be silently ignored.
        path = tmp_path / "c.yaml"
        path.write_text("risk:\n  max_dialy_loss_usdc: 10\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


class TestSettlementGates:
    """The single most important set of assertions in Module 1."""

    def test_verification_cannot_be_disabled_in_paper(self) -> None:
        with pytest.raises(ConfigError, match="require_verified"):
            load_config(
                SHIPPED_CONFIG,
                app={"mode": "paper"},
                settlement={"require_verified": False},
            )

    def test_verification_cannot_be_disabled_in_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_INTERLOCK_ENV, "yes")
        with pytest.raises(ConfigError, match="require_verified"):
            load_config(
                SHIPPED_CONFIG,
                app={"mode": "live"},
                live={"enabled": True},
                settlement={"require_verified": False},
            )

    def test_unknown_source_cannot_be_allowed_when_trading(self) -> None:
        with pytest.raises(ConfigError, match="allow_unknown_source"):
            load_config(
                SHIPPED_CONFIG,
                app={"mode": "paper"},
                settlement={"allow_unknown_source": True},
            )

    def test_backtest_may_relax_verification(self) -> None:
        # Reconstructing history sometimes means accepting an unlabelled source;
        # that is acceptable only when no money can move.
        cfg = load_config(
            SHIPPED_CONFIG,
            app={"mode": "backtest"},
            settlement={"require_verified": False, "allow_unknown_source": True},
        )
        assert cfg.settlement.require_verified is False

    def test_expected_source_must_have_a_provider(self) -> None:
        with pytest.raises(ConfigError, match="provider"):
            load_config(
                SHIPPED_CONFIG,
                settlement={
                    "expected_source": "chainlink",
                    "providers": {"binance_spot": {"enabled": True, "symbol": "BTCUSDT"}},
                },
            )

    def test_expected_source_provider_must_be_enabled(self) -> None:
        with pytest.raises(ConfigError, match="disabled provider"):
            load_config(SHIPPED_CONFIG, settlement={"expected_source": "pyth"})

    def test_expected_source_may_not_be_unknown(self) -> None:
        with pytest.raises(ConfigError):
            load_config(
                SHIPPED_CONFIG,
                settlement={
                    "expected_source": "unknown",
                    "providers": {"unknown": {"enabled": True}},
                },
            )

    def test_detection_confidence_defaults_to_certainty(self, shipped_config) -> None:  # type: ignore[no-untyped-def]
        assert shipped_config.settlement.min_detection_confidence == 1.0
        # Verified against live Gamma payloads: the BTC 5m family resolves off
        # Chainlink, not Binance (which is the hourly family's source).
        assert shipped_config.settlement.expected_source is SettlementSource.CHAINLINK


class TestLiveInterlocks:
    def test_live_requires_live_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_INTERLOCK_ENV, "yes")
        with pytest.raises(ConfigError, match=re.escape("live.enabled")):
            load_config(SHIPPED_CONFIG, app={"mode": "live"})

    def test_live_requires_env_interlock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_INTERLOCK_ENV, raising=False)
        with pytest.raises(ConfigError, match="interlocked"):
            load_config(SHIPPED_CONFIG, app={"mode": "live"}, live={"enabled": True})

    def test_live_with_all_gates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_INTERLOCK_ENV, "yes")
        cfg = load_config(
            SHIPPED_CONFIG,
            app={"mode": "live"},
            live={"enabled": True, "dry_run": False},
        )
        assert cfg.is_live is True

    def test_dry_run_still_blocks_real_orders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_INTERLOCK_ENV, "yes")
        cfg = load_config(
            SHIPPED_CONFIG, app={"mode": "live"}, live={"enabled": True, "dry_run": True}
        )
        assert cfg.is_live is False


class TestEconomicInvariants:
    def test_edge_below_round_trip_cost_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="round-trip"):
            load_config(SHIPPED_CONFIG, prediction={"min_edge": 0.001})

    def test_confidence_gate_must_bind(self) -> None:
        with pytest.raises(ConfigError, match="min_confidence"):
            load_config(SHIPPED_CONFIG, prediction={"min_edge": 0.10, "min_confidence": 0.58})

    def test_timing_gates_must_leave_a_tradeable_interval(self) -> None:
        with pytest.raises(ConfigError, match="tradeable interval"):
            load_config(
                SHIPPED_CONFIG,
                execution={"min_seconds_into_window": 280, "min_seconds_to_settlement": 30},
            )

    def test_slippage_gate_must_be_tighter_than_spread(self) -> None:
        with pytest.raises(ConfigError, match="max_slippage"):
            load_config(SHIPPED_CONFIG, execution={"max_spread": 0.01, "max_slippage": 0.02})

    def test_prediction_cadence_must_fit_the_window(self) -> None:
        with pytest.raises(ConfigError, match="predict_every_seconds"):
            load_config(SHIPPED_CONFIG, prediction={"predict_every_seconds": 600})

    def test_weekly_loss_must_exceed_daily(self) -> None:
        with pytest.raises(ConfigError, match="max_weekly_loss_fraction"):
            load_config(
                SHIPPED_CONFIG,
                risk={"max_daily_loss_fraction": 0.10, "max_weekly_loss_fraction": 0.05},
            )

    def test_position_bounds_must_be_ordered(self) -> None:
        with pytest.raises(ConfigError, match="min_position_usdc"):
            load_config(SHIPPED_CONFIG, sizing={"min_position_usdc": 100, "max_position_usdc": 50})

    def test_order_bounds_must_be_ordered(self) -> None:
        with pytest.raises(ConfigError, match="max_order_usdc"):
            load_config(SHIPPED_CONFIG, polymarket={"min_order_usdc": 10, "max_order_usdc": 5})

    def test_kelly_fraction_capped_at_full_kelly(self) -> None:
        with pytest.raises(ConfigError):
            load_config(SHIPPED_CONFIG, sizing={"kelly_fraction": 1.5})

    def test_risk_per_trade_capped(self) -> None:
        with pytest.raises(ConfigError):
            load_config(SHIPPED_CONFIG, sizing={"max_risk_per_trade": 0.5})


class TestPaths:
    def test_ensure_directories(self, tmp_config) -> None:  # type: ignore[no-untyped-def]
        tmp_config.ensure_directories()
        for path in (
            tmp_config.app.data_dir,
            tmp_config.app.log_dir,
            tmp_config.app.artifact_dir,
            tmp_config.storage.parquet_dir,
            tmp_config.storage.raw_dir,
        ):
            assert tmp_config.resolved_path(path).is_dir()

    def test_absolute_paths_are_left_alone(self, tmp_config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        absolute = tmp_path / "elsewhere"
        assert tmp_config.resolved_path(absolute) == absolute


class TestSecrets:
    def test_missing_credential_raises_actionable_error(self) -> None:
        secrets = BotSecrets(polymarket_private_key=None)
        assert secrets.has("polymarket_private_key") is False
        with pytest.raises(MissingCredentialError, match="POLYMARKET_PRIVATE_KEY"):
            secrets.require("polymarket_private_key")

    def test_secret_value_never_renders(self) -> None:
        secrets = BotSecrets(polymarket_private_key="0xdeadbeef")  # type: ignore[arg-type]
        assert "0xdeadbeef" not in repr(secrets)
        assert "0xdeadbeef" not in str(secrets.model_dump())
        assert secrets.require("polymarket_private_key") == "0xdeadbeef"

    def test_config_and_secrets_share_no_fields(self, shipped_config) -> None:  # type: ignore[no-untyped-def]
        # Structural guarantee that a credential can never arrive via YAML:
        # no field name exists in both objects, so there is nowhere in the
        # reviewable config file for a secret to live.
        def field_names(model: BaseModel) -> set[str]:
            names: set[str] = set()
            for name in type(model).model_fields:
                names.add(name)
                value = getattr(model, name)
                if isinstance(value, BaseModel):
                    names |= field_names(value)
            return names

        overlap = field_names(shipped_config) & set(BotSecrets.model_fields)
        assert overlap == set()

    def test_config_is_frozen(self, shipped_config) -> None:  # type: ignore[no-untyped-def]
        # Nothing may mutate the running configuration: every invariant above
        # was checked once, at load.
        with pytest.raises(ValidationError):
            shipped_config.app.mode = RunMode.LIVE  # type: ignore[misc]
