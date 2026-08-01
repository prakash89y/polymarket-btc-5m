"""Typed, layered configuration.

Design decisions
----------------
1. **Three layers, explicit precedence: environment > YAML > code defaults.**
   YAML is the reviewable, version-controlled description of *how the bot
   behaves*. Environment variables are how a container overrides one knob
   without editing a file. Defaults live in the models, so a missing YAML key is
   never a crash.

2. **Secrets never touch the YAML.** :class:`BotSecrets` reads credentials from
   the environment / ``.env`` only and stores them as ``SecretStr``, so an
   accidental ``print(config)`` or exception dump cannot leak a key.

3. **Validated at load, not at use.** A wrong risk limit must fail at startup,
   not at 03:00 when the first order is sized. Cross-field validators enforce the
   invariants that matter.

4. **Settlement verification is not a tunable.** ``settlement.require_verified``
   may only be ``false`` in backtest mode, and ``allow_unknown_source`` is
   rejected outright outside backtests. The config layer is where that gets
   made structurally impossible rather than left to reviewer discipline.

5. **Live trading is off by default and triple-gated:** ``mode=live`` requires
   ``live.enabled=true`` *and* ``PMBTC_I_UNDERSTAND_LIVE_RISK=yes`` in the
   environment *and* a paper-trading record that cleared the promotion gate
   (checked at runtime by Module 9, using the thresholds declared here).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from pmbtc.constants import ModelName, OrderType, RunMode, SettlementSource
from pmbtc.exceptions import ConfigError, MissingCredentialError

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
ENV_PREFIX = "PMBTC_"
LIVE_INTERLOCK_ENV = "PMBTC_I_UNDERSTAND_LIVE_RISK"

_ENV_INTERPOLATION_RE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


class _Base(BaseModel):
    """Strict base: unknown keys are errors, not silent typos."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
class AppConfig(_Base):
    name: str = "pmbtc"
    mode: RunMode = RunMode.PAPER
    asset: str = "BTC"
    #: Length of one Up/Down market. The whole system is parameterised by this,
    #: so the same code trades the 1-minute or 15-minute family unchanged.
    window_seconds: int = Field(default=300, ge=30, le=3_600)
    #: Bars of history the feature pipeline needs before its first valid row.
    warmup_bars: int = Field(default=1_000, ge=100)
    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    artifact_dir: Path = Path("artifacts")

    @field_validator("asset")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def window_ms(self) -> int:
        return self.window_seconds * 1_000


class LoggingConfig(_Base):
    level: str = "INFO"
    #: JSON in containers (machine-parseable), pretty console locally.
    json_logs: bool = False
    file_enabled: bool = True
    file_name: str = "pmbtc.log"
    rotate_mb: int = Field(default=64, ge=1)
    backup_count: int = Field(default=14, ge=0)
    #: Separate append-only JSONL stream of every decision, kept forever. This is
    #: the training set for the self-learning loop, so it is never rotated away.
    decision_log_name: str = "decisions.jsonl"
    #: Keys scrubbed from every record before emission, at any nesting depth.
    redact_keys: tuple[str, ...] = (
        "api_key",
        "api_secret",
        "secret",
        "passphrase",
        "password",
        "token",
        "private_key",
        "signature",
        "authorization",
        "poly_api_key",
        "poly_passphrase",
    )

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"Invalid log level {v!r}")
        return level


class HttpConfig(_Base):
    connect_timeout_s: float = Field(default=5.0, gt=0)
    read_timeout_s: float = Field(default=15.0, gt=0)
    total_timeout_s: float = Field(default=30.0, gt=0)
    max_connections: int = Field(default=50, ge=1)
    max_keepalive: int = Field(default=20, ge=0)
    max_retries: int = Field(default=4, ge=0, le=10)
    backoff_base_s: float = Field(default=0.5, gt=0)
    backoff_max_s: float = Field(default=30.0, gt=0)
    user_agent: str = "pmbtc/0.1"
    #: Fail fast if a venue's clock and ours disagree by more than this. On a
    #: 300-second instrument, a 2-second clock error is a 0.7% mis-timed window.
    max_clock_skew_ms: int = Field(default=2_000, ge=100)


class RateLimitConfig(_Base):
    """Per-host token bucket."""

    capacity: int = Field(default=600, ge=1)
    refill_per_second: float = Field(default=10.0, gt=0)
    #: Stop issuing requests once this fraction of the bucket is consumed, so
    #: there is always headroom for an urgent cancel.
    soft_limit_ratio: float = Field(default=0.85, gt=0, le=1.0)
    max_wait_s: float = Field(default=10.0, gt=0)


# --------------------------------------------------------------------------- #
# Polymarket
# --------------------------------------------------------------------------- #
class PolymarketConfig(_Base):
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    data_api_base_url: str = "https://data-api.polymarket.com"
    ws_base_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws"
    chain_id: int = 137
    #: Structured identity of the family we trade, from ``events[].series[]``.
    #: Far stronger than a slug pattern, and the primary shortlist key.
    series_slug: str = "btc-up-or-down-5m"
    #: Slug prefixes used to shortlist candidates. Verified live on 2026-07-31:
    #: markets are ``btc-updown-5m-<window_open_epoch>``. Matching is necessary
    #: but never sufficient — the settlement verifier has the final say, and it
    #: is what stops the sibling ``btc-up-or-down-hourly`` family (which
    #: resolves off Binance, not Chainlink) from being traded by mistake.
    market_slug_patterns: tuple[str, ...] = ("btc-updown-5m-",)
    #: Fallback tick size. The real value is per-market (``orderPriceMinTickSize``
    #: was 0.01 on the July markets and 0.001 on the April ones), so execution
    #: reads it from the market and treats this only as a sanity bound.
    tick_size: float = Field(default=0.01, gt=0, lt=1)
    #: Observed ``orderMinSize`` on live BTC 5m markets is 5.
    min_order_usdc: float = Field(default=5.0, gt=0)
    max_order_usdc: float = Field(default=500.0, gt=0)
    #: Polymarket's CLOB has historically charged no explicit trading fee, but
    #: markets may carry one and the value can change. It is configurable, and
    #: the *real* cost — half the spread — is modelled separately in CostConfig.
    taker_fee_bps: float = Field(default=0.0, ge=0)
    maker_fee_bps: float = Field(default=0.0, ge=0)
    neg_risk: bool = False
    rate_limit: RateLimitConfig = RateLimitConfig()
    #: How long a fetched market definition may be reused before re-fetching.
    market_cache_seconds: int = Field(default=60, ge=1)

    @model_validator(mode="after")
    def _check_sizes(self) -> PolymarketConfig:
        if self.max_order_usdc < self.min_order_usdc:
            raise ConfigError(
                "polymarket.max_order_usdc must be >= min_order_usdc",
                context={"min": self.min_order_usdc, "max": self.max_order_usdc},
            )
        return self


class SettlementProviderConfig(_Base):
    """One pluggable settlement price provider.

    Providers are interchangeable behind a single interface (Module 2). Trading
    logic never learns which one is in use; it only asks for "the reference
    price at ``t`` according to this market's declared authority".
    """

    enabled: bool = True
    base_url: str = ""
    symbol: str = ""
    #: Feed/aggregator identifier (Chainlink aggregator address, Pyth price id,
    #: exchange symbol) — whatever uniquely pins the series for this provider.
    feed_id: str = ""
    #: Tolerated age of the quoted price relative to the settlement instant.
    max_lag_ms: int = Field(default=1_500, ge=0)
    options: dict[str, Any] = Field(default_factory=dict)


class SettlementConfig(_Base):
    """Settlement is the immutable core of the system.

    The bot forecasts *one* price series. If a market resolves off a different
    series, every probability the model emits is measuring the wrong thing, and
    the error is invisible in the PnL until it has compounded. So:

    * the source is auto-detected per market from the Gamma rules text,
    * detection must be unambiguous — one and only one provider may match,
    * a market whose source cannot be pinned is skipped, never traded,
    * the detected source, settlement timestamp, and settlement price are
      persisted with every completed trade for post-hoc audit.
    """

    #: The series this bot's models are trained against.
    #: Verified against live Gamma payloads on 2026-07-31: the BTC 5m and 15m
    #: Up/Down families resolve off the Chainlink BTC/USD data stream, while the
    #: hourly family of the same product line resolves off Binance BTCUSDT.
    expected_source: SettlementSource = SettlementSource.CHAINLINK
    #: Hard gate. May only be disabled in backtest mode (enforced below).
    require_verified: bool = True
    #: Never true outside backtests: trading an UNKNOWN source is the exact
    #: failure this whole subsystem exists to prevent.
    allow_unknown_source: bool = False
    #: Re-verify the rules text on every window rather than trusting a cache.
    #: Cheap (one Gamma call) against an unbounded downside.
    reverify_each_window: bool = True
    #: Detection is rejected unless the parser's confidence reaches this. The
    #: parser reports 1.0 only for an exact, unambiguous phrase match.
    min_detection_confidence: float = Field(default=1.0, gt=0, le=1.0)
    #: Settlement price must be retrievable within this long after the window
    #: closes, otherwise the trade is marked unresolved and flagged, not guessed.
    resolution_timeout_s: int = Field(default=180, ge=10)
    #: Cross-check the provider's settlement price against our own recorded tape;
    #: a mismatch beyond this many basis points raises rather than silently
    #: labelling a training row incorrectly.
    price_cross_check_bps: float = Field(default=2.0, ge=0)
    providers: dict[str, SettlementProviderConfig] = Field(
        default_factory=lambda: {
            SettlementSource.CHAINLINK.value: SettlementProviderConfig(
                base_url="https://data.chain.link/streams/btc-usd",
                symbol="BTC/USD",
                feed_id="btc-usd",
            ),
            SettlementSource.BINANCE_SPOT.value: SettlementProviderConfig(
                base_url="https://api.binance.com", symbol="BTCUSDT", feed_id="BTCUSDT"
            ),
            SettlementSource.PYTH.value: SettlementProviderConfig(
                enabled=False, base_url="https://hermes.pyth.network", feed_id=""
            ),
            SettlementSource.COINBASE_SPOT.value: SettlementProviderConfig(
                enabled=False, base_url="https://api.exchange.coinbase.com", symbol="BTC-USD"
            ),
        }
    )

    @model_validator(mode="after")
    def _expected_provider_exists(self) -> SettlementConfig:
        key = self.expected_source.value
        if key not in self.providers:
            raise ConfigError(
                "settlement.expected_source has no matching provider entry",
                context={"expected": key, "configured": sorted(self.providers)},
            )
        if not self.providers[key].enabled:
            raise ConfigError(
                "settlement.expected_source points at a disabled provider",
                context={"expected": key},
            )
        if self.expected_source is SettlementSource.UNKNOWN:
            raise ConfigError("settlement.expected_source may not be 'unknown'")
        return self


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class ClockConfig(_Base):
    """Clock synchronisation and the pre-settlement safety gate.

    On a 300-second instrument a one-second clock error is 0.33% of the window,
    applied to every label and every entry decision. Defaults are deliberately
    tight, and the service fails closed when it cannot establish the offset.
    """

    #: Reference clocks, queried independently; the median offset wins.
    sources: tuple[str, ...] = ("clob", "binance")
    sync_interval_seconds: int = Field(default=60, ge=5)
    #: Beyond this the local clock is untrusted and trading stops.
    max_drift_ms: int = Field(default=1_500, ge=50)
    #: Samples slower than this cannot pin the clock and are discarded.
    max_sample_rtt_ms: int = Field(default=2_000, ge=50)
    #: Residual uncertainty above this also blocks trading.
    max_uncertainty_ms: int = Field(default=750, ge=10)
    min_samples: int = Field(default=1, ge=1)
    #: A sync older than this is treated as no sync at all.
    stale_sync_seconds: int = Field(default=300, ge=10)
    #: No order may be submitted inside this many seconds of settlement.
    order_safety_window_seconds: int = Field(default=20, ge=0)


class GammaConfig(_Base):
    """Market discovery against the Gamma API."""

    #: Discovery is series-driven, never slug-driven. Verified live: querying
    #: ``/events/pagination?series_slug=...&closed=false&end_date_min=<now>``
    #: returns upcoming windows in order, with no slug construction at all.
    discovery_poll_seconds: int = Field(default=30, ge=1)
    #: How many upcoming windows to keep verified and ready.
    lookahead_windows: int = Field(default=3, ge=1, le=50)
    page_limit: int = Field(default=50, ge=1, le=500)
    max_markets_per_scan: int = Field(default=25, ge=1)
    #: Fall back to slug construction only if series discovery returns nothing,
    #: and treat any market it finds as unverified until the normal gates pass.
    allow_slug_fallback: bool = True
    #: Archive every raw response so a parser change can be replayed against it.
    archive_responses: bool = True
    archive_dir: Path = Path("data/raw/gamma")
    #: Retain the last N discovered-but-unparseable payloads for debugging.
    keep_rejected_payloads: int = Field(default=50, ge=0)


class SchemaConfig(_Base):
    """Gamma payload schema drift detection.

    The parser must fail safely rather than silently accept an incompatible
    payload. A removed or retyped required field is fatal; a *new* field is
    recorded and alerted on but does not stop trading, because Polymarket adds
    fields routinely and halting on that would be a self-inflicted outage.
    """

    strict: bool = True
    alert_on_new_fields: bool = True
    #: Persisted registry of schema fingerprints we have already seen.
    registry_file: Path = Path("data/gamma_schema.json")


class HealthConfig(_Base):
    """Per-market health gates applied before a market is considered tradeable."""

    require_accepting_orders: bool = True
    require_order_book: bool = True
    #: Gamma-reported liquidity floor, in USDC.
    min_liquidity_usdc: float = Field(default=100.0, ge=0)
    #: A market with no two-sided quote cannot be sized or exited.
    require_two_sided_quote: bool = True
    max_spread: float = Field(default=0.10, gt=0, lt=1)
    #: Reject a window whose open is further than this from a clean boundary.
    max_boundary_skew_ms: int = Field(default=0, ge=0)
    #: Refuse markets whose settlement is already in the past.
    reject_past_settlement: bool = True
    #: Depth levels to capture in a liquidity snapshot (filled by Module 5).
    depth_levels: int = Field(default=10, ge=1, le=100)


class FeedConfig(_Base):
    """Live streaming feeds — the primary microstructure source.

    Gamma is deliberately not listed here. Measured at 29-72 seconds stale, it
    provides discovery, metadata, settlement information, and historical
    context; it is never a low-latency signal.
    """

    clob_enabled: bool = True
    binance_enabled: bool = True
    #: Silence beyond this means the feed is dead, whatever the socket says.
    clob_staleness_budget_ms: int = Field(default=15_000, ge=1_000)
    binance_staleness_budget_ms: int = Field(default=10_000, ge=1_000)
    reconnect_backoff_base_s: float = Field(default=0.5, gt=0)
    reconnect_backoff_max_s: float = Field(default=30.0, gt=0)
    #: Subscribe this long before a window opens, so the book is warm and the
    #: tape has history by the time the first snapshot is due.
    warmup_seconds: int = Field(default=90, ge=0)
    #: Keep a feed alive this long after settlement, to catch late trades.
    cooldown_seconds: int = Field(default=30, ge=0)
    #: Archiving is what makes live features reproducible. Turning it off makes
    #: every stream-derived feature untrainable by policy, not by accident.
    archive_ticks: bool = True
    archive_dir: Path = Path("data/raw/ticks")
    archive_buffer_frames: int = Field(default=200, ge=1)
    #: Maximum markets streamed at once; each is its own socket.
    max_concurrent_markets: int = Field(default=4, ge=1, le=20)


class ServiceConfig(_Base):
    """Continuous, long-running collection.

    The dataset is the asset. It should keep growing while later modules are
    being written, so the service is designed to run for weeks unattended.
    """

    status_interval_seconds: int = Field(default=300, ge=10)
    #: Liveness is written far more often than the human-readable status log.
    #: A monitor that can only tell the service died five minutes ago is a
    #: monitor that is five minutes late.
    heartbeat_interval_seconds: int = Field(default=15, ge=1)
    #: Restart the collection loop after an unhandled error rather than exiting.
    restart_on_error: bool = True
    max_consecutive_errors: int = Field(default=10, ge=1)
    restart_backoff_s: float = Field(default=5.0, gt=0)
    #: Run the label backfill pass on this cadence.
    label_backfill_seconds: int = Field(default=300, ge=30)


class DatasetConfig(_Base):
    """Historical dataset collection and export.

    This module prioritises integrity over throughput: a missing snapshot is
    always preferable to one that misrepresents when its information was known.
    """

    root_dir: Path = Path("data/dataset")
    export_dir: Path = Path("data/exports")
    #: Seconds before settlement at which a feature snapshot is taken. The
    #: densest sampling is at the end, where the fair value moves fastest, and
    #: any of these can serve as a training horizon.
    snapshot_horizons_s: tuple[int, ...] = (300, 240, 180, 120, 60, 30, 15, 10, 5, 2, 1)
    #: How far after its nominal instant a snapshot may still be captured.
    #: Beyond this the capture is skipped and counted, never backdated.
    max_snapshot_jitter_ms: int = Field(default=1_500, ge=0)
    #: Tolerance the leakage guard allows on ``observation_time``. Small on
    #: purpose: it is for timer slop, not for genuinely late data.
    observation_tolerance_ms: int = Field(default=500, ge=0)
    #: Give up waiting for an official outcome after this long past settlement.
    resolution_max_wait_s: int = Field(default=86_400, ge=60)
    resolution_poll_seconds: int = Field(default=60, ge=5)
    #: Default training-time exclusion thresholds.
    min_snapshot_quality: float = Field(default=0.5, ge=0, le=1)
    min_coverage: float = Field(default=0.6, ge=0, le=1)
    default_format: Literal["parquet", "arrow", "csv", "sqlite"] = "parquet"
    #: Bumped by hand when the dataset's meaning changes; recorded in exports.
    dataset_version: str = "0.1.0"


class VenueConfig(_Base):
    """A market-data venue. These never execute orders — they feed features."""

    enabled: bool = True
    rest_base_url: str = ""
    ws_base_url: str = ""
    symbol: str = ""
    #: Order-book depth levels to subscribe to. 20 is enough for imbalance and
    #: microprice; deeper books cost bandwidth and add little at this horizon.
    book_depth: int = Field(default=20, ge=1, le=1_000)
    kline_limit: int = Field(default=1_000, ge=1, le=1_500)
    rate_limit: RateLimitConfig = RateLimitConfig()


class VenuesConfig(_Base):
    binance_futures: VenueConfig = VenueConfig(
        rest_base_url="https://fapi.binance.com",
        ws_base_url="wss://fstream.binance.com",
        symbol="BTCUSDT",
    )
    binance_spot: VenueConfig = VenueConfig(
        rest_base_url="https://api.binance.com",
        ws_base_url="wss://stream.binance.com:9443",
        symbol="BTCUSDT",
    )
    coinbase: VenueConfig = VenueConfig(
        rest_base_url="https://api.exchange.coinbase.com",
        ws_base_url="wss://advanced-trade-ws.coinbase.com",
        symbol="BTC-USD",
    )
    bybit: VenueConfig = VenueConfig(
        rest_base_url="https://api.bybit.com",
        ws_base_url="wss://stream.bybit.com/v5/public/linear",
        symbol="BTCUSDT",
    )
    hyperliquid: VenueConfig = VenueConfig(
        enabled=False,
        rest_base_url="https://api.hyperliquid.xyz",
        ws_base_url="wss://api.hyperliquid.xyz/ws",
        symbol="BTC",
    )


class SourceConfig(_Base):
    """A low-frequency auxiliary source (regime tier)."""

    enabled: bool = False
    poll_seconds: int = Field(default=300, ge=1)
    #: Past this age the feature is emitted as NaN rather than as a stale value.
    max_staleness_seconds: int = Field(default=3_600, ge=1)
    base_url: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class SourcesConfig(_Base):
    coinglass: SourceConfig = SourceConfig(
        base_url="https://open-api-v4.coinglass.com", poll_seconds=60
    )
    glassnode: SourceConfig = SourceConfig(
        base_url="https://api.glassnode.com", poll_seconds=3_600, max_staleness_seconds=86_400
    )
    cryptoquant: SourceConfig = SourceConfig(
        base_url="https://api.cryptoquant.com", poll_seconds=3_600, max_staleness_seconds=86_400
    )
    fear_greed: SourceConfig = SourceConfig(
        enabled=True,
        base_url="https://api.alternative.me",
        poll_seconds=3_600,
        max_staleness_seconds=172_800,
    )
    etf_flows: SourceConfig = SourceConfig(poll_seconds=21_600, max_staleness_seconds=172_800)
    stablecoin_flows: SourceConfig = SourceConfig(poll_seconds=3_600, max_staleness_seconds=86_400)
    macro_calendar: SourceConfig = SourceConfig(poll_seconds=1_800, max_staleness_seconds=86_400)
    news: SourceConfig = SourceConfig(poll_seconds=300, max_staleness_seconds=3_600)
    twitter: SourceConfig = SourceConfig(poll_seconds=300, max_staleness_seconds=3_600)
    tradingview: SourceConfig = SourceConfig(poll_seconds=300)
    whale_alerts: SourceConfig = SourceConfig(poll_seconds=120, max_staleness_seconds=3_600)


class StorageConfig(_Base):
    #: SQLite by default so the repo runs on a laptop with zero setup; point at
    #: Postgres/Timescale for production without touching another module.
    db_url: str = "sqlite+aiosqlite:///data/pmbtc.db"
    echo: bool = False
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    #: Bulk tick/candle history lives in Parquet; the DB holds state, markets,
    #: decisions, trades, and model metadata.
    parquet_dir: Path = Path("data/parquet")
    #: Raw payload capture. Expensive but it is the only way to rebuild a
    #: feature after a bug is found, so it defaults on.
    raw_capture_enabled: bool = True
    raw_dir: Path = Path("data/raw")
    #: Retention of raw payloads, in days. Decisions and trades are never purged.
    raw_retention_days: int = Field(default=90, ge=1)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
class FeatureTierConfig(_Base):
    """A priority tier of features.

    Tiers express a *prior*, not a hard-coded importance: they control what is
    computed, how fresh it must be, and how much of the selection budget each
    group may occupy. Which features actually survive is decided empirically by
    :class:`FeatureSelectionConfig`.
    """

    enabled: bool = True
    #: Upper bound on how many features from this tier may enter the final set.
    #: Prevents 300 slow-moving regime columns from crowding out microstructure.
    max_selected: int = Field(default=100, ge=0)
    #: Rows where a tier's inputs are staler than this get NaN for that tier.
    max_staleness_seconds: int = Field(default=60, ge=1)


class FeatureSelectionConfig(_Base):
    """Evidence-driven selection. No manual importance anywhere.

    All four methods run; a feature must clear the vote threshold on the
    aggregate rank. Selection is fitted on training folds only and re-fitted at
    every retrain, so it cannot leak validation information.
    """

    methods: tuple[str, ...] = ("shap", "mutual_info", "permutation", "rfe")
    #: Fraction of methods that must rank a feature above the cutoff.
    vote_threshold: float = Field(default=0.5, gt=0, le=1.0)
    max_features: int = Field(default=120, ge=5)
    #: Drop one of any pair correlated above this before ranking, keeping the
    #: cheaper/faster feature. Collinear inputs split SHAP credit and make
    #: importance unreadable.
    max_correlation: float = Field(default=0.95, gt=0, le=1.0)
    #: Features whose permutation importance is below this on validation are
    #: dropped regardless of votes.
    min_permutation_importance: float = Field(default=0.0)
    #: Re-run selection every N retrains (1 = every time).
    refit_every_n_retrains: int = Field(default=1, ge=1)
    random_seed: int = 42


class FeatureConfig(_Base):
    #: Base bar sizes the feature pipeline resamples the tape into. Sub-window
    #: bars matter most: a 5-minute market is decided by 1s-30s dynamics.
    bar_intervals: tuple[str, ...] = ("1s", "5s", "15s", "1m", "5m")
    higher_timeframes: tuple[str, ...] = ("15m", "1h", "4h")
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21)
    ema_periods: tuple[int, ...] = (9, 21, 50, 200)
    rsi_periods: tuple[int, ...] = (7, 14, 21)
    atr_periods: tuple[int, ...] = (14, 50)
    bb_period: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    adx_period: int = Field(default=14, ge=2)
    macd: tuple[int, int, int] = (12, 26, 9)
    #: Windows (seconds) for realized volatility. The normaliser for
    #: distance-from-open — the single most important feature in this market.
    realized_vol_windows_s: tuple[int, ...] = (30, 60, 120, 300, 900)
    #: Order-book imbalance depth levels.
    book_imbalance_levels: tuple[int, ...] = (1, 5, 10, 20)
    #: CVD / volume-delta lookbacks in seconds.
    flow_windows_s: tuple[int, ...] = (10, 30, 60, 300)
    #: Reject the whole feature frame if more than this fraction is NaN.
    max_nan_ratio: float = Field(default=0.05, ge=0, le=1)
    microstructure: FeatureTierConfig = FeatureTierConfig(max_selected=100, max_staleness_seconds=5)
    short_term: FeatureTierConfig = FeatureTierConfig(max_selected=40, max_staleness_seconds=60)
    regime: FeatureTierConfig = FeatureTierConfig(max_selected=15, max_staleness_seconds=86_400)
    selection: FeatureSelectionConfig = FeatureSelectionConfig()


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
class ModelConfig(_Base):
    enabled_models: tuple[ModelName, ...] = (
        ModelName.BASELINE_VOL,
        ModelName.LOGISTIC,
        ModelName.LIGHTGBM,
        ModelName.XGBOOST,
        ModelName.CATBOOST,
        ModelName.RANDOM_FOREST,
        ModelName.LSTM,
        ModelName.TRANSFORMER,
        ModelName.TFT,
        ModelName.TABNET,
    )
    ensemble: bool = True
    ensemble_method: Literal["stacking", "weighted_average", "rank_average"] = "stacking"
    #: Purged, embargoed walk-forward CV — the only honest validation scheme for
    #: overlapping financial series.
    cv_folds: int = Field(default=6, ge=2)
    embargo_windows: int = Field(default=12, ge=0)
    train_days: int = Field(default=120, ge=7)
    validation_days: int = Field(default=21, ge=1)
    #: Holdout is never touched during training or selection. It exists solely
    #: to decide champion vs challenger.
    holdout_days: int = Field(default=21, ge=1)
    calibration: Literal["isotonic", "sigmoid", "beta", "none"] = "isotonic"
    #: Calibration is fitted on its own slice so it cannot inherit the fit error
    #: of the model it is calibrating.
    calibration_days: int = Field(default=14, ge=1)
    hyperparam_trials: int = Field(default=40, ge=0)
    early_stopping_rounds: int = Field(default=100, ge=1)
    random_seed: int = 42
    n_jobs: int = -1


class TrainingConfig(_Base):
    """Scheduled retraining with champion/challenger promotion.

    The production model is never updated per trade. Every settled market is
    logged; retraining happens on a schedule; a challenger is promoted only if it
    beats the champion on an unseen holdout by a statistically meaningful margin.
    """

    #: Retrain when either trigger fires.
    retrain_every_settled_markets: int = Field(default=576, ge=10)  # ~2 days of 5m windows
    retrain_every_hours: int = Field(default=24, ge=1)
    #: Minimum settled markets before any model may be trained at all.
    min_samples_to_train: int = Field(default=5_000, ge=100)
    #: Emergency retrain if live rolling accuracy falls this far below holdout.
    degradation_trigger: float = Field(default=0.03, ge=0, le=0.5)
    degradation_window: int = Field(default=200, ge=20)
    #: Promotion gate — challenger must beat champion on the holdout by all of:
    min_brier_improvement: float = Field(default=0.002, ge=0)
    min_logloss_improvement: float = Field(default=0.002, ge=0)
    #: One-sided test p-value the improvement must clear (paired, per-window).
    promotion_significance: float = Field(default=0.05, gt=0, lt=1)
    #: Challenger must also be at least this well calibrated in absolute terms.
    max_holdout_calibration_error: float = Field(default=0.03, ge=0, le=1)
    #: Keep this many past production models for instant rollback.
    keep_model_versions: int = Field(default=10, ge=1)
    #: Shadow-run a promoted model for this many windows before it sizes real
    #: money; its predictions are logged and scored but not traded.
    shadow_windows_before_live: int = Field(default=288, ge=0)

    # --- Model readiness gate ------------------------------------------ #
    #: Module 7 refuses to train until the dataset clears all of these. The
    #: point is to make "we do not have enough data yet" an explicit, loud
    #: failure rather than a quietly overfitted model nobody questions.
    min_labelled_markets: int = Field(default=2_000, ge=1)
    #: Markets with every scheduled snapshot present. A model trained mostly on
    #: partial timelines cannot be evaluated at a fixed horizon.
    min_complete_timelines: int = Field(default=1_000, ge=1)
    #: Fraction of registered features that must be present across the dataset.
    min_feature_coverage: float = Field(default=0.8, ge=0, le=1)
    #: Both classes must be represented; a degenerate split trains a constant.
    min_class_balance: float = Field(default=0.35, gt=0, lt=0.5)
    #: Reject the dataset if this much of it is below the quality floor.
    max_low_quality_fraction: float = Field(default=0.3, ge=0, le=1)
    #: Every model must beat every benchmark baseline by this margin on the
    #: holdout before it may be promoted.
    min_baseline_improvement: float = Field(default=0.005, ge=0)


class PredictionConfig(_Base):
    """Abstention thresholds.

    A 5-minute BTC coin flip is close to fair. The only route to positive
    expectancy is to trade a small, high-conviction subset and abstain the rest
    of the time, so these gates are deliberately strict by default.
    """

    min_confidence: float = Field(default=0.60, gt=0.5, lt=1.0)
    #: Model probability must exceed the market's implied probability by this
    #: much (in probability points) before the edge is considered real.
    min_edge: float = Field(default=0.04, gt=0, le=0.5)
    #: Minimum expected value per USDC of stake, after modelled costs.
    min_ev: float = Field(default=0.02, gt=0)
    min_expected_roi: float = Field(default=0.03, gt=0)
    #: Refuse to trade at all while the deployed model's calibration error on
    #: recent settled markets exceeds this.
    max_calibration_error: float = Field(default=0.04, ge=0, le=1)
    #: Fraction of ensemble members that must agree on direction.
    min_model_agreement: float = Field(default=0.6, ge=0, le=1)
    #: How often the prediction loop runs inside a window.
    predict_every_seconds: int = Field(default=15, ge=1)


class CostConfig(_Base):
    """One cost model, used identically by backtest, paper, and live.

    On Polymarket the dominant cost is not a fee — it is crossing the spread on
    a two-sided book that is often 1-3 cents wide. Half-spread plus slippage is
    modelled explicitly so a backtest cannot flatter itself.
    """

    #: Assumed adverse fill beyond the touch, in probability points.
    slippage: float = Field(default=0.005, ge=0)
    #: Charged on notional, in basis points; usually 0 on Polymarket today.
    taker_fee_bps: float = Field(default=0.0, ge=0)
    maker_fee_bps: float = Field(default=0.0, ge=0)
    #: Gas is relayer-sponsored for CLOB trades but non-zero for redemptions.
    gas_cost_usdc: float = Field(default=0.0, ge=0)
    #: If we let a winner settle rather than selling out, we pay no exit cost —
    #: modelled explicitly because it materially changes optimal exit policy.
    assume_hold_to_settlement: bool = True


class ExecutionConfig(_Base):
    order_type: OrderType = OrderType.FOK
    #: Timing gates inside the window. Entering in the first seconds means
    #: trading a near-coin-flip against a wide book; entering in the last
    #: seconds means fill and settlement-latency risk.
    min_seconds_into_window: int = Field(default=45, ge=0)
    min_seconds_to_settlement: int = Field(default=30, ge=0)
    #: Override for the late window: allowed only when the edge is this large.
    late_entry_min_edge: float = Field(default=0.12, gt=0, le=1)
    max_spread: float = Field(default=0.03, gt=0, lt=1)
    #: Depth required within the slippage budget, on the side we must lift.
    min_book_depth_usdc: float = Field(default=200.0, ge=0)
    max_slippage: float = Field(default=0.01, gt=0)
    #: Volatility circuit breaker: skip if short-horizon realized vol is this
    #: many z-scores above its recent norm (fat-tail windows are unmodellable).
    max_vol_zscore: float = Field(default=3.0, gt=0)
    #: Book staleness budget for the Polymarket side.
    max_book_age_ms: int = Field(default=2_000, ge=100)
    #: Allow closing a position before settlement when the edge inverts.
    allow_early_exit: bool = True
    early_exit_edge_flip: float = Field(default=0.06, gt=0, le=1)
    #: One attempt per window per outcome; no averaging into a loser.
    max_entries_per_window: int = Field(default=1, ge=1, le=5)
    order_timeout_s: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check_costs_vs_gates(self) -> ExecutionConfig:
        if self.max_slippage >= self.max_spread:
            raise ConfigError(
                "execution.max_slippage must be tighter than max_spread, otherwise "
                "the slippage gate can never bind",
                context={"max_slippage": self.max_slippage, "max_spread": self.max_spread},
            )
        return self


class SizingConfig(_Base):
    """Kelly with a safety factor, or fixed fractional.

    For a binary contract bought at price ``c`` with true probability ``p``, the
    full-Kelly fraction of bankroll is ``(p - c) / (1 - c)``. Full Kelly is
    correct only if ``p`` is exactly right; since ``p`` is a model output, the
    default is quarter-Kelly and hard caps sit above it.
    """

    mode: Literal["kelly", "fixed_fraction", "fixed_usdc"] = "kelly"
    kelly_fraction: float = Field(default=0.25, gt=0, le=1.0)
    fixed_fraction: float = Field(default=0.01, gt=0, le=0.25)
    fixed_usdc: float = Field(default=10.0, gt=0)
    #: Hard cap on bankroll risked in one market, whatever Kelly says.
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=0.10)
    max_position_usdc: float = Field(default=250.0, gt=0)
    min_position_usdc: float = Field(default=5.0, gt=0)
    #: Shrink size when the model's recent calibration is poor: multiply the
    #: Kelly fraction by ``(1 - calibration_error / max_calibration_error)``.
    calibration_scaling: bool = True
    #: Round stake down to this many USDC to avoid dust orders.
    stake_rounding_usdc: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> SizingConfig:
        if self.min_position_usdc > self.max_position_usdc:
            raise ConfigError(
                "sizing.min_position_usdc must be <= max_position_usdc",
                context={"min": self.min_position_usdc, "max": self.max_position_usdc},
            )
        return self


class RiskConfig(_Base):
    max_daily_loss_usdc: float = Field(default=100.0, gt=0)
    max_daily_loss_fraction: float = Field(default=0.05, gt=0, le=0.5)
    max_weekly_loss_fraction: float = Field(default=0.10, gt=0, le=0.8)
    max_total_exposure_usdc: float = Field(default=500.0, gt=0)
    max_concurrent_positions: int = Field(default=1, ge=1, le=20)
    #: Stand down for the session after this many consecutive losses. Guards
    #: against a regime the model has not seen, which looks exactly like this.
    max_consecutive_losses: int = Field(default=5, ge=1)
    #: Cool-off after a stand-down, in minutes.
    cooldown_minutes: int = Field(default=60, ge=0)
    #: Flatten and stop if market data goes quiet for this long.
    heartbeat_timeout_s: int = Field(default=60, ge=5)
    kill_switch_file: Path = Path("data/KILL_SWITCH")

    @model_validator(mode="after")
    def _check_ordering(self) -> RiskConfig:
        if self.max_weekly_loss_fraction < self.max_daily_loss_fraction:
            raise ConfigError(
                "risk.max_weekly_loss_fraction must be >= max_daily_loss_fraction",
                context={
                    "daily": self.max_daily_loss_fraction,
                    "weekly": self.max_weekly_loss_fraction,
                },
            )
        return self


class BacktestConfig(_Base):
    initial_bankroll_usdc: float = Field(default=1_000.0, gt=0)
    windows_days: tuple[int, ...] = (30, 90, 180)
    #: Deployment gate: every evaluation window must clear all of these.
    min_accuracy: float = Field(default=0.54, gt=0, lt=1)
    min_roi: float = Field(default=0.05)
    min_profit_factor: float = Field(default=1.15, gt=0)
    min_sharpe: float = Field(default=1.0)
    max_drawdown: float = Field(default=0.20, gt=0, lt=1)
    max_brier: float = Field(default=0.245, gt=0, le=1)
    min_trades: int = Field(default=200, ge=1)
    #: Reconstructed-book fills are optimistic by nature; penalise them.
    pessimistic_fill: bool = True


class PaperConfig(_Base):
    initial_bankroll_usdc: float = Field(default=1_000.0, gt=0)
    #: Live trading stays locked until this many paper markets have settled.
    min_trades_before_live: int = Field(default=500, ge=1)
    #: One-sided binomial p-value the paper record must beat versus break-even.
    required_significance: float = Field(default=0.05, gt=0, lt=1)
    min_roi_before_live: float = Field(default=0.03)
    #: Fill against the live book, at the touch, with modelled slippage.
    fill_model: Literal["touch", "mid", "aggressive"] = "touch"


class LiveConfig(_Base):
    enabled: bool = False
    #: Even with live enabled, dry_run short-circuits order submission.
    dry_run: bool = True
    funder_address: str = ""
    max_orders_per_minute: int = Field(default=6, ge=1)
    #: Re-check the paper-promotion gate on every start, not just once.
    enforce_promotion_gate: bool = True


class MonitoringConfig(_Base):
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = Field(default=8787, ge=1, le=65_535)
    metrics_enabled: bool = True
    metrics_port: int = Field(default=9090, ge=1, le=65_535)
    #: Alert channels are configured by URL in secrets, not here.
    alert_on_halt: bool = True
    alert_on_promotion: bool = True
    alert_on_settlement_mismatch: bool = True


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class Config(BaseSettings):
    """Root configuration object.

    Precedence (highest first): environment → ``.env`` → YAML/init → defaults.
    ``settings_customise_sources`` flips the pydantic-settings default (which
    puts init first) so an env var always beats the checked-in YAML.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    http: HttpConfig = HttpConfig()
    polymarket: PolymarketConfig = PolymarketConfig()
    settlement: SettlementConfig = SettlementConfig()
    clock: ClockConfig = ClockConfig()
    gamma: GammaConfig = GammaConfig()
    schema_check: SchemaConfig = SchemaConfig()
    health: HealthConfig = HealthConfig()
    dataset: DatasetConfig = DatasetConfig()
    feeds: FeedConfig = FeedConfig()
    service: ServiceConfig = ServiceConfig()
    venues: VenuesConfig = VenuesConfig()
    sources: SourcesConfig = SourcesConfig()
    storage: StorageConfig = StorageConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    prediction: PredictionConfig = PredictionConfig()
    costs: CostConfig = CostConfig()
    execution: ExecutionConfig = ExecutionConfig()
    sizing: SizingConfig = SizingConfig()
    risk: RiskConfig = RiskConfig()
    backtest: BacktestConfig = BacktestConfig()
    paper: PaperConfig = PaperConfig()
    live: LiveConfig = LiveConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _cross_section_checks(self) -> Config:
        self._check_settlement_gates()
        self._check_live_interlocks()
        self._check_edge_vs_costs()
        self._check_window_timing()
        return self

    def _check_settlement_gates(self) -> None:
        """Settlement verification may not be weakened outside a backtest."""
        if self.app.mode is RunMode.BACKTEST:
            return
        if not self.settlement.require_verified:
            raise ConfigError(
                "settlement.require_verified may only be false in backtest mode; "
                "trading an unverified settlement source is never permitted",
                context={"mode": self.app.mode.value},
            )
        if self.settlement.allow_unknown_source:
            raise ConfigError(
                "settlement.allow_unknown_source may only be true in backtest mode",
                context={"mode": self.app.mode.value},
            )

    def _check_live_interlocks(self) -> None:
        if self.app.mode is not RunMode.LIVE:
            return
        if not self.live.enabled:
            raise ConfigError("mode=live requires live.enabled=true")
        if os.getenv(LIVE_INTERLOCK_ENV, "").strip().lower() not in {"yes", "true", "1"}:
            raise ConfigError(
                f"Live mode is interlocked. Set {LIVE_INTERLOCK_ENV}=yes in the "
                "environment to arm real-money trading."
            )

    def _check_edge_vs_costs(self) -> None:
        """A signal that cannot pay for its own execution is not a signal."""
        round_trip = self.costs.slippage + self.execution.max_spread / 2
        if self.prediction.min_edge < round_trip:
            raise ConfigError(
                "prediction.min_edge is below the modelled round-trip cost; every "
                "accepted signal would have negative expectancy by construction",
                context={"min_edge": self.prediction.min_edge, "round_trip": round_trip},
            )
        if self.prediction.min_confidence <= 0.5 + self.prediction.min_edge:
            raise ConfigError(
                "prediction.min_confidence must exceed 0.5 + min_edge, otherwise the "
                "abstention gate is a no-op",
                context={
                    "min_confidence": self.prediction.min_confidence,
                    "min_edge": self.prediction.min_edge,
                },
            )

    def _check_window_timing(self) -> None:
        exec_cfg = self.execution
        # The clock safety window is the hard deadline; the execution gate must
        # not claim to allow entries the clock service would refuse.
        if self.clock.order_safety_window_seconds > exec_cfg.min_seconds_to_settlement:
            raise ConfigError(
                "clock.order_safety_window_seconds exceeds "
                "execution.min_seconds_to_settlement; the execution gate would "
                "permit entries the clock service always blocks",
                context={
                    "safety_window_s": self.clock.order_safety_window_seconds,
                    "min_seconds_to_settlement": exec_cfg.min_seconds_to_settlement,
                },
            )
        span = exec_cfg.min_seconds_into_window + exec_cfg.min_seconds_to_settlement
        if span >= self.app.window_seconds:
            raise ConfigError(
                "execution timing gates leave no tradeable interval inside the window",
                context={
                    "window_seconds": self.app.window_seconds,
                    "min_seconds_into_window": exec_cfg.min_seconds_into_window,
                    "min_seconds_to_settlement": exec_cfg.min_seconds_to_settlement,
                },
            )
        if self.prediction.predict_every_seconds > self.app.window_seconds:
            raise ConfigError(
                "prediction.predict_every_seconds exceeds the window length; the bot "
                "would never evaluate a market"
            )

    # ------------------------------------------------------------------ #
    @property
    def window_ms(self) -> int:
        return self.app.window_ms

    @property
    def is_live(self) -> bool:
        """True only when real orders can actually leave the process."""
        return self.app.mode is RunMode.LIVE and self.live.enabled and not self.live.dry_run

    def resolved_path(self, path: Path) -> Path:
        """Resolve a configured path relative to ``app.base_dir``."""
        return path if path.is_absolute() else (self.app.base_dir / path)

    def ensure_directories(self) -> None:
        """Create the runtime directories the bot writes to."""
        for path in (
            self.app.data_dir,
            self.app.log_dir,
            self.app.artifact_dir,
            self.storage.parquet_dir,
            self.storage.raw_dir,
        ):
            self.resolved_path(path).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
class BotSecrets(BaseSettings):
    """Credentials, sourced from the environment only.

    ``SecretStr`` keeps these out of logs, tracebacks, and ``repr``. Note the
    absence of a ``polymarket_private_key`` default anywhere in YAML — the key
    that can move funds exists in exactly one place, the process environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    polymarket_private_key: SecretStr | None = None
    polymarket_api_key: SecretStr | None = None
    polymarket_api_secret: SecretStr | None = None
    polymarket_passphrase: SecretStr | None = None
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None
    bybit_api_key: SecretStr | None = None
    bybit_api_secret: SecretStr | None = None
    coinglass_api_key: SecretStr | None = None
    glassnode_api_key: SecretStr | None = None
    cryptoquant_api_key: SecretStr | None = None
    news_api_key: SecretStr | None = None
    twitter_bearer_token: SecretStr | None = None
    alert_webhook_url: SecretStr | None = None

    def require(self, field: str) -> str:
        """Return a credential or raise a typed, actionable error."""
        value: SecretStr | None = getattr(self, field, None)
        if value is None or not value.get_secret_value():
            raise MissingCredentialError(
                f"Missing credential {field!r}; set {field.upper()} in your .env or environment"
            )
        return value.get_secret_value()

    def has(self, field: str) -> bool:
        value: SecretStr | None = getattr(self, field, None)
        return value is not None and bool(value.get_secret_value())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _interpolate_env(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` inside YAML string values."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            env_value = os.getenv(name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ConfigError(f"Config references undefined environment variable ${{{name}}}")

        return _ENV_INTERPOLATION_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML config file, returning ``{}`` when it does not exist."""
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root of {path} must be a mapping, got {type(raw).__name__}")
    return _interpolate_env(raw)


def load_config(path: Path | str | None = None, **overrides: Any) -> Config:
    """Build a validated :class:`Config`.

    Args:
        path: YAML file; defaults to ``$PMBTC_CONFIG`` or ``config/config.yaml``.
        **overrides: programmatic overrides (used heavily in tests).
    """
    if path is None:
        path = os.getenv("PMBTC_CONFIG", str(DEFAULT_CONFIG_PATH))
    data = load_yaml(Path(path))
    data.update(overrides)
    try:
        return Config(**data)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError wraps our ConfigErrors
        raise ConfigError(f"Invalid configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide singleton. Call :func:`reset_config` in tests."""
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> BotSecrets:
    return BotSecrets()


def reset_config() -> None:
    """Clear the cached config/secrets (test helper; also used on SIGHUP reload)."""
    get_config.cache_clear()
    get_secrets.cache_clear()
