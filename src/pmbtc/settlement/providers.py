"""Pluggable settlement price providers.

Two distinct things live here, deliberately separated:

``ProviderDescriptor``
    The *declarative* identity of a settlement authority: how its markets
    describe it, which trading pairs it publishes, what precision it reports at.
    This is what the parser matches against and what the verifier checks. Adding
    a new provider is a descriptor, not a code change in the trading path.

``SettlementPriceProvider``
    The *runtime* interface for asking "what was the reference price at instant
    ``t``". Module 2 defines the protocol and the registry; the concrete network
    clients arrive with the data collectors (Modules 4-5). Trading logic only
    ever touches this protocol, so swapping Chainlink for Pyth changes a config
    key and nothing else.

Facts encoded below were read from live Gamma payloads on 2026-07-31, not
assumed. The BTC 5-minute and 15-minute families resolve off **Chainlink**; the
hourly family resolves off **Binance BTCUSDT**. That divergence inside one
product line is the reason this module exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pmbtc.constants import SettlementSource
from pmbtc.exceptions import ConfigError

# --------------------------------------------------------------------------- #
# Trading pairs
# --------------------------------------------------------------------------- #
#: BTC/USD and BTC/USDT are *different instruments*. They differ by the USDT peg
#: basis, which is small but not zero and widens exactly when volatility spikes.
#: Collapsing them into "BTC" would silently mix two series.
CANONICAL_PAIRS: dict[str, str] = {
    "btc/usd": "BTC/USD",
    "btcusd": "BTC/USD",
    "btc-usd": "BTC/USD",
    "btc_usd": "BTC/USD",
    "xbt/usd": "BTC/USD",
    "btc/usdt": "BTC/USDT",
    "btcusdt": "BTC/USDT",
    "btc-usdt": "BTC/USDT",
    "btc_usdt": "BTC/USDT",
    "btc/usdc": "BTC/USDC",
    "btcusdc": "BTC/USDC",
}


def canonical_pair(raw: str) -> str | None:
    """Normalise a pair as written in market text to a canonical form."""
    return CANONICAL_PAIRS.get(raw.strip().lower().replace(" ", ""))


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Declarative identity of one settlement authority.

    Attributes:
        url_markers: Substrings that identify this provider in the *structured*
            ``resolutionSource`` field. Structured evidence is preferred over
            prose everywhere in this module.
        text_marker_groups: Groups of substrings for the free-text fallback. A
            group matches only if **every** substring in it is present, so
            "chainlink" alone is not enough — the text must also mention the
            data stream. Prose is only ever used to *corroborate or contradict*
            the structured field, never as the sole basis for trading.
        supported_pairs: Pairs this provider is accepted as an authority for.
        price_decimals: Reporting precision. Declared by us from provider
            documentation, not parsed from the market — markets do not state it.
            The verifier requires it to be declared so that a rounding rule is
            never merely implicit.
        historical_access: Honest note on whether the settlement series can be
            replayed for backtesting. Read this before trusting a backtest.
    """

    source: SettlementSource
    display_name: str
    venue: str
    url_markers: tuple[str, ...]
    text_marker_groups: tuple[tuple[str, ...], ...]
    supported_pairs: frozenset[str]
    default_pair: str
    price_decimals: int
    rounding_mode: str = "half_even"
    #: Does the provider timestamp a point-in-time observation, or a candle?
    timestamp_semantics: str = "instant"
    historical_access: str = ""
    notes: str = ""

    def matches_url(self, resolution_source: str) -> bool:
        text = (resolution_source or "").lower()
        return any(marker in text for marker in self.url_markers)

    def matches_text(self, description: str) -> bool:
        text = (description or "").lower()
        return any(all(part in text for part in group) for group in self.text_marker_groups)

    def supports(self, pair: str) -> bool:
        return pair in self.supported_pairs


CHAINLINK = ProviderDescriptor(
    source=SettlementSource.CHAINLINK,
    display_name="Chainlink Data Streams",
    venue="Chainlink",
    url_markers=("data.chain.link", "chain.link/streams"),
    text_marker_groups=(
        ("chainlink", "data stream"),
        ("chainlink", "btc/usd"),
    ),
    supported_pairs=frozenset({"BTC/USD"}),
    default_pair="BTC/USD",
    # Data Streams v3 reports carry 18-decimal fixed-point prices.
    price_decimals=18,
    timestamp_semantics="instant",
    historical_access=(
        "Data Streams is a credentialed low-latency service. Historical replay "
        "requires API access; without it, backtest labels must come from "
        "Polymarket's own recorded resolution rather than a reconstructed feed."
    ),
    notes="Authority for the BTC 5m and 15m Up/Down families as of 2026-07.",
)

BINANCE_SPOT = ProviderDescriptor(
    source=SettlementSource.BINANCE_SPOT,
    display_name="Binance Spot",
    venue="Binance",
    url_markers=("binance.com", "binance.us"),
    text_marker_groups=(
        ("binance", "btcusdt"),
        ("binance", "btc/usdt"),
        ("binance", "btc_usdt"),
    ),
    supported_pairs=frozenset({"BTC/USDT"}),
    default_pair="BTC/USDT",
    price_decimals=2,
    timestamp_semantics="candle_close",
    historical_access="Fully replayable via public klines REST (no credentials).",
    notes="Authority for the BTC hourly Up/Down family and older quarterly markets.",
)

PYTH = ProviderDescriptor(
    source=SettlementSource.PYTH,
    display_name="Pyth Network",
    venue="Pyth",
    url_markers=("pyth.network", "pyth.xyz"),
    text_marker_groups=(("pyth", "btc/usd"),),
    supported_pairs=frozenset({"BTC/USD"}),
    default_pair="BTC/USD",
    price_decimals=8,
    timestamp_semantics="instant",
    historical_access="Hermes serves historical price updates by publish time.",
)

COINBASE_SPOT = ProviderDescriptor(
    source=SettlementSource.COINBASE_SPOT,
    display_name="Coinbase Exchange",
    venue="Coinbase",
    url_markers=("coinbase.com", "exchange.coinbase.com"),
    text_marker_groups=(("coinbase", "btc-usd"), ("coinbase", "btc/usd")),
    supported_pairs=frozenset({"BTC/USD"}),
    default_pair="BTC/USD",
    price_decimals=2,
    timestamp_semantics="candle_close",
    historical_access="Public candles REST.",
)

PROVIDER_REGISTRY: dict[SettlementSource, ProviderDescriptor] = {
    d.source: d
    for d in (CHAINLINK, BINANCE_SPOT, PYTH, COINBASE_SPOT)
}


def descriptor_for(source: SettlementSource) -> ProviderDescriptor:
    """Look up a descriptor, or fail loudly."""
    try:
        return PROVIDER_REGISTRY[source]
    except KeyError as exc:
        raise ConfigError(
            f"No provider descriptor registered for {source!r}",
            context={"known": sorted(s.value for s in PROVIDER_REGISTRY)},
        ) from exc


def identify_by_url(resolution_source: str) -> list[ProviderDescriptor]:
    """All providers whose URL markers appear in a structured resolution source.

    Returns a *list* on purpose: more than one match means the market is
    ambiguous, and ambiguity must reach the verifier rather than be resolved by
    an arbitrary tie-break here.
    """
    return [d for d in PROVIDER_REGISTRY.values() if d.matches_url(resolution_source)]


def identify_by_text(description: str) -> list[ProviderDescriptor]:
    """All providers whose text markers appear in the free-text rules."""
    return [d for d in PROVIDER_REGISTRY.values() if d.matches_text(description)]


# --------------------------------------------------------------------------- #
# Runtime price interface
# --------------------------------------------------------------------------- #
@runtime_checkable
class SettlementPriceProvider(Protocol):
    """Runtime access to a settlement authority's price series.

    Implementations arrive with the data collectors. Everything downstream --
    labelling, audit, PnL attribution -- depends only on this protocol.
    """

    source: SettlementSource

    async def price_at(self, timestamp_ms: int, pair: str) -> Decimal:
        """Reference price at an instant, per this provider's own semantics."""
        ...

    async def is_available(self) -> bool:
        """Whether the provider can currently serve prices."""
        ...


@dataclass
class _ProviderRuntimeRegistry:
    """Late-bound registry of concrete price clients.

    Empty in Module 2. The verifier reports a provider as "declared but not yet
    wired", which is a legitimate state for paper trading against Polymarket's
    own resolution, and a blocker for settlement-price audit.
    """

    _clients: dict[SettlementSource, SettlementPriceProvider] = field(default_factory=dict)

    def register(self, client: SettlementPriceProvider) -> None:
        self._clients[client.source] = client

    def get(self, source: SettlementSource) -> SettlementPriceProvider | None:
        return self._clients.get(source)

    def has(self, source: SettlementSource) -> bool:
        return source in self._clients

    def clear(self) -> None:
        self._clients.clear()


PRICE_PROVIDERS = _ProviderRuntimeRegistry()
