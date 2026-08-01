"""Gamma market -> canonical :class:`SettlementSpecification`.

Extraction order is always **structured metadata first**:

* ``resolutionSource``            -> provider  (a URL, not prose)
* ``events[].series[].recurrence``-> interval
* ``events[].startTime``          -> window open
* ``endDate``                     -> settlement instant
* ``slug`` (``btc-updown-5m-<epoch>``) -> independent check on open + interval

Free text is used for exactly two things: to extract rules that exist *only* in
prose (the tie rule, the reference-price rule), and to **cross-examine** the
structured fields. If the prose says Binance and ``resolutionSource`` points at
Chainlink, that is a conflict and the market is refused — we do not pick a
winner.

Confidence semantics
--------------------
A field scores 1.0 only when it can be established beyond doubt: unambiguous in
its own channel, and agreeing with every other channel that can speak to it.
Where two channels exist (provider, interval, timing) 1.0 requires both. Where a
fact exists only in prose (tie rule) 1.0 requires an exact match against one and
only one known rule dialect. Where a fact is declared by us rather than by the
market (reporting precision) the evidence is labelled ``provider_default`` so no
reader mistakes it for something Polymarket published.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pmbtc.constants import SettlementSource
from pmbtc.exceptions import DataError
from pmbtc.settlement.providers import (
    ProviderDescriptor,
    canonical_pair,
    descriptor_for,
    identify_by_text,
    identify_by_url,
)
from pmbtc.settlement.spec import (
    EvidenceSource,
    FieldEvidence,
    SettlementSpecification,
    TieRule,
)
from pmbtc.utils.timeutils import to_ms, utc_now_ms

PARSER_VERSION = "2.0"

#: ``btc-updown-5m-1785503100`` -> interval token and window-open epoch.
_SLUG_RE = re.compile(r"(?P<interval>\d+\s*[mhd])-(?P<epoch>\d{9,13})$", re.IGNORECASE)

_RECURRENCE_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "hourly": 3_600,
    "4h": 14_400,
    "1d": 86_400,
    "daily": 86_400,
    "weekly": 604_800,
}

_INTERVAL_TOKEN_SECONDS: dict[str, int] = {"m": 60, "h": 3_600, "d": 86_400}

#: Pair mentions as they appear in rules prose.
_PAIR_RE = re.compile(
    r"\b(?:btc|xbt)\s*[/_-]?\s*(?:usdt|usdc|usd)\b", re.IGNORECASE
)

#: Tie-rule dialects. Order matters: an explicit 50-50 clause outranks a
#: ">=" clause, because markets that offer both state the 50-50 case last.
_TIE_DIALECTS: tuple[tuple[TieRule, tuple[str, ...]], ...] = (
    (TieRule.TIE_FIFTY_FIFTY, ("50-50",)),
    (TieRule.TIE_FIFTY_FIFTY, ("50/50",)),
    (TieRule.TIE_UP, ("greater than or equal to",)),
    (TieRule.TIE_UP, ("at or above",)),
    (TieRule.TIE_DOWN, ("less than or equal to",)),
    (TieRule.TIE_DOWN, ("at or below",)),
)

_REFERENCE_DIALECTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("range_open_price", ("price at the beginning of that range",)),
    ("candle_open_price", ("open price for the",)),
    ("candle_open_price", ("open price of the",)),
    ("fixed_strike", ("higher than", "lower than")),
)


class SettlementParseError(DataError):
    """The Gamma payload could not be read as a market at all."""


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _event_of(market: dict[str, Any]) -> dict[str, Any]:
    events = market.get("events") or []
    return events[0] if events and isinstance(events[0], dict) else {}


def _series_of(event: dict[str, Any]) -> dict[str, Any]:
    series = event.get("series") or []
    return series[0] if series and isinstance(series[0], dict) else {}


def _parse_outcomes(raw: Any) -> tuple[str, ...]:
    """``outcomes`` arrives as a JSON-encoded string, not a list."""
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ()
        if isinstance(parsed, list):
            return tuple(str(x) for x in parsed)
    return ()


def _excerpt(text: str, needle: str, width: int = 90) -> str:
    """A short verbatim window around a matched phrase, for the audit trail."""
    lowered = text.lower()
    idx = lowered.find(needle.lower())
    if idx < 0:
        return text[:width].strip()
    start = max(0, idx - width // 3)
    return text[start : start + width].strip()


# --------------------------------------------------------------------------- #
# Field detectors
# --------------------------------------------------------------------------- #
def _detect_provider(
    market: dict[str, Any], event: dict[str, Any]
) -> tuple[SettlementSource, str, ProviderDescriptor | None, FieldEvidence]:
    """Provider from the structured URL, cross-examined against the prose."""
    url = _first_non_empty(market.get("resolutionSource"), event.get("resolutionSource"))
    description = _first_non_empty(market.get("description"), event.get("description"))

    by_url = identify_by_url(url) if url else []
    by_text = identify_by_text(description) if description else []

    if len(by_url) > 1:
        names = ", ".join(d.display_name for d in by_url)
        return (
            SettlementSource.UNKNOWN,
            url,
            None,
            FieldEvidence(
                source=EvidenceSource.STRUCTURED_FIELD,
                confidence=0.0,
                locator="market.resolutionSource",
                excerpt=url[:120],
                conflict=f"resolutionSource matches multiple providers: {names}",
            ),
        )

    if by_url and by_text:
        structured, textual = by_url[0], by_text
        if len(textual) == 1 and textual[0].source is structured.source:
            return (
                structured.source,
                url,
                structured,
                FieldEvidence(
                    source=EvidenceSource.STRUCTURED_FIELD,
                    confidence=1.0,
                    locator="market.resolutionSource + market.description",
                    excerpt=url[:120],
                ),
            )
        found = ", ".join(d.display_name for d in textual) or "none"
        return (
            SettlementSource.UNKNOWN,
            url,
            None,
            FieldEvidence(
                source=EvidenceSource.STRUCTURED_FIELD,
                confidence=0.0,
                locator="market.resolutionSource vs market.description",
                excerpt=url[:120],
                conflict=(
                    f"structured source says {structured.display_name}; "
                    f"rules text says {found}"
                ),
            ),
        )

    if by_url:
        # Structured only. Credible, but nothing corroborates it, so it cannot
        # clear a 1.0 gate.
        return (
            by_url[0].source,
            url,
            by_url[0],
            FieldEvidence(
                source=EvidenceSource.STRUCTURED_FIELD,
                confidence=0.9,
                locator="market.resolutionSource",
                excerpt=url[:120],
                conflict="",
            ),
        )

    if len(by_text) == 1:
        # Prose only -- e.g. older markets with an empty resolutionSource.
        # Deliberately capped below any tradeable threshold.
        return (
            by_text[0].source,
            url,
            by_text[0],
            FieldEvidence(
                source=EvidenceSource.TEXT_PATTERN,
                confidence=0.7,
                locator="market.description",
                excerpt=_excerpt(description, by_text[0].venue),
            ),
        )

    if len(by_text) > 1:
        names = ", ".join(d.display_name for d in by_text)
        return (
            SettlementSource.UNKNOWN,
            url,
            None,
            FieldEvidence(
                source=EvidenceSource.TEXT_PATTERN,
                confidence=0.0,
                locator="market.description",
                conflict=f"rules text names multiple providers: {names}",
            ),
        )

    return (
        SettlementSource.UNKNOWN,
        url,
        None,
        FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="market.resolutionSource",
            conflict="no settlement provider could be identified",
        ),
    )


def _detect_pair(
    description: str, descriptor: ProviderDescriptor | None
) -> tuple[str, FieldEvidence]:
    """Trading pair from prose, validated against what the provider publishes."""
    if descriptor is None:
        return "", FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="market.description",
            conflict="no provider, so no pair authority",
        )

    mentions = {
        canonical_pair(m.group(0)) for m in _PAIR_RE.finditer(description or "")
    } - {None}

    if not mentions:
        return descriptor.default_pair, FieldEvidence(
            source=EvidenceSource.PROVIDER_DEFAULT,
            confidence=0.9,
            locator="provider_descriptor.default_pair",
            excerpt=descriptor.default_pair,
            conflict="",
        )

    supported = {p for p in mentions if p and descriptor.supports(p)}
    if len(supported) == 1 and len(mentions) == 1:
        pair = supported.pop()
        return pair, FieldEvidence(
            source=EvidenceSource.TEXT_PATTERN,
            confidence=1.0,
            locator="market.description",
            excerpt=_excerpt(description, pair.split("/")[0]),
        )
    if not supported:
        listed = ", ".join(sorted(str(m) for m in mentions))
        return "", FieldEvidence(
            source=EvidenceSource.TEXT_PATTERN,
            confidence=0.0,
            locator="market.description",
            conflict=(
                f"rules text names {listed}, which {descriptor.display_name} "
                "is not an authority for"
            ),
        )
    listed = ", ".join(sorted(str(m) for m in mentions))
    return "", FieldEvidence(
        source=EvidenceSource.TEXT_PATTERN,
        confidence=0.0,
        locator="market.description",
        conflict=f"rules text names multiple pairs: {listed}",
    )


def _interval_from_slug(slug: str) -> int | None:
    match = _SLUG_RE.search(slug or "")
    if not match:
        return None
    token = match.group("interval").replace(" ", "").lower()
    return int(token[:-1]) * _INTERVAL_TOKEN_SECONDS[token[-1]]


def _epoch_from_slug(slug: str) -> int | None:
    match = _SLUG_RE.search(slug or "")
    if not match:
        return None
    raw = int(match.group("epoch"))
    return raw * 1000 if raw < 1e11 else raw


def _detect_interval(
    market: dict[str, Any], series: dict[str, Any], derived_seconds: int | None
) -> tuple[int, FieldEvidence]:
    """Interval from the series recurrence, checked against slug and timestamps."""
    recurrence = str(series.get("recurrence") or "").strip().lower()
    from_series = _RECURRENCE_SECONDS.get(recurrence)
    from_slug = _interval_from_slug(str(market.get("slug") or ""))

    candidates = {
        name: value
        for name, value in (
            ("series.recurrence", from_series),
            ("slug", from_slug),
            ("endDate-startTime", derived_seconds),
        )
        if value
    }
    if not candidates:
        return 0, FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="events[0].series[0].recurrence",
            conflict="no interval could be established",
        )

    distinct = set(candidates.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{k}={v}s" for k, v in sorted(candidates.items()))
        return 0, FieldEvidence(
            source=EvidenceSource.STRUCTURED_SERIES,
            confidence=0.0,
            locator="series.recurrence vs slug vs timestamps",
            conflict=f"interval sources disagree: {detail}",
        )

    value = distinct.pop()
    # 1.0 needs at least two independent channels agreeing.
    confidence = 1.0 if len(candidates) >= 2 else 0.9
    return value, FieldEvidence(
        source=EvidenceSource.STRUCTURED_SERIES
        if from_series
        else (EvidenceSource.SLUG if from_slug else EvidenceSource.DERIVED),
        confidence=confidence,
        locator=" + ".join(sorted(candidates)),
        excerpt=f"{value}s",
    )


def _detect_times(
    market: dict[str, Any], event: dict[str, Any]
) -> tuple[int, int, FieldEvidence, FieldEvidence]:
    """Window open and settlement instant, in UTC epoch millis.

    Market *titles* carry an ET range ("9:05AM-9:10AM ET"). That string is never
    used for arithmetic: the structured fields are already UTC and unambiguous,
    and a DST bug in a hand-rolled ET conversion would silently shift every
    label by an hour.
    """
    raw_open = _first_non_empty(event.get("startTime"), market.get("eventStartTime"))
    raw_close = _first_non_empty(market.get("endDate"), event.get("endDate"))
    slug_open = _epoch_from_slug(str(market.get("slug") or ""))

    open_ms = to_ms(raw_open) if raw_open else None
    close_ms = to_ms(raw_close) if raw_close else None

    if open_ms is not None and slug_open is not None:
        if open_ms == slug_open:
            open_ev = FieldEvidence(
                source=EvidenceSource.STRUCTURED_TIME,
                confidence=1.0,
                locator="events[0].startTime + slug epoch",
                excerpt=raw_open,
            )
        else:
            open_ev = FieldEvidence(
                source=EvidenceSource.STRUCTURED_TIME,
                confidence=0.0,
                locator="events[0].startTime vs slug epoch",
                excerpt=raw_open,
                conflict=f"startTime={open_ms} but slug encodes {slug_open}",
            )
            open_ms = None
    elif open_ms is not None:
        open_ev = FieldEvidence(
            source=EvidenceSource.STRUCTURED_TIME,
            confidence=0.9,
            locator="events[0].startTime",
            excerpt=raw_open,
        )
    elif slug_open is not None:
        open_ms = slug_open
        open_ev = FieldEvidence(
            source=EvidenceSource.SLUG,
            confidence=0.7,
            locator="market.slug",
            excerpt=str(market.get("slug")),
        )
    else:
        open_ev = FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="events[0].startTime",
            conflict="no window open time published",
        )

    if close_ms is not None:
        close_ev = FieldEvidence(
            source=EvidenceSource.STRUCTURED_TIME,
            confidence=1.0,
            locator="market.endDate",
            excerpt=raw_close,
        )
    else:
        close_ev = FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="market.endDate",
            conflict="no settlement timestamp published",
        )

    return open_ms or 0, close_ms or 0, open_ev, close_ev


def _detect_tie_rule(description: str) -> tuple[TieRule, FieldEvidence]:
    """Tie handling. Exists only in prose, so it needs an exact dialect match."""
    text = (description or "").lower()
    hits = [(rule, phrase) for rule, phrases in _TIE_DIALECTS for phrase in phrases
            if phrase in text]
    if not hits:
        return TieRule.UNKNOWN, FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="market.description",
            conflict="no recognised tie-resolution clause",
        )
    distinct = {rule for rule, _ in hits}
    if len(distinct) > 1:
        # Note the ordering of _TIE_DIALECTS: a market stating both ">=" and a
        # 50-50 clause is genuinely ambiguous to us and must not be traded.
        names = ", ".join(sorted(r.value for r in distinct))
        return TieRule.UNKNOWN, FieldEvidence(
            source=EvidenceSource.TEXT_PATTERN,
            confidence=0.0,
            locator="market.description",
            conflict=f"multiple tie dialects present: {names}",
        )
    rule, phrase = hits[0]
    return rule, FieldEvidence(
        source=EvidenceSource.TEXT_PATTERN,
        confidence=1.0,
        locator="market.description",
        excerpt=_excerpt(description, phrase),
    )


def _detect_reference_rule(description: str) -> tuple[str, FieldEvidence]:
    text = (description or "").lower()
    for name, phrases in _REFERENCE_DIALECTS:
        if all(p in text for p in phrases):
            return name, FieldEvidence(
                source=EvidenceSource.TEXT_PATTERN,
                confidence=1.0,
                locator="market.description",
                excerpt=_excerpt(description, phrases[0]),
            )
    return "", FieldEvidence(
        source=EvidenceSource.MISSING,
        confidence=0.0,
        locator="market.description",
        conflict="no recognised reference-price clause",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_settlement_spec(
    market: dict[str, Any], *, now_ms: int | None = None
) -> SettlementSpecification:
    """Build a canonical specification from a raw Gamma market payload.

    Never raises on a *merely unverifiable* market: it returns a spec whose
    evidence records exactly which fields are missing or conflicted, so the
    verifier can reject it with a reason and the report can show why. Raises only
    when the payload is not a market at all.
    """
    if not isinstance(market, dict) or not market.get("conditionId"):
        raise SettlementParseError(
            "Payload is not a Gamma market (no conditionId)",
            context={"keys": sorted(market)[:10] if isinstance(market, dict) else type(market)},
        )

    event = _event_of(market)
    series = _series_of(event)
    description = _first_non_empty(market.get("description"), event.get("description"))

    provider, url, descriptor, provider_ev = _detect_provider(market, event)
    pair, pair_ev = _detect_pair(description, descriptor)
    open_ms, close_ms, open_ev, close_ev = _detect_times(market, event)
    derived = (close_ms - open_ms) // 1000 if open_ms and close_ms and close_ms > open_ms else None
    interval_s, interval_ev = _detect_interval(market, series, derived)
    tie_rule, tie_ev = _detect_tie_rule(description)
    reference_rule, reference_ev = _detect_reference_rule(description)

    if descriptor is not None:
        decimals_ev = FieldEvidence(
            source=EvidenceSource.PROVIDER_DEFAULT,
            confidence=1.0,
            locator=f"provider_descriptor[{provider.value}].price_decimals",
            excerpt=f"{descriptor.price_decimals} decimals, {descriptor.rounding_mode}",
        )
        price_decimals = descriptor.price_decimals
        rounding_mode = descriptor.rounding_mode
        semantics = descriptor.timestamp_semantics
        venue = descriptor.venue
    else:
        decimals_ev = FieldEvidence(
            source=EvidenceSource.MISSING,
            confidence=0.0,
            locator="provider_descriptor",
            conflict="no provider, so no declared rounding rule",
        )
        price_decimals, rounding_mode, semantics, venue = 0, "", "", ""

    return SettlementSpecification(
        market_id=str(market.get("id") or ""),
        condition_id=str(market.get("conditionId")),
        slug=str(market.get("slug") or ""),
        question=str(market.get("question") or event.get("title") or ""),
        series_slug=str(series.get("slug") or event.get("seriesSlug") or ""),
        provider=provider,
        venue=venue,
        trading_pair=pair,
        resolution_source_url=url,
        interval_seconds=interval_s,
        window_open_ms=open_ms,
        window_close_ms=close_ms,
        timezone="UTC",
        tie_rule=tie_rule,
        reference_rule=reference_rule,
        price_decimals=price_decimals,
        rounding_mode=rounding_mode,
        timestamp_semantics=semantics,
        outcomes=_parse_outcomes(market.get("outcomes")) or ("Up", "Down"),
        tick_size=_as_float(market.get("orderPriceMinTickSize")),
        min_order_size=_as_float(market.get("orderMinSize")),
        fees_enabled=market.get("feesEnabled") if isinstance(market.get("feesEnabled"), bool)
        else None,
        evidence={
            "provider": provider_ev,
            "trading_pair": pair_ev,
            "interval_seconds": interval_ev,
            "window_open_ms": open_ev,
            "window_close_ms": close_ev,
            "tie_rule": tie_ev,
            "reference_rule": reference_ev,
            "price_decimals": decimals_ev,
        },
        detected_at_ms=now_ms if now_ms is not None else utc_now_ms(),
        parser_version=PARSER_VERSION,
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def descriptor_or_none(source: SettlementSource) -> ProviderDescriptor | None:
    try:
        return descriptor_for(source)
    except Exception:
        return None
