"""Async Gamma API client with archival capture.

Two responsibilities beyond "make the request":

1. **Archive every response.** A parser change must be replayable against the
   exact bytes that produced yesterday's interpretation (Module 3 requirement
   7). Capture happens here, at the boundary, so no caller can forget.

2. **Be a clock reference.** ``/time`` on the CLOB host returns epoch seconds,
   and every HTTP response carries a ``Date`` header. Both feed the clock
   service, which is why the client exposes ``server_time_ms``.

Retries are bounded and only applied to genuinely transient failures. A 4xx is
never retried: the request was wrong and repeating it just burns rate limit.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import httpx
import orjson

from pmbtc.config import Config
from pmbtc.exceptions import HttpStatusError, NetworkError, RateLimitError
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import http_requests
from pmbtc.utils.timeutils import isoformat, utc_now_ms

log = get_logger("pmbtc.gamma.client")

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class ResponseArchive:
    """Append-only capture of raw API responses, partitioned by UTC day."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def write(self, endpoint: str, params: dict[str, Any], payload: Any) -> Path | None:
        if not self.enabled:
            return None
        now = utc_now_ms()
        day = isoformat(now)[:10]
        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        safe = endpoint.strip("/").replace("/", "_") or "root"
        path = directory / f"{safe}-{now}.json"
        record = {
            "captured_at_ms": now,
            "endpoint": endpoint,
            "params": params,
            "payload": payload,
        }
        path.write_bytes(orjson.dumps(record, default=str))
        return path

    def days(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def files(self, day: str | None = None) -> list[Path]:
        if not self.root.exists():
            return []
        directories = [self.root / day] if day else [p for p in self.root.iterdir() if p.is_dir()]
        files: list[Path] = []
        for directory in directories:
            if directory.exists():
                files.extend(sorted(directory.glob("*.json")))
        return files


class GammaClient:
    """Thin async client for the Gamma and CLOB read APIs."""

    def __init__(self, config: Config, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.base_url = config.polymarket.gamma_base_url.rstrip("/")
        self.clob_url = config.polymarket.clob_base_url.rstrip("/")
        self.archive = ResponseArchive(
            config.resolved_path(config.gamma.archive_dir),
            enabled=config.gamma.archive_responses,
        )
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> GammaClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._client is None:
            http = self.config.http
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=http.connect_timeout_s,
                    read=http.read_timeout_s,
                    write=http.read_timeout_s,
                    pool=http.total_timeout_s,
                ),
                limits=httpx.Limits(
                    max_connections=http.max_connections,
                    max_keepalive_connections=http.max_keepalive,
                ),
                headers={"User-Agent": http.user_agent, "Accept": "application/json"},
                follow_redirects=True,
            )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    async def _request(
        self, base: str, endpoint: str, params: dict[str, Any] | None = None
    ) -> Any:
        await self.start()
        assert self._client is not None
        url = f"{base}{endpoint}"
        query = {k: v for k, v in (params or {}).items() if v is not None}
        http = self.config.http
        last_error: Exception | None = None

        for attempt in range(http.max_retries + 1):
            try:
                response = await self._client.get(url, params=query)
            except httpx.TimeoutException as exc:
                last_error = NetworkError(f"Timeout calling {url}: {exc}")
            except httpx.HTTPError as exc:
                last_error = NetworkError(f"Transport error calling {url}: {exc}")
            else:
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    last_error = RateLimitError(
                        f"Rate limited by {url}",
                        retry_after_s=float(retry_after) if retry_after else None,
                    )
                elif response.status_code in _RETRYABLE_STATUS:
                    last_error = HttpStatusError(
                        f"Retryable status from {url}",
                        status_code=response.status_code,
                        url=url,
                        body=response.text[:500],
                        retryable=True,
                    )
                elif response.is_error:
                    http_requests.inc(outcome="error", endpoint=endpoint)
                    raise HttpStatusError(
                        f"Request to {url} failed",
                        status_code=response.status_code,
                        url=url,
                        body=response.text[:500],
                        retryable=False,
                    )
                else:
                    http_requests.inc(outcome="ok", endpoint=endpoint)
                    payload = response.json()
                    self.archive.write(endpoint, query, payload)
                    return payload

            if attempt < http.max_retries:
                # Full jitter: synchronised retries from several coroutines would
                # otherwise arrive together and re-trip the same rate limit.
                backoff = min(http.backoff_base_s * (2**attempt), http.backoff_max_s)
                await asyncio.sleep(random.uniform(0, backoff))

        http_requests.inc(outcome="failed", endpoint=endpoint)
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ #
    async def events_by_series(
        self,
        series_slug: str,
        *,
        end_date_min: str | None = None,
        limit: int = 50,
        closed: bool = False,
        active: bool = True,
    ) -> list[dict[str, Any]]:
        """Upcoming events for a series.

        This is the discovery primitive: no slug is constructed, so a change in
        Polymarket's naming convention cannot break discovery. Verified live --
        ``end_date_min=<now>`` returns the in-progress window first, then each
        subsequent one in order.
        """
        payload = await self._request(
            self.base_url,
            "/events/pagination",
            {
                "series_slug": series_slug,
                "closed": str(closed).lower(),
                "active": str(active).lower(),
                "limit": limit,
                "end_date_min": end_date_min,
            },
        )
        if isinstance(payload, dict):
            data = payload.get("data")
            return list(data) if isinstance(data, list) else []
        return list(payload) if isinstance(payload, list) else []

    async def markets_by_slug(self, slug: str) -> list[dict[str, Any]]:
        """Fallback lookup by exact slug."""
        payload = await self._request(self.base_url, "/markets", {"slug": slug})
        return list(payload) if isinstance(payload, list) else []

    async def market_by_condition_id(
        self, condition_id: str, *, closed: bool | None = None
    ) -> dict[str, Any] | None:
        """Fetch one market by its on-chain condition id.

        ``/markets`` silently excludes closed markets unless ``closed=true`` is
        passed explicitly, so a settled market is invisible to an unfiltered
        query -- which is precisely the market label backfill needs. When
        ``closed`` is not specified we therefore try the closed set first and
        fall back to the open one, rather than trusting a default that hides
        exactly the rows we came for.
        """
        attempts: list[dict[str, Any]] = (
            [{"condition_ids": condition_id, "closed": str(closed).lower()}]
            if closed is not None
            else [
                {"condition_ids": condition_id, "closed": "true"},
                {"condition_ids": condition_id},
            ]
        )
        for params in attempts:
            payload = await self._request(self.base_url, "/markets", params)
            if isinstance(payload, list) and payload:
                return dict(payload[0])
        return None

    async def series(self, series_id: str) -> dict[str, Any]:
        payload = await self._request(self.base_url, f"/series/{series_id}")
        return dict(payload) if isinstance(payload, dict) else {}

    async def server_time_ms(self) -> int:
        """CLOB server time, for the clock service.

        The endpoint returns whole seconds, so this contributes a coarse but
        independent reference; Binance's millisecond endpoint is the precise one.
        """
        payload = await self._request(self.clob_url, "/time")
        if isinstance(payload, (int, float)):
            value = float(payload)
        elif isinstance(payload, dict):
            value = float(payload.get("time") or payload.get("serverTime") or 0)
        else:
            value = float(str(payload).strip().strip('"'))
        if value <= 0:
            raise NetworkError("CLOB /time returned no usable timestamp")
        return int(value * 1000) if value < 1e11 else int(value)
