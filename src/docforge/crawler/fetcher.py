"""Single-URL async HTTP fetcher with rate limiting, retries, timeouts, and conditional requests."""

from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime

import httpx

from docforge.core.config import CrawlerConfig
from docforge.core.interfaces import CrawlFetcher
from docforge.core.models import FetchResult

DEFAULT_USER_AGENT = "DocForge/0.1 (+https://github.com/docforge/docforge)"


class FetchError(Exception):
    """Raised when a URL cannot be fetched after all retries."""


class TokenBucket:
    """Token bucket rate limiter for restricting requests per second per domain."""

    def __init__(self, rps: float) -> None:
        self.rps = max(0.1, rps)
        self.capacity = float(self.rps)
        self.tokens = float(self.rps)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, sleeping if necessary to respect the rate limit."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rps)
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rps
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


class HTTPFetcher(CrawlFetcher):
    """Async HTTP fetcher using httpx.AsyncClient."""

    def __init__(
        self,
        config: CrawlerConfig | None = None,
        client: httpx.AsyncClient | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.config = config or CrawlerConfig()
        self.user_agent = user_agent
        self._external_client = client
        self._client: httpx.AsyncClient | None = client
        self._rate_limiters: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self.config.timeout_seconds)),
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        return self._client

    async def _get_rate_limiter(self, host: str) -> TokenBucket:
        async with self._lock:
            if host not in self._rate_limiters:
                self._rate_limiters[host] = TokenBucket(float(self.config.rate_limit_rps))
            return self._rate_limiters[host]

    async def fetch(
        self,
        url: str,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch a single URL over HTTP with rate limiting and exponential retries.

        Args:
            url: The canonical URL to fetch.
            etag: Optional ETag for conditional If-None-Match requests.
            last_modified: Optional Last-Modified date for If-Modified-Since.

        Returns:
            A FetchResult containing response details.

        Raises:
            FetchError: If all retries fail.
        """
        client = await self._get_client()
        parsed = httpx.URL(url)
        limiter = await self._get_rate_limiter(parsed.host)
        await limiter.acquire()

        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        last_exception: Exception | None = None
        max_attempts = max(1, self.config.retry_attempts + 1)

        for attempt in range(max_attempts):
            try:
                response = await client.get(url, headers=headers)

                # 304 Not Modified is a valid response for conditional requests
                if response.status_code == 304:
                    return FetchResult(
                        url=str(response.url),
                        status_code=304,
                        html="",
                        headers=dict(response.headers),
                        etag=response.headers.get("etag") or etag,
                        last_modified=response.headers.get("last-modified") or last_modified,
                        fetched_at=datetime.now(UTC),
                    )

                # Retry on 5xx server errors or 429 Too Many Requests
                if response.status_code >= 500 or response.status_code == 429:
                    response.raise_for_status()

                return FetchResult(
                    url=str(response.url),
                    status_code=response.status_code,
                    html=response.text,
                    headers=dict(response.headers),
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                    fetched_at=datetime.now(UTC),
                )
            except (httpx.HTTPError, httpx.StreamError) as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    backoff = (2**attempt) * 0.5 + random.uniform(0, 0.1)
                    await asyncio.sleep(backoff)

        msg = f"Failed to fetch '{url}' after {max_attempts} attempt(s): {last_exception}"
        raise FetchError(msg) from last_exception

    async def aclose(self) -> None:
        """Close HTTP client if owned by this fetcher."""
        if self._client is not None and self._external_client is None:
            await self._client.aclose()
            self._client = None
