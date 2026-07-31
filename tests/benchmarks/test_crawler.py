"""Crawler benchmark tests."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import respx

from docforge.core.config import CrawlerConfig
from docforge.core.models import DiscoveryResult, FetchResult
from docforge.crawler.cache import ResponseCache
from docforge.crawler.engine import CrawlEngine
from docforge.crawler.fetcher import HTTPFetcher

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def crawler_config() -> CrawlerConfig:
    return CrawlerConfig(
        max_pages_per_version=100,
        rate_limit_rps=1000,  # High rate limit for benchmarking
        timeout_seconds=30,
        retry_attempts=0,
        respect_robots_txt=False,
    )


class MockFetcher(HTTPFetcher):
    """Fetcher that returns pre-loaded HTML without network calls."""

    def __init__(self, config: CrawlerConfig, pages: dict[str, str]):
        super().__init__(config)
        self._pages = pages
        self._call_count = 0

    async def fetch(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        self._call_count += 1
        html = self._pages.get(url, "<html><body>Not found</body></html>")
        return FetchResult(
            url=url,
            status_code=200,
            html=html,
            headers={"content-type": "text/html"},
        )


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_crawler_benchmark_cached_pages(crawler_config: CrawlerConfig) -> None:
    """Benchmark crawler with cached responses (no network)."""
    # Prepare test pages
    pages = {}
    for i in range(50):
        url = f"https://example.com/docs/page{i}.html"
        pages[url] = f"""
        <html><body>
            <article><h1>Page {i}</h1><p>Content for page {i}.</p></article>
            <a href="https://example.com/docs/page{i+1}.html">Next</a>
        </body></html>
        """

    fetcher = MockFetcher(crawler_config, pages)
    cache = ResponseCache()
    engine = CrawlEngine(config=crawler_config, cache=cache)

    # Replace fetcher
    engine._fetcher = fetcher

    discovery = DiscoveryResult(
        software="bench",
        display_name="Benchmark",
        base_url="https://example.com/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
    )

    # Warm up - first run populates cache
    await engine.crawl("https://example.com/docs/page0.html", discovery_result=discovery, max_pages=10)
    engine.close()

    # Benchmark - second run should hit cache
    cache2 = ResponseCache()
    engine2 = CrawlEngine(config=crawler_config, cache=cache2)
    engine2._fetcher = fetcher

    start = time.perf_counter()
    results = await engine2.crawl(
        "https://example.com/docs/page0.html",
        discovery_result=discovery,
        max_pages=50,
    )
    elapsed = time.perf_counter() - start

    pages_per_sec = len(results) / elapsed
    assert pages_per_sec >= 500, f"Crawler cached throughput {pages_per_sec:.1f} pages/sec below target 500"

    from tests.benchmarks import benchmark
    with benchmark("crawler_cached_pages_per_sec", len(results)):
        pass

    engine2.close()


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_crawler_benchmark_with_respx(respx, crawler_config: CrawlerConfig) -> None:
    """Benchmark crawler with respx mocked HTTP responses."""
    # Setup 20 pages
    base = "https://bench.example.com"
    for i in range(20):
        url = f"{base}/docs/page{i}.html"
        next_url = f"{base}/docs/page{i+1}.html" if i < 19 else ""
        html = f"""
        <html><body>
            <article><h1>Page {i}</h1><p>Content for page {i}.</p></article>
            {f'<a href="{next_url}">Next</a>' if next_url else ''}
        </body></html>
        """
        respx.get(url).respond(status_code=200, text=html)
    respx.get(f"{base}/robots.txt").respond(status_code=404)

    cache = ResponseCache()
    engine = CrawlEngine(config=crawler_config, cache=cache)

    discovery = DiscoveryResult(
        software="bench",
        display_name="Benchmark",
        base_url=f"{base}/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
    )

    start = time.perf_counter()
    results = await engine.crawl(
        f"{base}/docs/page0.html",
        discovery_result=discovery,
        max_pages=20,
    )
    elapsed = time.perf_counter() - start

    pages_per_sec = len(results) / elapsed
    # With mocked HTTP, we expect at least 200 pages/sec
    assert pages_per_sec >= 200, f"Crawler mocked HTTP throughput {pages_per_sec:.1f} pages/sec below target 200"

    from tests.benchmarks import benchmark
    with benchmark("crawler_mocked_http_pages_per_sec", len(results)):
        pass

    engine.close()


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_fetcher_benchmark_concurrent_requests(crawler_config: CrawlerConfig) -> None:
    """Benchmark fetcher concurrent request handling."""
    fetcher = HTTPFetcher(crawler_config)

    # Mock 100 concurrent requests
    urls = [f"https://example.com/docs/page{i}.html" for i in range(100)]

    async def mock_fetch(url: str) -> FetchResult:
        await asyncio.sleep(0.001)  # Simulate minimal network delay
        return FetchResult(
            url=url,
            status_code=200,
            html=f"<html><body><h1>Page</h1></body></html>",
            headers={"content-type": "text/html"},
        )

    fetcher._client = AsyncMock()
    fetcher._client.get = AsyncMock(side_effect=lambda url, **kw: mock_fetch(url))
    fetcher._client.aclose = AsyncMock()

    # Benchmark concurrent fetches
    start = time.perf_counter()
    tasks = [fetcher.fetch(url) for url in urls]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    req_per_sec = len(urls) / elapsed
    # Should handle at least 500 req/sec with concurrent requests
    assert req_per_sec >= 500, f"Fetcher concurrent throughput {req_per_sec:.1f} req/sec below target 500"

    from tests.benchmarks import benchmark
    with benchmark("fetcher_concurrent_req_per_sec", len(urls)):
        pass

    await fetcher.aclose()