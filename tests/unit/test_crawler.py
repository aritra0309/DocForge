"""Unit tests for the crawling engine (fetcher, cache, filters, robots policy, crawl engine)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import respx

from docforge.core.config import CrawlerConfig
from docforge.core.models import DiscoveryResult, FetchResult
from docforge.crawler.cache import ResponseCache
from docforge.crawler.engine import CrawlEngine
from docforge.crawler.fetcher import FetchError, HTTPFetcher, TokenBucket
from docforge.crawler.filters import URLFilter, glob_to_regex, normalize_url
from docforge.crawler.robots_policy import RobotsPolicyEnforcer

# ---------------------------------------------------------------------------
# URL Normalisation and Filtering Tests
# ---------------------------------------------------------------------------


def test_normalize_url() -> None:
    """Test URL normalisation rules."""
    assert normalize_url("HTTP://EXAMPLE.COM/docs/") == "http://example.com/docs"
    assert normalize_url("http://example.com/docs/page#section") == "http://example.com/docs/page"
    assert normalize_url("http://example.com/search?b=2&a=1") == "http://example.com/search?a=1&b=2"
    assert normalize_url("https://example.com/") == "https://example.com/"


def test_glob_to_regex() -> None:
    """Test glob pattern translation to regular expressions."""
    regex = glob_to_regex("/docs/17/**")
    assert regex.search("/docs/17/index.html")
    assert regex.search("/docs/17/tutorial/step1.html")
    assert not regex.search("/docs/16/index.html")

    exclude_regex = glob_to_regex("**/release-*")
    assert exclude_regex.search("/docs/17/release-notes")
    assert not exclude_regex.search("/docs/17/tutorial")


def test_url_filter_heuristics_and_patterns() -> None:
    """Test URLFilter for domain enforcement, non-doc heuristics, and patterns."""
    url_filter = URLFilter(
        base_url="https://www.postgresql.org/docs/17/",
        include_patterns=["/docs/17/**"],
        exclude_patterns=["**/release-*"],
    )

    # Allowed
    assert url_filter.is_allowed("https://www.postgresql.org/docs/17/tutorial.html")
    assert url_filter.is_allowed("https://www.postgresql.org/docs/17/datatype.html")

    # Disallowed: different domain
    assert not url_filter.is_allowed("https://otherdomain.com/docs/17/tutorial.html")

    # Disallowed: non-doc heuristics
    assert not url_filter.is_allowed("https://www.postgresql.org/blog/news.html")
    assert not url_filter.is_allowed("https://www.postgresql.org/docs/17/diagram.png")
    assert not url_filter.is_allowed("https://www.postgresql.org/docs/17/doc.pdf")

    # Disallowed: excluded pattern
    assert not url_filter.is_allowed("https://www.postgresql.org/docs/17/release-17.1")

    # Disallowed: outside include pattern
    assert not url_filter.is_allowed("https://www.postgresql.org/docs/16/tutorial.html")


def test_url_filter_extract_links() -> None:
    """Test link extraction from HTML."""
    url_filter = URLFilter(base_url="https://example.com/docs/")

    html = """
    <html>
        <body>
            <a href="/docs/page1.html">Page 1</a>
            <a href="https://example.com/docs/page2.html">Page 2</a>
            <a href="/blog/post1">Blog Post</a>
            <a href="https://other.com/docs">External</a>
            <a href="/docs/image.png">Image</a>
        </body>
    </html>
    """

    links = url_filter.extract_links(html, "https://example.com/docs/index.html")
    assert links == [
        "https://example.com/docs/page1.html",
        "https://example.com/docs/page2.html",
    ]


# ---------------------------------------------------------------------------
# ResponseCache Tests
# ---------------------------------------------------------------------------


def test_response_cache_put_and_get(tmp_path: pytest.TempPathFactory) -> None:
    """Test putting and retrieving items from the SQLite response cache."""
    db_file = tmp_path / "cache.db"
    cache = ResponseCache(db_path=db_file)

    now = datetime.now(UTC)
    result = FetchResult(
        url="https://example.com/page",
        status_code=200,
        html="<html>Hello World</html>",
        headers={"content-type": "text/html"},
        etag='"12345"',
        last_modified="Wed, 21 Oct 2025 07:28:00 GMT",
        fetched_at=now,
    )

    cache.put(result.url, result, ttl_hours=24)

    cached = cache.get("https://example.com/page")
    assert cached is not None
    assert cached.url == "https://example.com/page"
    assert cached.status_code == 200
    assert cached.html == "<html>Hello World</html>"
    assert cached.etag == '"12345"'
    assert cached.headers == {"content-type": "text/html"}

    cache.close()


def test_response_cache_expiration(tmp_path: pytest.TempPathFactory) -> None:
    """Test that expired cache entries return None for get() but are available via get_stale()."""
    db_file = tmp_path / "cache.db"
    cache = ResponseCache(db_path=db_file)

    past = datetime.now(UTC) - timedelta(hours=5)
    result = FetchResult(
        url="https://example.com/old",
        status_code=200,
        html="Old Content",
        etag='"old-etag"',
        fetched_at=past,
    )

    cache.put(result.url, result, ttl_hours=1)  # expired 4 hours ago

    assert cache.get("https://example.com/old") is None

    stale = cache.get_stale("https://example.com/old")
    assert stale is not None
    assert stale.etag == '"old-etag"'

    cache.close()


# ---------------------------------------------------------------------------
# HTTPFetcher & RateLimiter Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter() -> None:
    """Test TokenBucket rate limiter timing."""
    bucket = TokenBucket(rps=10.0)  # 10 rps -> 0.1s per token
    start = asyncio.get_event_loop().time()
    for _ in range(3):
        await bucket.acquire()
    elapsed = asyncio.get_event_loop().time() - start
    # Acquire first tokens should be near instantaneous
    assert elapsed < 0.5


@pytest.mark.asyncio
@respx.mock
async def test_fetcher_success() -> None:
    """Test successful page fetch with HTTPFetcher."""
    respx.get("https://example.com/doc").respond(
        status_code=200,
        text="<h1>Doc Page</h1>",
        headers={"ETag": '"abc"', "Last-Modified": "Thu, 1 Jan 2026 00:00:00 GMT"},
    )

    config = CrawlerConfig(rate_limit_rps=100)
    fetcher = HTTPFetcher(config=config)

    res = await fetcher.fetch("https://example.com/doc")
    assert res.status_code == 200
    assert res.html == "<h1>Doc Page</h1>"
    assert res.etag == '"abc"'

    await fetcher.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetcher_conditional_304() -> None:
    """Test conditional HTTP fetch returning 304 Not Modified."""
    respx.get("https://example.com/doc").respond(status_code=304)

    config = CrawlerConfig(rate_limit_rps=100)
    fetcher = HTTPFetcher(config=config)

    res = await fetcher.fetch("https://example.com/doc", etag='"abc"')
    assert res.status_code == 304

    await fetcher.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetcher_retry_failure() -> None:
    """Test exponential retries exhaustion raising FetchError."""
    respx.get("https://example.com/fail").respond(status_code=500)

    config = CrawlerConfig(retry_attempts=2, rate_limit_rps=100)
    fetcher = HTTPFetcher(config=config)

    with pytest.raises(FetchError):
        await fetcher.fetch("https://example.com/fail")

    await fetcher.aclose()


# ---------------------------------------------------------------------------
# RobotsPolicyEnforcer Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_robots_policy_enforcer() -> None:
    """Test robots.txt enforcement and caching."""
    robots_content = """
    User-agent: *
    Disallow: /docs/private/
    """
    respx.get("https://example.com/robots.txt").respond(status_code=200, text=robots_content)

    enforcer = RobotsPolicyEnforcer(respect_robots_txt=True)

    assert await enforcer.is_allowed("https://example.com/docs/public/")
    assert not await enforcer.is_allowed("https://example.com/docs/private/")

    # Bypassing robots.txt
    enforcer_disabled = RobotsPolicyEnforcer(respect_robots_txt=False)
    assert await enforcer_disabled.is_allowed("https://example.com/docs/private/")


# ---------------------------------------------------------------------------
# CrawlEngine Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_crawl_engine_end_to_end() -> None:
    """Test CrawlEngine crawling multi-page fixture site, respecting link filter and cache."""
    page_root = """
    <html><body>
        <h1>PostgreSQL 17 Docs</h1>
        <a href="https://www.postgresql.org/docs/17/intro.html">Intro</a>
        <a href="https://www.postgresql.org/docs/17/tutorial.html">Tutorial</a>
        <a href="https://www.postgresql.org/blog/post">Blog Post</a>
    </body></html>
    """
    page_intro = "<html><body><h1>Intro</h1><a href='https://www.postgresql.org/docs/17/tutorial.html'>Tutorial</a></body></html>"
    page_tutorial = "<html><body><h1>Tutorial</h1></body></html>"

    respx.get("https://www.postgresql.org/robots.txt").respond(status_code=404)
    respx.get("https://www.postgresql.org/docs/17/index.html").respond(
        status_code=200, text=page_root
    )
    respx.get("https://www.postgresql.org/docs/17/intro.html").respond(
        status_code=200, text=page_intro
    )
    respx.get("https://www.postgresql.org/docs/17/tutorial.html").respond(
        status_code=200, text=page_tutorial
    )

    config = CrawlerConfig(max_pages_per_version=10, rate_limit_rps=100)
    cache = ResponseCache()
    engine = CrawlEngine(config=config, cache=cache)

    discovery = DiscoveryResult(
        software="postgresql",
        display_name="PostgreSQL",
        base_url="https://www.postgresql.org/docs/17/index.html",
        versions=["17"],
        latest_version="17",
        url_filters={"include": ["/docs/17/**"]},
    )

    results = await engine.crawl(
        "https://www.postgresql.org/docs/17/index.html", discovery_result=discovery
    )

    urls = {r.url for r in results}
    assert "https://www.postgresql.org/docs/17/index.html" in urls
    assert "https://www.postgresql.org/docs/17/intro.html" in urls
    assert "https://www.postgresql.org/docs/17/tutorial.html" in urls
    assert "https://www.postgresql.org/blog/post" not in urls

    # Second crawl run should hit cache
    engine_cached = CrawlEngine(config=config, cache=cache)
    cached_results = await engine_cached.crawl(
        "https://www.postgresql.org/docs/17/index.html", discovery_result=discovery
    )
    assert len(cached_results) == len(results)

    engine.close()
    engine_cached.close()


@pytest.mark.asyncio
@respx.mock
async def test_crawl_skips_robots_disallowed() -> None:
    """Crawl engine skips URLs disallowed by robots.txt."""
    robots = "User-agent: *\nDisallow: /docs/private/\n"
    respx.get("https://example.com/robots.txt").respond(status_code=200, text=robots)
    respx.get("https://example.com/docs/index.html").respond(
        status_code=200,
        text="""
        <html><body>
            <a href="https://example.com/docs/public.html">Public</a>
            <a href="https://example.com/docs/private/secret.html">Private</a>
        </body></html>
        """,
    )
    respx.get("https://example.com/docs/public.html").respond(
        status_code=200, text="<html><body>Public</body></html>"
    )
    respx.get("https://example.com/docs/private/secret.html").respond(
        status_code=200, text="<html><body>Secret</body></html>"
    )

    config = CrawlerConfig(max_pages_per_version=10, rate_limit_rps=100)
    engine = CrawlEngine(config=config)

    results = await engine.crawl("https://example.com/docs/index.html")
    urls = {r.url for r in results}

    assert "https://example.com/docs/public.html" in urls
    assert "https://example.com/docs/private/secret.html" not in urls
    engine.close()


@pytest.mark.asyncio
@respx.mock
async def test_crawl_resume_checkpoint(tmp_path: pytest.TempPathFactory) -> None:
    """Crawl resumes from SQLite queue checkpoint after partial run."""
    page_root = """
    <html><body>
        <a href="https://example.com/docs/page1.html">P1</a>
        <a href="https://example.com/docs/page2.html">P2</a>
    </body></html>
    """
    respx.get("https://example.com/robots.txt").respond(status_code=404)
    respx.get("https://example.com/docs/index.html").respond(status_code=200, text=page_root)
    respx.get("https://example.com/docs/page1.html").respond(
        status_code=200, text="<html><body>Page 1</body></html>"
    )
    respx.get("https://example.com/docs/page2.html").respond(
        status_code=200, text="<html><body>Page 2</body></html>"
    )

    config = CrawlerConfig(max_pages_per_version=10, rate_limit_rps=100)
    cache = ResponseCache(db_path=tmp_path / "cache.db")
    queue_db = tmp_path / "queue.db"

    engine1 = CrawlEngine(config=config, cache=cache, queue_db_path=queue_db)
    partial = await engine1.crawl("https://example.com/docs/index.html", max_pages=1)
    assert len(partial) == 1
    engine1.close()

    engine2 = CrawlEngine(config=config, cache=cache, queue_db_path=queue_db)
    resumed = await engine2.crawl("https://example.com/docs/index.html", resume=True)
    assert len(resumed) >= 2
    engine2.close()
