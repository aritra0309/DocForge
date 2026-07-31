"""Integration tests for CrawlEngine against a static fixture site."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx

from docforge.core.config import CrawlerConfig
from docforge.core.models import DiscoveryResult
from docforge.crawler.cache import ResponseCache
from docforge.crawler.engine import CrawlEngine
from tests.fixtures.site import (
    FIXTURE_SITE_BASE as BASE,
    FIXTURE_SITE_PAGE_COUNT,
    mock_fixture_site,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_crawl_fixture_site(tmp_path: Path) -> None:
    """Crawl a multi-page fixture site respecting filters and robots.txt."""
    robots = "User-agent: *\nDisallow: /docs/private/\n"
    mock_fixture_site(
        index_paths=("/docs/index.html",),
        robots_status=200,
        robots_body=robots,
        private_status=200,
    )

    config = CrawlerConfig(max_pages_per_version=50, rate_limit_rps=100)
    cache = ResponseCache(db_path=tmp_path / "cache.db")
    queue_db = tmp_path / "queue.db"
    engine = CrawlEngine(config=config, cache=cache, queue_db_path=queue_db)

    discovery = DiscoveryResult(
        software="fixture",
        display_name="Fixture",
        base_url=f"{BASE}/docs/index.html",
        versions=["1"],
        latest_version="1",
        url_filters={"include": ["/docs/**"]},
    )

    results = await engine.crawl(f"{BASE}/docs/index.html", discovery_result=discovery)

    urls = {r.url for r in results}
    assert f"{BASE}/docs/index.html" in urls
    assert f"{BASE}/docs/page1.html" in urls
    assert f"{BASE}/docs/page2.html" in urls
    assert f"{BASE}/docs/page20.html" in urls
    assert len(results) >= FIXTURE_SITE_PAGE_COUNT
    assert f"{BASE}/docs/private/secret.html" not in urls
    assert f"{BASE}/blog/post.html" not in urls

    engine.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_crawl_resume_from_checkpoint(tmp_path: Path) -> None:
    """Interrupted crawl resumes from persisted queue state."""
    mock_fixture_site(index_paths=("/docs/index.html",))

    config = CrawlerConfig(max_pages_per_version=100, rate_limit_rps=100)
    cache = ResponseCache(db_path=tmp_path / "cache.db")
    queue_db = tmp_path / "queue.db"

    engine1 = CrawlEngine(config=config, cache=cache, queue_db_path=queue_db)
    discovery = DiscoveryResult(
        software="fixture",
        display_name="Fixture",
        base_url=f"{BASE}/docs/index.html",
        versions=["1"],
        latest_version="1",
        url_filters={"include": ["/docs/**"]},
    )

    partial = await engine1.crawl(
        f"{BASE}/docs/index.html",
        discovery_result=discovery,
        max_pages=1,
    )
    assert len(partial) == 1
    stats = await engine1.get_queue_stats()
    assert stats.get("completed", 0) >= 1
    engine1.close()

    engine2 = CrawlEngine(config=config, cache=cache, queue_db_path=queue_db)
    resumed = await engine2.crawl(
        f"{BASE}/docs/index.html",
        discovery_result=discovery,
        resume=True,
    )
    urls = {r.url for r in resumed}
    assert len(resumed) >= 2
    assert f"{BASE}/docs/page1.html" in urls or f"{BASE}/docs/page2.html" in urls
    engine2.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_cache_avoids_http_on_second_crawl(tmp_path: Path) -> None:
    """Second crawl run serves pages from cache without additional HTTP requests."""
    routes = mock_fixture_site(index_paths=("/docs/index.html",))
    route_index = routes["index"]
    route_page1 = routes["page1"]

    config = CrawlerConfig(max_pages_per_version=5, rate_limit_rps=100, cache_ttl_hours=24)
    cache = ResponseCache(db_path=tmp_path / "cache.db")

    discovery = DiscoveryResult(
        software="fixture",
        display_name="Fixture",
        base_url=f"{BASE}/docs/index.html",
        versions=["1"],
        latest_version="1",
        url_filters={"include": ["/docs/**"]},
    )

    engine1 = CrawlEngine(config=config, cache=cache)
    await engine1.crawl(f"{BASE}/docs/index.html", discovery_result=discovery)
    calls_after_first = route_index.call_count + route_page1.call_count  # type: ignore[union-attr]
    engine1.close()

    engine2 = CrawlEngine(config=config, cache=cache)
    await engine2.crawl(f"{BASE}/docs/index.html", discovery_result=discovery)
    calls_after_second = route_index.call_count + route_page1.call_count  # type: ignore[union-attr]
    engine2.close()

    assert calls_after_second == calls_after_first
