"""Async crawl engine orchestrator for documentation discovery and fetching."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from docforge.core.config import CrawlerConfig, DocForgeConfig
from docforge.core.models import DiscoveryResult, FetchResult
from docforge.crawler.cache import ResponseCache
from docforge.crawler.fetcher import HTTPFetcher
from docforge.crawler.filters import URLFilter, normalize_url
from docforge.crawler.robots_policy import RobotsPolicyEnforcer


class CrawlEngine:
    """Async crawl orchestrator managing URL queue, worker pool, caching, and robots policies."""

    def __init__(
        self,
        config: DocForgeConfig | CrawlerConfig | None = None,
        fetcher: HTTPFetcher | None = None,
        cache: ResponseCache | None = None,
        robots_enforcer: RobotsPolicyEnforcer | None = None,
        queue_db_path: str | Path = ":memory:",
    ) -> None:
        if isinstance(config, DocForgeConfig):
            self.crawler_config = config.crawler
            self.parallelism = config.general.parallelism
        elif isinstance(config, CrawlerConfig):
            self.crawler_config = config
            self.parallelism = 8
        else:
            self.crawler_config = CrawlerConfig()
            self.parallelism = 8

        self.fetcher = fetcher or HTTPFetcher(config=self.crawler_config)
        self._owns_cache = cache is None
        self.cache = cache or ResponseCache()
        self.robots_enforcer = robots_enforcer or RobotsPolicyEnforcer(
            fetcher=self.fetcher,
            respect_robots_txt=self.crawler_config.respect_robots_txt,
        )

        self.queue_db_path = str(queue_db_path)
        if self.queue_db_path != ":memory:":
            Path(self.queue_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.queue_db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._queue_lock = asyncio.Lock()
        self._init_queue_db()

    def _init_queue_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS crawl_queue (
                    url TEXT PRIMARY KEY,
                    depth INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    async def enqueue(self, url: str, depth: int = 0, priority: int = 0) -> bool:
        """Enqueue a URL if it has not been seen before.

        Returns:
            True if inserted, False if already in queue.
        """
        norm_url = normalize_url(url)
        now_str = datetime.now(UTC).isoformat()
        async with self._queue_lock:
            with self._conn:
                cursor = self._conn.cursor()
                cursor.execute("SELECT 1 FROM crawl_queue WHERE url = ?", (norm_url,))
                if cursor.fetchone() is not None:
                    return False
                cursor.execute(
                    """
                    INSERT INTO crawl_queue (url, depth, priority, status, created_at)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (norm_url, depth, priority, now_str),
                )
                return True

    async def pop_next(self) -> tuple[str, int] | None:
        """Pop the next pending URL with lowest depth and priority."""
        async with self._queue_lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT url, depth FROM crawl_queue
                WHERE status = 'pending'
                ORDER BY depth ASC, priority ASC, created_at ASC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row is None:
                return None
            url, depth = row[0], row[1]
            with self._conn:
                self._conn.execute(
                    "UPDATE crawl_queue SET status = 'processing' WHERE url = ?", (url,)
                )
            return url, depth

    async def mark_status(self, url: str, status: str) -> None:
        """Update URL status in queue."""
        async with self._queue_lock:
            with self._conn:
                self._conn.execute("UPDATE crawl_queue SET status = ? WHERE url = ?", (status, url))

    async def get_queue_stats(self) -> dict[str, int]:
        """Return counts of URLs by status."""
        async with self._queue_lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM crawl_queue GROUP BY status")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def clear_queue(self) -> None:
        """Remove all URLs from the crawl queue."""
        with self._conn:
            self._conn.execute("DELETE FROM crawl_queue")

    def prepare_resume(self) -> None:
        """Reset interrupted URLs so a crawl can continue from checkpoint."""
        with self._conn:
            self._conn.execute(
                "UPDATE crawl_queue SET status = 'pending' WHERE status = 'processing'"
            )

    async def crawl(
        self,
        seed_urls: str | list[str],
        discovery_result: DiscoveryResult | None = None,
        max_pages: int | None = None,
        *,
        resume: bool = False,
    ) -> list[FetchResult]:
        """Run async crawl starting from seed URLs up to max_pages or max_pages_per_version.

        Args:
            seed_urls: Seed URL string or list of seed URLs.
            discovery_result: Optional DiscoveryResult with base_url and url_filters.
            max_pages: Override for max_pages_per_version limit.
            resume: If True, continue from persisted queue state (resetting stuck
                ``processing`` URLs). If False, clear the queue and start fresh.

        Returns:
            List of successfully fetched FetchResult objects.
        """
        if isinstance(seed_urls, str):
            seeds = [seed_urls]
        else:
            seeds = list(seed_urls)

        if not seeds:
            return []

        if resume:
            self.prepare_resume()
        else:
            self.clear_queue()

        base_url = discovery_result.base_url if discovery_result else seeds[0]
        include_patterns = discovery_result.url_filters.get("include") if discovery_result else None
        exclude_patterns = discovery_result.url_filters.get("exclude") if discovery_result else None

        url_filter = URLFilter(
            base_url=base_url,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        limit = max_pages or self.crawler_config.max_pages_per_version

        if not resume:
            for seed in seeds:
                norm_seed = normalize_url(seed)
                if url_filter.is_allowed(norm_seed):
                    await self.enqueue(norm_seed, depth=0)

        fetched_results: list[FetchResult] = []
        fetched_urls: set[str] = set()
        count_lock = asyncio.Lock()
        fetched_count = 0
        active_workers = 0
        active_lock = asyncio.Lock()

        sem = asyncio.Semaphore(self.parallelism)

        async def worker() -> None:
            nonlocal fetched_count, active_workers
            async with active_lock:
                active_workers += 1
            try:
                while True:
                    async with count_lock:
                        if fetched_count >= limit:
                            break

                    item = await self.pop_next()
                    if item is None:
                        # Queue is empty — check if other workers are still active
                        # and may enqueue new URLs. If so, yield and retry.
                        async with active_lock:
                            others_running = active_workers > 1
                        if others_running:
                            await asyncio.sleep(0.01)
                            continue
                        break
                    url, depth = item

                    async with sem:
                        async with count_lock:
                            if fetched_count >= limit:
                                await self.mark_status(url, "pending")
                                break

                        result = await self._try_process_url(url, url_filter, depth)
                        if result is None:
                            continue

                        if result.url not in fetched_urls:
                            async with count_lock:
                                if fetched_count < limit:
                                    fetched_urls.add(result.url)
                                    fetched_results.append(result)
                                    fetched_count += 1
                            await self.mark_status(url, "completed")
                        else:
                            await self.mark_status(url, "completed")
            finally:
                async with active_lock:
                    active_workers -= 1

        workers = [asyncio.create_task(worker()) for _ in range(self.parallelism)]
        await asyncio.gather(*workers)

        return fetched_results

    async def _process_url(
        self,
        url: str,
        url_filter: URLFilter,
        depth: int,
    ) -> FetchResult | None:
        cached = self.cache.get(url)
        result: FetchResult | None = None

        if cached is not None:
            result = cached
        else:
            allowed = await self.robots_enforcer.is_allowed(url)
            if not allowed:
                return None

            stale = self.cache.get_stale(url)
            etag = stale.etag if stale else None
            last_modified = stale.last_modified if stale else None

            fetch_res = await self.fetcher.fetch(url, etag=etag, last_modified=last_modified)

            if fetch_res.status_code == 304 and stale is not None:
                result = FetchResult(
                    url=fetch_res.url,
                    status_code=200,
                    html=stale.html,
                    headers=fetch_res.headers,
                    etag=fetch_res.etag or stale.etag,
                    last_modified=fetch_res.last_modified or stale.last_modified,
                    fetched_at=fetch_res.fetched_at,
                )
                self.cache.put(url, result, ttl_hours=self.crawler_config.cache_ttl_hours)
            elif fetch_res.status_code == 200:
                result = fetch_res
                self.cache.put(url, result, ttl_hours=self.crawler_config.cache_ttl_hours)

        if result is not None and result.html:
            links = url_filter.extract_links(result.html, result.url)
            for link in links:
                await self.enqueue(link, depth=depth + 1)

        return result

    async def _try_process_url(
        self,
        url: str,
        url_filter: URLFilter,
        depth: int,
    ) -> FetchResult | None:
        try:
            return await self._process_url(url=url, url_filter=url_filter, depth=depth)
        except Exception:
            await self.mark_status(url, "failed")
            return None

    def close(self) -> None:
        """Close SQLite database connections."""
        self._conn.close()
        if self._owns_cache:
            self.cache.close()
