from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from docforge.core.config import DocForgeConfig
from docforge.core.models import DiscoveryResult, FetchResult
from docforge.crawler.fetcher import HTTPFetcher
from docforge.discovery.sitemap import SitemapUrl, fetch_sitemap
from docforge.storage.metadata_store import MetadataStore

HTTP_NOT_MODIFIED = 304

logger = logging.getLogger(__name__)


@dataclass
class UpdateReport:
    """Categorised list of pages that changed between indexed state and current docs."""

    new_urls: list[str] = field(default_factory=list)
    changed_urls: list[str] = field(default_factory=list)
    removed_urls: list[str] = field(default_factory=list)
    unchanged_urls: list[str] = field(default_factory=list)
    new_fetch_results: list[FetchResult] = field(default_factory=list)
    changed_fetch_results: list[FetchResult] = field(default_factory=list)

    @property
    def total_changed(self) -> int:
        return len(self.new_urls) + len(self.changed_urls) + len(self.removed_urls)


class UpdateDetector:
    """Detects which documentation pages have changed since the last index.

    Uses a strategy cascade (cheapest first):
    1. Sitemap <lastmod> comparison — avoids fetching unchanged pages
    2. HTTP conditional requests (ETag / Last-Modified) — 304 = unchanged
    3. Content hash (ground truth) — handled downstream during chunking
    """

    def __init__(self, config: DocForgeConfig | None = None) -> None:
        self.config = config or DocForgeConfig()
        self._fetcher: HTTPFetcher | None = None

    @property
    def fetcher(self) -> HTTPFetcher:
        if self._fetcher is None:
            self._fetcher = HTTPFetcher(config=self.config.crawler)
        return self._fetcher

    async def detect(
        self,
        discovery_result: DiscoveryResult,
        software: str,
        version: str,
        metadata_store: MetadataStore,
    ) -> UpdateReport:
        """Detect changes between the current docs and the indexed state.

        Args:
            discovery_result: The resolved documentation source for this software.
            software: Software identifier.
            version: Target version to check.
            metadata_store: Store with previously indexed page states.

        Returns:
            An UpdateReport with URLs grouped by change status.
        """
        stored_pages = metadata_store.list_page_states(software, version)
        stored_by_url: dict[str, dict[str, Any]] = {
            p["url"]: p for p in stored_pages
        }

        sitemap_url = discovery_result.sitemap_url
        if sitemap_url:
            return await self._detect_via_sitemap(
                sitemap_url, stored_by_url, discovery_result, version,
            )

        return await self._detect_via_all_fetch(
            list(stored_by_url.keys()), stored_by_url, discovery_result, version,
        )

    async def _detect_via_sitemap(
        self,
        sitemap_url: str,
        stored_by_url: dict[str, dict[str, Any]],
        discovery_result: DiscoveryResult,
        version: str,
    ) -> UpdateReport:
        """Detect changes by comparing sitemap <lastmod> with stored state."""
        try:
            sitemap_entries = await fetch_sitemap(sitemap_url)
        except Exception as exc:
            logger.warning("Failed to fetch sitemap %s: %s", sitemap_url, exc)
            return await self._detect_via_all_fetch(
                list(stored_by_url.keys()), stored_by_url, discovery_result, version,
            )

        base = discovery_result.base_url.rstrip("/")
        current_by_url: dict[str, SitemapUrl] = {
            e.loc: e for e in sitemap_entries if e.loc.startswith(base)
        }

        report = UpdateReport()

        for url, sitemap_entry in current_by_url.items():
            stored = stored_by_url.pop(url, None)
            if stored is None:
                report.new_urls.append(url)
                continue
            sitemap_lastmod = sitemap_entry.lastmod
            stored_lastmod = stored.get("last_modified") or None
            if sitemap_lastmod and stored_lastmod:
                if sitemap_lastmod == stored_lastmod:
                    report.unchanged_urls.append(url)
                else:
                    report.changed_urls.append(url)
            elif sitemap_lastmod and not stored_lastmod:
                report.changed_urls.append(url)
            elif not sitemap_lastmod and stored_lastmod:
                report.changed_urls.append(url)
            else:
                report.changed_urls.append(url)

        report.removed_urls = list(stored_by_url.keys())

        if report.new_urls or report.changed_urls:
            await self._fetch_changed_and_new(report, discovery_result)

        return report

    async def _detect_via_all_fetch(
        self,
        stored_urls: list[str],
        stored_by_url: dict[str, dict[str, Any]],
        discovery_result: DiscoveryResult,
        version: str,
    ) -> UpdateReport:
        """Fallback: treat all URLs as potentially changed and use conditional requests."""
        report = UpdateReport()
        version_base = f"{discovery_result.base_url.rstrip('/')}/{version}"

        if not stored_urls:
            report.changed_urls = [version_base]
        else:
            for url in stored_urls:
                report.changed_urls.append(url)

        if report.changed_urls:
            changed_fetched, still_changed = await self._fetch_with_conditionals(
                report.changed_urls, stored_by_url,
            )
            report.changed_fetch_results = changed_fetched
            for url in still_changed:
                if url in report.changed_urls:
                    report.changed_urls.remove(url)
                    report.unchanged_urls.append(url)

        return report

    async def _fetch_changed_and_new(
        self,
        report: UpdateReport,
        discovery_result: DiscoveryResult,
    ) -> None:
        """Fetch changed and new URLs, confirming changes via conditional requests."""
        stored_lookup: dict[str, dict[str, Any]] = {}

        all_to_fetch: list[str] = []
        for url in report.changed_urls + report.new_urls:
            all_to_fetch.append(url)

        if not all_to_fetch:
            return

        changed_fetched, still_changed = await self._fetch_with_conditionals(
            all_to_fetch, stored_lookup,
        )

        report.changed_fetch_results = [
            r for r in changed_fetched
            if r.url in report.changed_urls
        ]
        report.new_fetch_results = [
            r for r in changed_fetched
            if r.url in report.new_urls
        ]

        for url in still_changed:
            if url in report.changed_urls:
                report.changed_urls.remove(url)
                report.unchanged_urls.append(url)
            elif url in report.new_urls:
                pass

    async def _fetch_with_conditionals(
        self,
        urls: list[str],
        stored_lookup: dict[str, dict[str, Any]],
    ) -> tuple[list[FetchResult], list[str]]:
        """Fetch a batch of URLs with conditional request headers.

        Returns:
            (fetch_results, unchanged_urls) where unchanged_urls are URLs that
            returned 304 Not Modified.
        """
        fetch_results: list[FetchResult] = []
        unchanged_urls: list[str] = []

        for url in urls:
            stored = stored_lookup.get(url)
            etag = stored.get("etag") if stored else None
            last_modified = stored.get("last_modified") if stored else None

            try:
                result = await self.fetcher.fetch(
                    url, etag=etag or None, last_modified=last_modified or None,
                )
                if result.status_code == HTTP_NOT_MODIFIED:
                    unchanged_urls.append(url)
                else:
                    fetch_results.append(result)
            except Exception as exc:
                logger.warning("Failed to fetch %s for update detection: %s", url, exc)

        return fetch_results, unchanged_urls

    async def close(self) -> None:
        if self._fetcher is not None:
            await self._fetcher.aclose()
            self._fetcher = None
