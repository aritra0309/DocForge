"""Robots.txt enforcement wrapper for the crawling engine."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import httpx

from docforge.core.interfaces import CrawlFetcher
from docforge.discovery.robots import RobotsPolicy, fetch_robots_txt


class RobotsPolicyEnforcer:
    """Domain-level robots.txt policy enforcer and cache."""

    def __init__(
        self,
        fetcher: CrawlFetcher | None = None,
        respect_robots_txt: bool = True,
    ) -> None:
        self.fetcher = fetcher
        self.respect_robots_txt = respect_robots_txt
        self._policies: dict[str, RobotsPolicy] = {}
        self._lock = asyncio.Lock()

    async def get_policy(self, url: str) -> RobotsPolicy:
        """Fetch and cache RobotsPolicy for the domain of url."""
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()

        async with self._lock:
            if netloc in self._policies:
                return self._policies[netloc]

            base_url = f"{parsed.scheme}://{netloc}/"
            try:
                # If fetcher has an underlying client, pass it, otherwise let fetch_robots_txt handle it
                policy = await fetch_robots_txt(base_url)
            except Exception:
                # On error, default to empty policy (allow all)
                policy = RobotsPolicy()

            self._policies[netloc] = policy
            return policy

    async def is_allowed(self, url: str, user_agent: str = "DocForge/0.1") -> bool:
        """Check if crawling url is allowed by robots.txt policy.

        Args:
            url: The URL to check.
            user_agent: User agent string.

        Returns:
            True if allowed (or respect_robots_txt is False), False otherwise.
        """
        if not self.respect_robots_txt:
            return True

        parsed = urlparse(url)
        path = parsed.path or "/"
        policy = await self.get_policy(url)
        return policy.is_allowed(path, user_agent=user_agent)
