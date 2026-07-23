"""robots.txt parser — identifies disallowed paths and crawl delay."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

# HTTP status codes to treat as "robots.txt not available"
_ROBOTS_UNAVAILABLE_CODES = frozenset({404, 401, 403})


class RobotsError(Exception):
    """Raised when robots.txt cannot be fetched or parsed."""


@dataclass
class RobotsPolicy:
    """Parsed robots.txt policy for a domain."""

    crawl_delay: float | None = None
    disallowed_paths: list[str] = field(default_factory=list)
    sitemap_urls: list[str] = field(default_factory=list)
    _raw_lines: list[str] = field(default_factory=list, repr=False)

    def is_allowed(self, path: str, user_agent: str = "*") -> bool:
        """Check if a path is allowed for the given user agent.

        Args:
            path: The URL path to check (e.g. '/docs/17/').
            user_agent: The user agent string to match against.

        Returns:
            True if the path is allowed, False if disallowed.
        """
        for pattern in self.disallowed_paths:
            if path.startswith(pattern):
                return False
        return True


async def fetch_robots_txt(
    base_url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = 10.0,
) -> RobotsPolicy:
    """Fetch and parse robots.txt for a given base URL.

    Args:
        base_url: The root URL of the documentation site.
        client: Optional pre-configured httpx client. Creates one if None.
        timeout: Request timeout in seconds.

    Returns:
        A RobotsPolicy with parsed rules.
    """
    robots_url = _build_robots_url(base_url)
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    assert client is not None

    try:
        response = await client.get(robots_url, headers={"User-Agent": "DocForge/0.1"})
        if response.status_code in _ROBOTS_UNAVAILABLE_CODES:
            return RobotsPolicy()
        response.raise_for_status()
        return parse_robots_txt(response.text)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in _ROBOTS_UNAVAILABLE_CODES:
            return RobotsPolicy()
        msg = f"Failed to fetch robots.txt: {e}"
        raise RobotsError(msg) from e
    finally:
        if own_client:
            await client.aclose()


def parse_robots_txt(content: str) -> RobotsPolicy:
    """Parse robots.txt content into a RobotsPolicy.

    Handles the most common robots.txt directives:
    - User-agent, Disallow, Allow, Crawl-delay, Sitemap

    Args:
        content: Raw text content of robots.txt.

    Returns:
        A parsed RobotsPolicy.
    """
    policy = RobotsPolicy(_raw_lines=content.splitlines())
    current_agent: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r"^([^:]+):\s*(.*)", line)
        if not match:
            continue

        directive = match.group(1).strip().lower()
        value = match.group(2).strip()

        if directive == "user-agent":
            current_agent = value
        elif directive == "disallow" and _agent_matches(current_agent):
            if value:
                policy.disallowed_paths.append(value)
        elif directive == "crawl-delay" and _agent_matches(current_agent):
            try:
                policy.crawl_delay = float(value)
            except ValueError:
                pass
        elif directive == "sitemap":
            policy.sitemap_urls.append(value)

    return policy


def _agent_matches(agent: str | None) -> bool:
    """Check if the user agent matches wildcard rules."""
    if agent is None:
        return False
    return agent == "*" or "docforge" in agent.lower()


def _build_robots_url(base_url: str) -> str:
    """Build the robots.txt URL from a base URL."""
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"
