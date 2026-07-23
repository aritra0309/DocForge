"""Discovery engine — finds documentation URLs from a name or direct URL input."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from docforge.core.models import DiscoveryResult
from docforge.discovery.heuristics import HeuristicCandidate, probe_heuristics
from docforge.discovery.registry import Registry, RegistryEntry, load_registry
from docforge.discovery.version_detector import detect_versions

# Exception messages are kept short per TRY003; details go in the exception class.
_NO_DISCOVERY_MSG = "Could not discover documentation"
_FAILED_ACCESS_MSG = "Failed to access documentation"


class DiscoveryError(Exception):
    """Raised when documentation cannot be discovered for the given input."""


class DiscoveryEngine:
    """Orchestrates documentation discovery from a name or URL.

    Supports two discovery modes:
    1. **Name-based**: Looks up the software in the registry, then falls back
       to heuristic URL probing.
    2. **URL-based**: Takes a direct documentation URL and probes it to build
       a DiscoveryResult.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the discovery engine.

        Args:
            registry: Pre-loaded registry. Loads from disk if None.
            client: Optional httpx client for network requests.
            timeout: Default request timeout in seconds.
        """
        self._registry = registry
        self._client = client
        self._timeout = timeout

    async def discover(self, name_or_url: str) -> DiscoveryResult:
        """Discover documentation for a software name or URL.

        Determines whether the input is a URL or a name string and routes
        to the appropriate discovery strategy.

        Args:
            name_or_url: Either a software name (e.g. 'postgresql') or a
                documentation URL (e.g. 'https://www.postgresql.org/docs/').

        Returns:
            A DiscoveryResult with base URL, versions, and content selectors.

        Raises:
            DiscoveryError: If documentation cannot be discovered.
        """
        if _is_url(name_or_url):
            return await self.discover_from_url(name_or_url)
        return await self.discover_from_name(name_or_url)

    async def discover_from_name(self, name: str) -> DiscoveryResult:
        """Discover documentation from a software name string.

        Strategy:
        1. Check registry -> if found, build DiscoveryResult immediately
        2. Apply URL heuristics -> try 5-6 common patterns
        3. Return highest-confidence candidate

        Args:
            name: Canonical software identifier (e.g. 'postgresql').

        Returns:
            A DiscoveryResult with base URL, versions, and content selectors.

        Raises:
            DiscoveryError: If software cannot be found via any strategy.
        """
        # Strategy 1: Registry lookup
        entry = self._get_registry().lookup(name)
        if entry is not None:
            return await self._build_from_registry(entry)

        # Strategy 2: URL heuristics
        candidates = await probe_heuristics(name, client=self._client, timeout=self._timeout)
        if candidates:
            best = candidates[0]
            return await self._build_from_heuristic(name, best)

        msg = f"{_NO_DISCOVERY_MSG} for '{name}'. Try providing the documentation URL directly."
        raise DiscoveryError(msg)

    async def discover_from_url(self, url: str) -> DiscoveryResult:
        """Discover documentation from a direct URL input.

        Probes the URL to extract a software name (from the domain/path)
        and attempts to enumerate versions.

        Args:
            url: The documentation base URL.

        Returns:
            A DiscoveryResult built from the provided URL.

        Raises:
            DiscoveryError: If the URL is unreachable or not documentation.
        """
        name = _extract_name_from_url(url)

        created_client = self._client is None
        if created_client:
            client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        else:
            client = self._client

        try:
            # Probe the URL
            response = await client.get(url, headers={"User-Agent": "DocForge/0.1"})
            response.raise_for_status()

            # Try to detect versions from the page
            versions = await _detect_versions_from_page(str(response.url), client)

            return DiscoveryResult(
                software=name,
                display_name=name.replace("-", " ").replace("_", " ").title(),
                base_url=str(response.url),
                versions=versions or ["latest"],
                latest_version=versions[0] if versions else "latest",
            )
        except httpx.HTTPError as e:
            msg = f"{_FAILED_ACCESS_MSG} at '{url}': {e}"
            raise DiscoveryError(msg) from e
        finally:
            if created_client:
                await client.aclose()

    def _get_registry(self) -> Registry:
        """Load the registry lazily if not provided."""
        if self._registry is None:
            self._registry = load_registry()
        return self._registry

    async def _build_from_registry(self, entry: RegistryEntry) -> DiscoveryResult:
        """Build a DiscoveryResult from a registry entry."""
        known_versions = entry.known_versions
        latest = entry.latest_version or (known_versions[0] if known_versions else "latest")
        strategy = entry.version_strategy or "explicit"

        if strategy != "explicit" and known_versions:
            try:
                versions = await detect_versions(
                    strategy,
                    base_url=entry.base_url,
                    version_pattern=entry.version_pattern,
                    known_versions=known_versions,
                    latest=latest,
                    sitemap_url=entry.sitemap_url,
                    client=self._client,
                )
                if versions:
                    known_versions = versions
                    latest = versions[0]
            except Exception:
                pass

        return DiscoveryResult(
            software=entry.name,
            display_name=entry.display_name,
            base_url=entry.base_url,
            versions=known_versions or ["latest"],
            latest_version=latest,
            sitemap_url=entry.sitemap_url,
            content_selectors=entry.content_selectors,
            url_filters=entry.url_filters,
        )

    async def _build_from_heuristic(
        self, name: str, candidate: HeuristicCandidate
    ) -> DiscoveryResult:
        """Build a DiscoveryResult from a heuristic candidate."""
        created_client = self._client is None
        if created_client:
            client = httpx.AsyncClient(timeout=self._timeout)
        else:
            client = self._client
        try:
            versions = await _detect_versions_from_page(candidate.url, client)
        finally:
            if created_client:
                await client.aclose()

        return DiscoveryResult(
            software=name,
            display_name=name.replace("-", " ").replace("_", " ").title(),
            base_url=candidate.url,
            versions=versions or ["latest"],
            latest_version=versions[0] if versions else "latest",
        )


async def _detect_versions_from_page(url: str, client: httpx.AsyncClient) -> list[str]:
    """Try to detect versions from a documentation page."""
    try:
        response = await client.get(url, headers={"User-Agent": "DocForge/0.1"})
        html = response.text
    except Exception:
        return []
    else:
        versions: list[str] = []
        pattern = re.compile(r'href="[^"]*?/(\d+(?:\.\d+)*)/[^"]*"')
        for match in pattern.finditer(html):
            ver = match.group(1)
            if ver not in versions:
                versions.append(ver)
        return versions


def _is_url(value: str) -> bool:
    """Check if the input looks like a URL."""
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _extract_name_from_url(url: str) -> str:
    """Extract a software name from a URL's domain or path."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # Remove common prefixes
    host = re.sub(r"^(docs?|www|api)\.", "", host)

    # Remove common suffixes
    host = re.sub(r"\.(com|org|io|dev|net)$", "", host)

    # For readthedocs: extract from subdomain
    if "readthedocs" in host:
        parts = host.split(".")
        if parts and parts[0] not in {"readthedocs", "docs"}:
            return parts[0]

    # Use the first meaningful segment
    name = host.split(".")[0]
    return re.sub(r"[^a-z0-9-]", "", name.lower()) or "unknown"
