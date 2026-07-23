"""Version enumeration strategies — detect available versions from various sources."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from docforge.discovery.sitemap import fetch_sitemap

# HTTP status code threshold for successful version probes
_HTTP_SUCCESS_THRESHOLD = 400


class VersionDetectionError(Exception):
    """Raised when versions cannot be detected."""


async def detect_versions(
    strategy: str,
    *,
    base_url: str,
    version_pattern: str | None = None,
    known_versions: list[str] | None = None,
    latest: str | None = None,
    sitemap_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Detect available versions using the specified strategy.

    Args:
        strategy: One of 'url_enumeration', 'sitemap', 'explicit', 'dropdown_scraping'.
        base_url: Root documentation URL.
        version_pattern: URL template with {version} placeholder (for url_enumeration).
        known_versions: Pre-defined version list (for explicit strategy).
        latest: Latest version string (for explicit strategy).
        sitemap_url: URL of the sitemap (for sitemap strategy).
        client: Optional httpx client for network requests.

    Returns:
        List of version strings, newest first.
    """
    if strategy == "explicit":
        return _explicit_strategy(known_versions or [])
    elif strategy == "url_enumeration":
        return await _url_enumeration_strategy(
            base_url=base_url,
            version_pattern=version_pattern,
            known_versions=known_versions or [],
            client=client,
        )
    elif strategy == "sitemap":
        return await _sitemap_strategy(
            sitemap_url=sitemap_url or f"{base_url.rstrip('/')}/sitemap.xml",
            base_url=base_url,
            client=client,
        )
    elif strategy == "dropdown_scraping":
        return await _dropdown_scraping_strategy(base_url=base_url, client=client)
    else:
        msg = f"Unknown version strategy: {strategy}"
        raise VersionDetectionError(msg)


def _explicit_strategy(known_versions: list[str]) -> list[str]:
    """Return the known versions list directly."""
    return list(known_versions)


async def _url_enumeration_strategy(
    *,
    base_url: str,
    version_pattern: str | None,
    known_versions: list[str],
    client: httpx.AsyncClient | None,
) -> list[str]:
    """Probe known version URLs to verify which versions actually exist.

    If a version_pattern is provided, uses it to build probe URLs.
    Otherwise, tries appending version strings to the base URL.
    """
    if not known_versions:
        return []

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
    assert client is not None

    try:
        verified: list[str] = []
        for version in known_versions:
            url = _build_version_url(base_url, version_pattern, version)
            try:
                resp = await client.head(url, headers={"User-Agent": "DocForge/0.1"})
                if resp.status_code < _HTTP_SUCCESS_THRESHOLD:
                    verified.append(version)
            except Exception:
                continue
        return verified
    finally:
        if own_client:
            await client.aclose()


async def _sitemap_strategy(
    *,
    sitemap_url: str,
    base_url: str,
    client: httpx.AsyncClient | None,
) -> list[str]:
    """Extract version segments from sitemap URLs."""
    try:
        sitemap_urls = await fetch_sitemap(sitemap_url, client=client)
    except Exception:
        return []

    versions = _extract_versions_from_urls([u.loc for u in sitemap_urls], base_url)
    return sorted(versions, key=_version_sort_key, reverse=True)


async def _dropdown_scraping_strategy(
    *,
    base_url: str,
    client: httpx.AsyncClient | None,
) -> list[str]:
    """Scrape a version selector element from the documentation page.

    This is a best-effort fallback that looks for common version dropdown
    patterns in HTML. Returns an empty list if no dropdown is found.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    assert client is not None

    try:
        resp = await client.get(base_url, headers={"User-Agent": "DocForge/0.1"})
        resp.raise_for_status()
        html = resp.text
        return _parse_version_dropdown(html)
    except Exception:
        return []
    finally:
        if own_client:
            await client.aclose()


def _build_version_url(base_url: str, pattern: str | None, version: str) -> str:
    """Build a URL for a specific version."""
    base = base_url.rstrip("/")
    if pattern:
        return pattern.format(version=version)
    return f"{base}/{version}/"


def _extract_versions_from_urls(urls: list[str], base_url: str) -> set[str]:
    """Extract version-like segments from URL paths."""
    versions: set[str] = set()
    base_path = urlparse(base_url).path.rstrip("/")

    for url in urls:
        path = urlparse(url).path
        # Remove the base path prefix
        if base_path and path.startswith(base_path):
            remainder = path[len(base_path) :].strip("/")
        else:
            remainder = path.strip("/")

        # Extract the first path segment that looks like a version
        parts = remainder.split("/")
        if parts and looks_like_version(parts[0]):
            versions.add(parts[0])

    return versions


def _parse_version_dropdown(html: str) -> list[str]:
    """Parse common version dropdown patterns from HTML."""
    versions: list[str] = []

    # Pattern 1: <select> with version options
    select_pattern = re.compile(
        r"<select[^>]*(?:version|release)[^>]*>(.*?)</select>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in select_pattern.finditer(html):
        option_pattern = re.compile(r'<option[^>]*value="([^"]*)"[^>]*>')
        for opt in option_pattern.finditer(match.group(1)):
            val = opt.group(1).strip("/")
            if val and looks_like_version(val):
                versions.append(val)

    # Pattern 2: Links with version-like paths
    if not versions:
        link_pattern = re.compile(r'href="[^"]*/(\d+(?:\.\d+)*)/[^"]*"')
        for match in link_pattern.finditer(html):
            ver = match.group(1)
            if looks_like_version(ver):
                versions.append(ver)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in versions:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique


def looks_like_version(s: str) -> bool:
    """Check if a string looks like a version number."""
    return bool(
        re.match(
            r"^v?\d+(?:\.\d+)*(?:[-.]?(?:alpha|beta|rc|pre|dev|stable))?$",
            s,
            re.IGNORECASE,
        )
    )


def _version_sort_key(version: str) -> tuple[int, ...]:
    """Create a sortable key from a version string."""
    clean_version = version.lstrip("vV")
    parts = re.split(r"[.\-]", clean_version)
    result: list[int] = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(0)
    return tuple(result)
