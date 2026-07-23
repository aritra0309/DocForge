"""XML sitemap parser — extracts URLs, lastmod, and priority from sitemaps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree

import httpx


class SitemapError(Exception):
    """Raised when a sitemap cannot be fetched or parsed."""


@dataclass(frozen=True)
class SitemapUrl:
    """A single URL entry from a sitemap."""

    loc: str
    lastmod: str | None = None
    priority: float | None = None


def parse_sitemap(xml_content: str) -> list[SitemapUrl]:
    """Parse a flat sitemap XML and return all listed URLs.

    Args:
        xml_content: Raw XML string of the sitemap.

    Returns:
        List of SitemapUrl entries.

    Raises:
        SitemapError: If the XML is malformed or not a valid sitemap.
    """
    try:
        root = ElementTree.fromstring(xml_content.strip())
    except ElementTree.ParseError as e:
        msg = f"Invalid XML: {e}"
        raise SitemapError(msg) from e

    ns = _detect_namespace(root)
    urls = _extract_urls(root, ns)

    if not urls:
        msg = "No <url> entries found in sitemap"
        raise SitemapError(msg)

    return urls


def parse_sitemap_index(xml_content: str) -> list[str]:
    """Parse a sitemap index and return child sitemap URLs.

    Args:
        xml_content: Raw XML string of the sitemap index.

    Returns:
        List of child sitemap URLs.

    Raises:
        SitemapError: If the XML is malformed or contains no sitemap entries.
    """
    try:
        root = ElementTree.fromstring(xml_content.strip())
    except ElementTree.ParseError as e:
        msg = f"Invalid XML: {e}"
        raise SitemapError(msg) from e

    ns = _detect_namespace(root)
    sitemap_locs: list[str] = []

    for sitemap in root.iter(_tag("sitemap", ns)):
        loc_el = sitemap.find(_tag("loc", ns))
        if loc_el is not None and loc_el.text:
            sitemap_locs.append(loc_el.text.strip())

    if not sitemap_locs:
        msg = "No <sitemap> entries found in sitemap index"
        raise SitemapError(msg)

    return sitemap_locs


async def fetch_sitemap(
    url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> list[SitemapUrl]:
    """Fetch and parse a sitemap, handling both flat and index formats.

    If the sitemap is an index, recursively fetches all child sitemaps.

    Args:
        url: URL of the sitemap to fetch.
        client: Optional pre-configured httpx client. Creates one if None.
        timeout: Request timeout in seconds.

    Returns:
        Combined list of all SitemapUrl entries from all sitemaps.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    assert client is not None

    try:
        return await _fetch_and_parse(url, client, timeout)
    finally:
        if own_client:
            await client.aclose()


async def _fetch_and_parse(
    url: str,
    client: httpx.AsyncClient,
    timeout: float,
) -> list[SitemapUrl]:
    """Fetch a URL and determine if it's a flat sitemap or index."""
    response = await client.get(url, headers={"User-Agent": "DocForge/0.1"})
    response.raise_for_status()
    content = response.text

    root = _safe_parse_root(content.strip())
    if root is None:
        return []

    ns = _detect_namespace(root)

    # Check if this is a sitemap index
    if root.find(_tag("sitemap", ns)) is not None:
        child_urls = parse_sitemap_index(content)
        all_entries: list[SitemapUrl] = []
        for child_url in child_urls:
            try:
                entries = await _fetch_and_parse(child_url, client, timeout)
                all_entries.extend(entries)
            except Exception:
                continue
        return all_entries

    # Flat sitemap
    return parse_sitemap(content)


def _safe_parse_root(xml_content: str) -> ElementTree.Element | None:
    """Safely parse XML, returning None on failure."""
    try:
        return ElementTree.fromstring(xml_content.strip())
    except ElementTree.ParseError:
        return None


def _detect_namespace(root: ElementTree.Element) -> str | None:
    """Detect the XML namespace from the root element."""
    tag = root.tag
    match = re.match(r"\{(.+?)\}", tag)
    return match.group(1) if match else None


def _tag(local_name: str, ns: str | None) -> str:
    """Build a namespaced tag string."""
    if ns:
        return f"{{{ns}}}{local_name}"
    return local_name


def _extract_urls(root: ElementTree.Element, ns: str | None) -> list[SitemapUrl]:
    """Extract all <url> entries from a sitemap root."""
    urls: list[SitemapUrl] = []

    for url_el in root.iter(_tag("url", ns)):
        loc_el = url_el.find(_tag("loc", ns))
        if loc_el is None or not loc_el.text:
            continue

        loc = loc_el.text.strip()
        lastmod_el = url_el.find(_tag("lastmod", ns))
        lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None

        priority_el = url_el.find(_tag("priority", ns))
        priority = None
        if priority_el is not None and priority_el.text:
            try:
                priority = float(priority_el.text)
            except ValueError:
                pass

        urls.append(SitemapUrl(loc=loc, lastmod=lastmod, priority=priority))

    return urls
