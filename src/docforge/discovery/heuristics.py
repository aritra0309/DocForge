"""Fallback URL heuristics — probe common documentation URL patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

# Confidence threshold for accepting a heuristic candidate
_MIN_CONFIDENCE = 0.3

# HTTP status code threshold for treating a response as an error
_HTTP_ERROR_THRESHOLD = 400


class HeuristicError(Exception):
    """Raised when all heuristic probes fail."""


@dataclass
class HeuristicCandidate:
    """A candidate documentation URL with a confidence score."""

    url: str
    confidence: float
    has_sitemap: bool = False
    has_headings: bool = False
    has_code_blocks: bool = False


# Common documentation URL patterns to probe
_URL_PATTERNS: list[str] = [
    "https://docs.{name}.com/",
    "https://{name}.readthedocs.io/",
    "https://{name}.readthedocs.io/en/latest/",
    "https://www.{name}.org/docs/",
    "https://{name}.org/documentation/",
    "https://devdocs.io/{name}/",
    "https://docs.{name}.dev/",
]


async def probe_heuristics(
    name: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 10.0,
    max_candidates: int = 5,
) -> list[HeuristicCandidate]:
    """Probe common documentation URL patterns for a software name.

    Tries several URL patterns, fetches each candidate page, and scores
    them based on documentation-like features.

    Args:
        name: Software name to probe (e.g. 'postgresql').
        client: Optional httpx client. Creates one if None.
        timeout: Request timeout per probe.
        max_candidates: Maximum number of candidates to evaluate.

    Returns:
        List of candidates sorted by confidence (highest first).
    """
    urls = _build_candidate_urls(name)
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    assert client is not None

    try:
        candidates: list[HeuristicCandidate] = []
        for url in urls[:max_candidates]:
            candidate = await _probe_url(url, client)
            if candidate and candidate.confidence > _MIN_CONFIDENCE:
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
    finally:
        if own_client:
            await client.aclose()


def _build_candidate_urls(name: str) -> list[str]:
    """Build candidate URLs from common patterns."""
    urls: list[str] = []
    for pattern in _URL_PATTERNS:
        url = pattern.format(name=name.lower())
        if url not in urls:
            urls.append(url)
    return urls


async def _probe_url(
    url: str,
    client: httpx.AsyncClient,
) -> HeuristicCandidate | None:
    """Probe a single URL and return a scored candidate."""
    try:
        response = await client.get(url, headers={"User-Agent": "DocForge/0.1"})
        if response.status_code >= _HTTP_ERROR_THRESHOLD:
            return None

        html = response.text
        return _score_candidate(url, html)
    except httpx.HTTPError:
        return None


def _score_candidate(url: str, html: str) -> HeuristicCandidate:
    """Score a candidate URL based on documentation-like features."""
    confidence = 0.0

    # Check for sitemap link
    has_sitemap = bool(re.search(r'<link[^>]*rel="sitemap"[^>]*>', html, re.IGNORECASE))
    if has_sitemap:
        confidence += 0.15

    # Check for heading structure (H1, H2)
    has_headings = bool(re.search(r"<h[12][^>]*>", html, re.IGNORECASE))
    if has_headings:
        confidence += 0.25

    # Check for code blocks
    has_code = bool(re.search(r"<pre[^>]*><code", html, re.IGNORECASE))
    if has_code:
        confidence += 0.25

    # Check for navigation/sidebar (typical doc pattern)
    has_nav = bool(re.search(r"<nav|class=['\"](?:sidebar|toc|menu)['\"]", html, re.IGNORECASE))
    if has_nav:
        confidence += 0.15

    # Check for common doc frameworks
    frameworks = ["mkdocs", "sphinx", "docusaurus", "readthedocs", "gitbook"]
    for fw in frameworks:
        if fw in html.lower():
            confidence += 0.1
            break

    # Penalize non-doc content
    penalty_signals = ["pricing", "enterprise", "signup", "sign-up", "login"]
    for sig in penalty_signals:
        if sig in html.lower():
            confidence -= 0.2

    confidence = max(0.0, min(1.0, confidence))

    return HeuristicCandidate(
        url=url,
        confidence=confidence,
        has_sitemap=has_sitemap,
        has_headings=has_headings,
        has_code_blocks=has_code,
    )
