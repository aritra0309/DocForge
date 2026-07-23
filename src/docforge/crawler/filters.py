"""URL filter chain and normalisation utilities for documentation crawling."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from lxml import html as lxml_html

NON_DOC_PATH_PATTERNS: tuple[str, ...] = (
    "/blog/",
    "/pricing/",
    "/community/",
    "/login/",
    "/signup/",
    "/careers/",
    "/about/",
    "/terms/",
    "/privacy/",
    "/contact/",
    "/download/",
    "/downloads/",
    "/forum/",
    "/shop/",
    "/cart/",
    "/press/",
    "/news/",
    "/events/",
    "/auth/",
    "/account/",
    "/billing/",
)

NON_DOC_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".json",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".exe",
    ".dmg",
    ".iso",
    ".xml",
    ".mp4",
    ".mp3",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".webp",
    ".avif",
)


def normalize_url(url: str) -> str:
    """Normalise a URL by lowercasing netloc, sorting query params, and stripping fragments.

    Args:
        url: The input URL string.

    Returns:
        The canonical normalised URL string.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"

    # Strip trailing slash if path is longer than 1 char
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    # Sort query params
    if parsed.query:
        query_params = sorted(parse_qsl(parsed.query, keep_blank_values=True))
        query = urlencode(query_params)
    else:
        query = ""

    # Omit fragment
    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a path glob pattern (supporting **) into a compiled regex object."""
    # Escape special regex chars except * and ?
    escaped = re.escape(pattern)

    # Convert escaped wildcards:
    # \*\* -> .*
    # \* -> [^/]*
    # \? -> [^/]
    regex_str = (
        escaped.replace(r"\*\*", ".*")
        .replace(r"\*", r"[^/]*")
        .replace(r"\?", r"[^/]")
    )

    if not regex_str.startswith("^"):
        regex_str = f"^{regex_str}"
    if not regex_str.endswith("$"):
        regex_str = f"{regex_str}$"

    return re.compile(regex_str)


class URLFilter:
    """URL filter enforcing domain boundaries, path include/exclude rules, and heuristics."""

    def __init__(
        self,
        base_url: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self.base_url = normalize_url(base_url)
        parsed_base = urlparse(self.base_url)
        self.target_netloc = parsed_base.netloc.lower()
        self.target_scheme = parsed_base.scheme.lower()

        self.include_regexes = [glob_to_regex(p) for p in (include_patterns or [])]
        self.exclude_regexes = [glob_to_regex(p) for p in (exclude_patterns or [])]

    def is_allowed(self, url: str) -> bool:
        """Check if a URL is allowed under the current filter rules.

        Args:
            url: URL string to validate.

        Returns:
            True if allowed, False otherwise.
        """
        normalized = normalize_url(url)
        parsed = urlparse(normalized)

        # Domain filter
        if parsed.netloc.lower() != self.target_netloc:
            return False

        path = parsed.path or "/"

        # Non-doc heuristics: path segments
        path_lower = path.lower()
        if any(pattern in path_lower for pattern in NON_DOC_PATH_PATTERNS):
            return False

        # Non-doc heuristics: file extensions
        if any(path_lower.endswith(ext) for ext in NON_DOC_EXTENSIONS):
            return False

        # Registry / config exclude patterns
        if any(regex.search(path) for regex in self.exclude_regexes):
            return False

        # Registry / config include patterns
        if self.include_regexes:
            if not any(regex.search(path) for regex in self.include_regexes):
                return False

        return True

    def extract_links(self, html_text: str, current_url: str) -> list[str]:
        """Extract, normalise, filter, and deduplicate all links from an HTML document.

        Args:
            html_text: Raw HTML string.
            current_url: Absolute URL of the document (used to resolve relative links).

        Returns:
            List of allowed, normalised absolute URLs.
        """
        if not html_text.strip():
            return []

        raw_links: set[str] = set()

        try:
            doc = lxml_html.fromstring(html_text)
            doc.make_links_absolute(current_url)
            for _, attr, link, _ in doc.iterlinks():
                if attr == "href" and link:
                    raw_links.add(link)
        except Exception:
            # Regex fallback if HTML is severely malformed
            for match in re.finditer(r'href=["\']([^"\']+)["\']', html_text, re.IGNORECASE):
                href = match.group(1).strip()
                if href and not href.startswith(("javascript:", "mailto:", "tel:")):
                    raw_links.add(urljoin(current_url, href))

        allowed: set[str] = set()
        for link in raw_links:
            norm = normalize_url(link)
            if self.is_allowed(norm):
                allowed.add(norm)

        return sorted(allowed)
