"""Breadcrumb extraction utilities — from HTML navigation or URL path segments."""

from __future__ import annotations

from urllib.parse import urlparse

from lxml import html as lxml_html


def extract_breadcrumb_from_html(
    html: str,
    selectors: dict[str, str] | None = None,
) -> list[str]:
    """Extract breadcrumb hierarchy from an HTML page's navigation structure.

    Checks, in order:
        1. A CSS selector provided via ``selectors.get("breadcrumb")``.
        2. Common breadcrumb class names: ``.breadcrumb``, ``.breadcrumbs``.
        3. Common semantic patterns: ``nav[aria-label="breadcrumb"]``,
           ``[role="navigation"][aria-label="breadcrumb"]``.

    Args:
        html: Raw HTML of the page.
        selectors: Optional dict of CSS selectors (key ``"breadcrumb"``).

    Returns:
        List of breadcrumb text items, in order from root to current page.
        Returns an empty list if no breadcrumb is found.
    """
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return []

    # Try explicit selector first
    if selectors and "breadcrumb" in selectors:
        elements = tree.cssselect(selectors["breadcrumb"])
        if elements:
            return _extract_breadcrumb_items(elements[0])

    # Try common class names
    for cls in (".breadcrumb", ".breadcrumbs"):
        elements = tree.cssselect(cls)
        if elements:
            return _extract_breadcrumb_items(elements[0])

    # Try semantic HTML patterns
    xpath_exprs = [
        '//nav[@aria-label="breadcrumb"]',
        '//nav[@aria-label="Breadcrumb"]',
        '//*[@role="navigation" and @aria-label="breadcrumb"]',
        '//ol[contains(@class, "breadcrumb")]',
        '//ul[contains(@class, "breadcrumb")]',
    ]
    for expr in xpath_exprs:
        navs = tree.xpath(expr)
        for nav in navs:
            items = _extract_breadcrumb_items(nav)
            if items:
                return items

    return []


def extract_breadcrumb_from_url(url: str) -> list[str]:
    """Extract breadcrumb-like hierarchy from URL path segments.

    For example:
        ``https://example.com/docs/17/tutorial/install``
        → ``["docs", "17", "tutorial", "install"]``

    Args:
        url: The page URL.

    Returns:
        List of path segments, with common file extensions stripped.
    """
    try:
        path = urlparse(url).path
    except Exception:
        return []

    segments = [seg for seg in path.split("/") if seg]
    result: list[str] = []
    for seg in segments:
        # Strip common file extensions
        if "." in seg:
            name = seg.rsplit(".", 1)[0]
            if name:
                result.append(name)
        else:
            result.append(seg)
    return result


def _extract_breadcrumb_items(element: lxml_html.HtmlElement) -> list[str]:
    """Extract text items from a breadcrumb navigation element.

    Handles both ``<li>``-based and ``<a>``-based breadcrumbs.
    """
    items: list[str] = []

    # Try <li> items first
    list_items = element.find_class("breadcrumb-item")
    if not list_items:
        list_items = element.cssselect("li")

    if list_items:
        for li in list_items:
            text = _get_element_text(li)
            if text:
                items.append(text)
        return items

    # Fall back to individual links
    links = element.cssselect("a")
    if links:
        for a in links:
            text = _get_element_text(a)
            if text:
                items.append(text)
        return items

    # Last resort: get all text nodes
    text = element.text_content().strip()
    if text:
        parts = [t.strip() for t in text.split("/") if t.strip()]
        return parts

    return items


def _get_element_text(element: lxml_html.HtmlElement) -> str:
    text = (element.text_content() or "").strip()
    return text


__all__ = [
    "extract_breadcrumb_from_html",
    "extract_breadcrumb_from_url",
]
