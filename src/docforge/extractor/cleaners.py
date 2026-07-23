"""HTML cleaning utilities to isolate main documentation content."""

from __future__ import annotations

import re
from typing import Any

from lxml import html as lxml_html
from readability import Document

REMOVE_SELECTORS: tuple[str, ...] = (
    "nav",
    "header",
    "footer",
    "aside",
    ".sidebar",
    ".side-bar",
    ".toc",
    ".table-of-contents",
    ".breadcrumb",
    ".breadcrumbs",
    ".cookie-banner",
    ".cookie-consent",
    "#cookie-banner",
    ".advertisement",
    ".ad",
    ".ads",
    "script",
    "style",
    "noscript",
    "iframe",
)

SEMANTIC_SELECTORS: tuple[str, ...] = (
    "main",
    "article",
    '[role="main"]',
)

HEURISTIC_SELECTORS: tuple[str, ...] = (
    "#content",
    "#docContent",
    ".doc-content",
    ".documentation",
    ".markdown-body",
    ".document",
    ".rst-content",
    ".bd-content",
    ".md-content",
)

STRIP_TAGS: tuple[str, ...] = ("script", "style", "noscript", "iframe", "svg")


def _strip_elements(root: lxml_html.HtmlElement, selectors: tuple[str, ...]) -> None:
    for selector in selectors:
        for element in root.cssselect(selector):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def _element_text_density(element: lxml_html.HtmlElement) -> float:
    text = element.text_content()
    text_len = len(text.strip())
    if text_len == 0:
        return 0.0
    tag_count = max(1, len(list(element.iter("*"))))
    link_count = max(1, len(element.cssselect("a")))
    return text_len / (tag_count + link_count * 2)


def _pick_best_heuristic(root: lxml_html.HtmlElement) -> lxml_html.HtmlElement | None:
    candidates: list[tuple[float, lxml_html.HtmlElement]] = []
    for selector in HEURISTIC_SELECTORS:
        for element in root.cssselect(selector):
            density = _element_text_density(element)
            if density > 0:
                candidates.append((density, element))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _readability_fallback(html_text: str) -> lxml_html.HtmlElement | None:
    try:
        doc = Document(html_text)
        summary_html = doc.summary()
        if not summary_html.strip():
            return None
        fragment = lxml_html.fragment_fromstring(summary_html, create_parent="div")
        return fragment
    except Exception:
        return None


def extract_main_content(
    html_text: str,
    content_selectors: dict[str, str] | None = None,
) -> tuple[lxml_html.HtmlElement, dict[str, Any]]:
    """Extract the main documentation content region from a full HTML page.

    Returns:
        Tuple of (content element, extraction metadata).
    """
    selectors = content_selectors or {}
    metadata: dict[str, Any] = {"method": "unknown"}

    try:
        doc = lxml_html.fromstring(html_text)
    except Exception:
        doc = lxml_html.fragment_fromstring(html_text, create_parent="div")

    if not isinstance(doc, lxml_html.HtmlElement):
        doc = lxml_html.fragment_fromstring(html_text, create_parent="div")

    main_selector = selectors.get("main_content")
    if main_selector:
        matches = doc.cssselect(main_selector)
        if matches:
            content = matches[0]
            _strip_elements(content, REMOVE_SELECTORS)
            metadata["method"] = f"registry:{main_selector}"
            return content, metadata

    for selector in SEMANTIC_SELECTORS:
        matches = doc.cssselect(selector)
        if matches:
            content = matches[0]
            _strip_elements(content, REMOVE_SELECTORS)
            metadata["method"] = f"semantic:{selector}"
            return content, metadata

    heuristic = _pick_best_heuristic(doc)
    if heuristic is not None:
        _strip_elements(heuristic, REMOVE_SELECTORS)
        metadata["method"] = "heuristic"
        return heuristic, metadata

    readability = _readability_fallback(html_text)
    if readability is not None:
        _strip_elements(readability, REMOVE_SELECTORS)
        metadata["method"] = "readability"
        return readability, metadata

    body = doc.find(".//body")
    content = body if body is not None else doc
    _strip_elements(content, REMOVE_SELECTORS)
    metadata["method"] = "body_fallback"
    return content, metadata


def extract_page_title(doc: lxml_html.HtmlElement, html_text: str) -> str:
    """Extract the page title from h1 or <title> tag.

    Prefers the h1 text inside the content element. Falls back to the
    <title> tag, stripping common " - Site Name" / " | Site Name" suffixes.
    """
    h1_elements = doc.cssselect("h1")
    for h1 in h1_elements:
        text = h1.text_content().strip()
        if text:
            return re.sub(r"\s+", " ", text)

    try:
        full_doc = lxml_html.fromstring(html_text)
        title_el = full_doc.find(".//title")
        if title_el is not None:
            title_text = title_el.text_content().strip()
            if title_text:
                # Strip common "Page Title - Site Name" patterns
                for sep in (" — ", " | ", " - "):
                    if sep in title_text:
                        title_text = title_text.split(sep)[0].strip()
                        break
                return re.sub(r"\s+", " ", title_text)
    except Exception:
        pass

    return "Untitled"


def extract_breadcrumb(
    html_text: str,
    content_selectors: dict[str, str] | None = None,
) -> list[str]:
    """Extract navigation breadcrumb from common doc site patterns."""
    selectors = content_selectors or {}
    crumbs: list[str] = []

    try:
        doc = lxml_html.fromstring(html_text)
    except Exception:
        return crumbs

    nav_selector = selectors.get("navigation")
    if nav_selector:
        for nav in doc.cssselect(nav_selector):
            for link in nav.cssselect("a"):
                text = link.text_content().strip()
                if text and text not in crumbs:
                    crumbs.append(text)

    for selector in (".breadcrumb", ".breadcrumbs", '[aria-label="breadcrumb"]'):
        for container in doc.cssselect(selector):
            for link in container.cssselect("a, span"):
                text = link.text_content().strip()
                if text and text not in crumbs:
                    crumbs.append(text)

    return crumbs


def extract_raw_metadata(html_text: str) -> dict[str, Any]:
    """Extract OpenGraph and standard meta tags from HTML."""
    metadata: dict[str, Any] = {}
    try:
        doc = lxml_html.fromstring(html_text)
    except Exception:
        return metadata

    for meta in doc.cssselect("meta"):
        name = meta.get("name") or meta.get("property")
        content = meta.get("content")
        if name and content:
            metadata[name.lower()] = content

    canonical = doc.cssselect('link[rel="canonical"]')
    if canonical:
        href = canonical[0].get("href")
        if href:
            metadata["canonical"] = href

    return metadata
