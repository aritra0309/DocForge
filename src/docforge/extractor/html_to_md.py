"""HTML to Markdown conversion with documentation-specific handling."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from lxml import html as lxml_html
from markdownify import markdownify as md

from docforge.extractor.code_blocks import detect_language_from_class


def _convert_images(element: lxml_html.HtmlElement, base_url: str) -> None:
    """Replace img tags with markdown-friendly alt/src references."""
    for img in element.cssselect("img"):
        alt = img.get("alt", "") or img.get("title", "") or "image"
        src = img.get("src", "")
        if src:
            src = urljoin(base_url, src)
        replacement = lxml_html.Element("p")
        replacement.text = f"![{alt}]({src})"
        parent = img.getparent()
        if parent is not None:
            parent.replace(img, replacement)


def _convert_definition_lists(element: lxml_html.HtmlElement) -> None:
    """Convert dl/dt/dd to bold term + indented definition paragraphs."""
    for dl in element.cssselect("dl"):
        replacement = lxml_html.Element("div")
        replacement.set("class", "docforge-dl")
        current: lxml_html.HtmlElement | None = None
        for child in dl:
            if child.tag == "dt":
                term = " ".join(child.text_content().split())
                current = lxml_html.Element("p")
                strong = lxml_html.Element("strong")
                strong.text = term
                current.append(strong)
                replacement.append(current)
            elif child.tag == "dd":
                definition = " ".join(child.text_content().split())
                if current is None:
                    current = lxml_html.Element("p")
                    replacement.append(current)
                    current.text = f": {definition}"
                else:
                    current[0].tail = f"\n: {definition}"
        parent = dl.getparent()
        if parent is not None:
            parent.replace(dl, replacement)


def _convert_callout_divs(element: lxml_html.HtmlElement) -> None:
    """Convert docforge-callout divs to blockquote-friendly format."""
    for callout in element.cssselect("div.docforge-callout"):
        callout_type = callout.get("data-callout-type", "Note")
        body = callout.text_content().strip()
        body = re.sub(rf"^\*{{0,2}}{re.escape(callout_type)}:\*{{0,2}}\s*", "", body)
        replacement = lxml_html.Element("blockquote")
        strong = lxml_html.Element("strong")
        strong.text = f"{callout_type}:"
        strong.tail = f" {body}" if body else ""
        replacement.append(strong)
        parent = callout.getparent()
        if parent is not None:
            parent.replace(callout, replacement)


def _resolve_links(element: lxml_html.HtmlElement, base_url: str) -> None:
    """Resolve relative href attributes to absolute URLs."""
    for anchor in element.cssselect("a[href]"):
        href = anchor.get("href", "")
        if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
            anchor.set("href", urljoin(base_url, href))


def html_to_markdown(content: lxml_html.HtmlElement, base_url: str) -> str:
    """Convert an HTML content subtree to clean Markdown."""
    fragment = lxml_html.Element("div")
    for child in content:
        fragment.append(child)

    _resolve_links(fragment, base_url)
    _convert_images(fragment, base_url)
    _convert_definition_lists(fragment)
    _convert_callout_divs(fragment)

    html_str = lxml_html.tostring(fragment, encoding="unicode", method="html")

    markdown = md(
        html_str,
        heading_style="ATX",
        bullets="-",
        code_language_callback=_code_language_callback,
        strip=["script", "style", "noscript"],
    )

    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = _ensure_absolute_links(markdown, base_url)
    return str(markdown).strip()


def _code_language_callback(el: lxml_html.HtmlElement) -> str | None:
    tag = getattr(el, "tag", None) or getattr(el, "name", None)

    class_attr = None
    if hasattr(el, "get"):
        raw_class = el.get("class")
        if isinstance(raw_class, list):
            class_attr = " ".join(raw_class)
        elif raw_class is not None:
            class_attr = str(raw_class)

    if tag == "code":
        return detect_language_from_class(class_attr)
    if tag == "pre":
        language = detect_language_from_class(class_attr)
        if language:
            return language
        code_el = None
        if hasattr(el, "find"):
            code_el = el.find("code")
            if code_el is None:
                code_el = el.find(".//code")
        if code_el is not None:
            code_class = code_el.get("class") if hasattr(code_el, "get") else None
            if isinstance(code_class, list):
                code_class = " ".join(code_class)
            return detect_language_from_class(str(code_class) if code_class is not None else None)
    return None


def _ensure_absolute_links(markdown: str, base_url: str) -> str:
    """Post-process markdown links to ensure relative URLs are absolute."""

    def replacer(match: re.Match[str]) -> str:
        text = match.group(1)
        url = match.group(2)
        if url.startswith(("/", "./", "../")):
            url = urljoin(base_url, url)
        return f"[{text}]({url})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replacer, markdown)


def extract_headings_from_markdown(markdown: str) -> list[str]:
    """Extract heading texts from ATX-style markdown headings."""
    headings: list[str] = []
    for line in markdown.split("\n"):
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            headings.append(match.group(1).strip())
    return headings
