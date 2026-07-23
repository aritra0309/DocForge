"""Normalise documentation callout/admonition elements to consistent blockquote format."""

from __future__ import annotations

import re

from lxml import html as lxml_html

CALLOUT_SELECTORS: tuple[tuple[str, str], ...] = (
    (".admonition.note, .admonition-note, div.note", "Note"),
    (".admonition.warning, .admonition-warning, div.warning", "Warning"),
    (".admonition.tip, .admonition-tip, div.tip", "Tip"),
    (".admonition.important, .admonition-important", "Important"),
    (".admonition.caution, .admonition-caution", "Caution"),
    (".admonition.danger, .admonition-danger, div.danger", "Danger"),
    (".admonition.info, .admonition-info, div.info", "Info"),
    (".theme-admonition-note", "Note"),
    (".theme-admonition-warning", "Warning"),
    (".theme-admonition-tip", "Tip"),
    (".theme-admonition-info", "Info"),
    (".theme-admonition-danger", "Danger"),
    (".theme-admonition-caution", "Caution"),
    (".md-typeset .admonition", "Note"),
    (".rst-content .admonition", "Note"),
)

TITLE_SELECTORS: tuple[str, ...] = (
    ".admonition-title",
    ".admonition-heading",
    ".theme-admonition-heading",
    "p.admonition-title",
    "p.first.admonition-title",
)


def _detect_callout_type(element: lxml_html.HtmlElement) -> str:
    classes = " ".join(element.get("class", "").split()).lower()
    if "warning" in classes:
        return "Warning"
    if "tip" in classes:
        return "Tip"
    if "important" in classes:
        return "Important"
    if "caution" in classes:
        return "Caution"
    if "danger" in classes:
        return "Danger"
    if "info" in classes:
        return "Info"
    return "Note"


def _extract_callout_body(element: lxml_html.HtmlElement) -> str:
    for title_sel in TITLE_SELECTORS:
        for title_el in element.cssselect(title_sel):
            parent = title_el.getparent()
            if parent is not None:
                parent.remove(title_el)

    text = element.text_content()
    return re.sub(r"\s+", " ", text).strip()


def normalise_callouts(root: lxml_html.HtmlElement) -> None:
    """Replace callout/admonition elements with markdown-friendly blockquote divs."""
    seen: set[int] = set()

    for selector, default_label in CALLOUT_SELECTORS:
        for element in root.cssselect(selector):
            element_id = id(element)
            if element_id in seen:
                continue
            seen.add(element_id)

            label = default_label
            for title_sel in TITLE_SELECTORS:
                titles = element.cssselect(title_sel)
                if titles:
                    label = titles[0].text_content().strip() or default_label
                    break
            if label == "Note" and element.get("class"):
                label = _detect_callout_type(element)

            body = _extract_callout_body(element)
            replacement = lxml_html.Element("div")
            replacement.set("class", "docforge-callout")
            replacement.set("data-callout-type", label)
            replacement.text = f"**{label}:** {body}"

            parent = element.getparent()
            if parent is not None:
                parent.replace(element, replacement)
