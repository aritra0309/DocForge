"""Extraction orchestrator — converts FetchResult HTML into ExtractedPage."""

from __future__ import annotations

from lxml import html as lxml_html

from docforge.core.interfaces import ContentExtractor
from docforge.core.models import ExtractedPage, FetchResult
from docforge.extractor.callouts import normalise_callouts
from docforge.extractor.cleaners import (
    extract_breadcrumb,
    extract_main_content,
    extract_page_title,
    extract_raw_metadata,
)
from docforge.extractor.code_blocks import (
    extract_code_blocks_from_markdown,
    normalise_code_blocks,
    post_process_markdown_code_blocks,
)
from docforge.extractor.html_to_md import extract_headings_from_markdown, html_to_markdown
from docforge.extractor.tables import normalise_tables


class ExtractionEngine(ContentExtractor):
    """Full HTML extraction pipeline for documentation pages."""

    def __init__(self, content_selectors: dict[str, str] | None = None) -> None:
        self.content_selectors = content_selectors or {}

    async def extract(self, fetch_result: FetchResult) -> ExtractedPage:
        """Extract structured Markdown content from a fetched HTML page."""
        content_element, extraction_meta = extract_main_content(
            fetch_result.html,
            content_selectors=self.content_selectors,
        )

        working = lxml_html.Element("div")
        for child in content_element:
            working.append(child)

        normalise_callouts(working)
        normalise_tables(working)
        normalise_code_blocks(working)

        markdown = html_to_markdown(working, base_url=fetch_result.url)
        markdown = post_process_markdown_code_blocks(markdown)

        title = extract_page_title(content_element, fetch_result.html)
        headings = extract_headings_from_markdown(markdown)
        if not headings and title:
            headings = [title]

        code_blocks = extract_code_blocks_from_markdown(markdown)
        breadcrumb = extract_breadcrumb(fetch_result.html, self.content_selectors)
        raw_metadata = extract_raw_metadata(fetch_result.html)
        raw_metadata["extraction_method"] = extraction_meta.get("method", "unknown")

        return ExtractedPage(
            url=fetch_result.url,
            title=title,
            markdown=markdown,
            headings=headings,
            code_blocks=code_blocks,
            breadcrumb=breadcrumb,
            raw_metadata=raw_metadata,
        )
