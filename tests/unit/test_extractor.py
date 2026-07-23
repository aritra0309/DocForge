"""Unit tests for the content extraction pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from lxml import html as lxml_html

from docforge.core.models import FetchResult
from docforge.extractor.engine import ExtractionEngine
from docforge.extractor.tables import flatten_table, is_renderable_gfm_table

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _fetch_result(html: str, url: str = "https://example.com/docs/page.html") -> FetchResult:
    return FetchResult(url=url, status_code=200, html=html)


@pytest.mark.asyncio
async def test_extract_postgresql_page() -> None:
    """Extract clean Markdown from a PostgreSQL-style documentation page."""
    html = _load_fixture("postgresql_create_index.html")
    url = "https://www.postgresql.org/docs/17/sql/createindex.html"
    engine = ExtractionEngine(content_selectors={"main_content": "#docContent"})

    page = await engine.extract(_fetch_result(html, url))

    assert page.title == "CREATE INDEX"
    assert "CREATE INDEX constructs an index" in page.markdown
    assert "nav" not in page.markdown.lower() or "navigation" not in page.markdown.lower()
    assert "Copyright" not in page.markdown
    assert "PostgreSQL Global Development Group" not in page.markdown
    assert "```sql" in page.markdown
    assert "CREATE INDEX index_name" in page.markdown
    assert "| Parameter |" in page.markdown or "Parameter" in page.markdown
    assert "drop-index.html" in page.markdown or "DROP INDEX" in page.markdown
    assert page.url == url
    assert len(page.headings) >= 2
    assert any(block["language"] == "sql" for block in page.code_blocks)


@pytest.mark.asyncio
async def test_extract_resolves_relative_urls() -> None:
    """Relative links are resolved to absolute URLs."""
    html = _load_fixture("postgresql_create_index.html")
    url = "https://www.postgresql.org/docs/17/sql/createindex.html"
    engine = ExtractionEngine(content_selectors={"main_content": "#docContent"})

    page = await engine.extract(_fetch_result(html, url))

    assert "https://www.postgresql.org/docs/17/sql/drop-index.html" in page.markdown


@pytest.mark.asyncio
async def test_extract_sphinx_callouts() -> None:
    """Sphinx admonitions are normalised to blockquote-style callouts."""
    html = _load_fixture("sphinx_tutorial.html")
    engine = ExtractionEngine()

    page = await engine.extract(_fetch_result(html, "https://example.com/getting-started.html"))

    assert page.title == "Getting Started"
    assert "**Note:**" in page.markdown
    assert "**Warning:**" in page.markdown
    assert "Python 3.11+" in page.markdown
    assert "Sidebar navigation" not in page.markdown
    assert "Built with Sphinx" not in page.markdown
    assert "```python" in page.markdown


@pytest.mark.asyncio
async def test_extract_docusaurus_callouts() -> None:
    """Docusaurus admonitions are normalised consistently with Sphinx."""
    html = _load_fixture("docusaurus_guide.html")
    engine = ExtractionEngine()

    page = await engine.extract(_fetch_result(html, "https://example.com/installation.html"))

    assert page.title == "Installation"
    assert "**Tip:**" in page.markdown
    assert "**Danger:**" in page.markdown
    assert "virtual environment" in page.markdown
    assert "```bash" in page.markdown


@pytest.mark.asyncio
async def test_extract_metadata_and_breadcrumb() -> None:
    """Page metadata and breadcrumbs are extracted."""
    html = _load_fixture("postgresql_create_index.html")
    engine = ExtractionEngine(
        content_selectors={"main_content": "#docContent", "navigation": ".toc"}
    )

    page = await engine.extract(
        _fetch_result(html, "https://www.postgresql.org/docs/17/sql/createindex.html")
    )

    assert page.raw_metadata.get("description") == "CREATE INDEX command reference"
    assert page.raw_metadata.get("og:title") == "CREATE INDEX"
    assert "Home" in page.breadcrumb or "SQL Commands" in page.breadcrumb


def test_flatten_table_colspan() -> None:
    """Tables with colspan are flattened for GFM rendering."""
    html = """
    <table>
      <tr><th colspan="2">Header</th></tr>
      <tr><td>a</td><td>b</td></tr>
    </table>
    """
    table = lxml_html.fromstring(html)
    flattened = flatten_table(table)
    assert is_renderable_gfm_table(flattened)
    header_cells = flattened.cssselect("thead th")
    assert len(header_cells) == 2


@pytest.mark.asyncio
async def test_extraction_throughput() -> None:
    """Extraction throughput meets the 100 pages/sec target on cached HTML."""
    html = _load_fixture("postgresql_create_index.html")
    engine = ExtractionEngine(content_selectors={"main_content": "#docContent"})
    fetch = _fetch_result(html)

    start = time.perf_counter()
    for _ in range(200):
        await engine.extract(fetch)
    elapsed = time.perf_counter() - start

    pages_per_sec = 200 / elapsed
    assert pages_per_sec >= 100, f"Only {pages_per_sec:.1f} pages/sec (target: 100)"
