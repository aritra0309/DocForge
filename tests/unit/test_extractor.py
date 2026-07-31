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
MARKDOWN_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "markdown"

# Registry-aligned selectors and canonical URLs for golden comparisons
_GOLDEN_CASES: list[tuple[str, str, dict[str, str] | None]] = [
    ("postgresql_create_index.html", "https://www.postgresql.org/docs/17/sql/createindex.html", {"main_content": "#docContent"}),
    ("postgresql_indexes.html", "https://www.postgresql.org/docs/17/indexes.html", {"main_content": "#docContent"}),
    ("postgresql_tutorial.html", "https://www.postgresql.org/docs/17/tutorial.html", {"main_content": "#docContent"}),
    ("fastapi_getting_started.html", "https://fastapi.tiangolo.com/tutorial/first-steps/", None),
    ("fastapi_routing.html", "https://fastapi.tiangolo.com/tutorial/path-params/", None),
    ("fastapi_request_body.html", "https://fastapi.tiangolo.com/tutorial/body/", None),
    ("fastapi_dependencies.html", "https://fastapi.tiangolo.com/tutorial/dependencies/", None),
    ("react_hooks.html", "https://react.dev/reference/react/hooks", None),
    ("react_usestate.html", "https://react.dev/reference/react/useState", None),
    ("react_components.html", "https://react.dev/learn/your-first-component", None),
    ("redis_getting_started.html", "https://redis.io/docs/latest/get-started/", None),
    ("redis_strings.html", "https://redis.io/docs/latest/develop/data-types/strings/", None),
    ("redis_data_types.html", "https://redis.io/docs/latest/develop/data-types/", None),
    ("kubernetes_pods.html", "https://kubernetes.io/docs/concepts/workloads/pods/", None),
    ("kubernetes_deployments.html", "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/", None),
    ("mongodb_find.html", "https://www.mongodb.com/docs/manual/tutorial/query-documents/", None),
    ("mongodb_aggregation.html", "https://www.mongodb.com/docs/manual/aggregation/", None),
    ("mysql_select.html", "https://dev.mysql.com/doc/refman/8.4/en/select.html", None),
    ("mysql_data_types.html", "https://dev.mysql.com/doc/refman/8.4/en/data-types.html", None),
    ("sphinx_tutorial.html", "https://example.com/getting-started.html", None),
    ("docusaurus_guide.html", "https://example.com/installation.html", None),
]


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _load_markdown_golden(name: str) -> str:
    return (MARKDOWN_DIR / name).read_text(encoding="utf-8")


def _fetch_result(html: str, url: str = "https://example.com/docs/page.html") -> FetchResult:
    return FetchResult(url=url, status_code=200, html=html)


def _normalize_md(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"


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


@pytest.mark.asyncio
@pytest.mark.parametrize(("html_name", "url", "selectors"), _GOLDEN_CASES)
async def test_extract_matches_markdown_golden(
    html_name: str,
    url: str,
    selectors: dict[str, str] | None,
) -> None:
    """HTML → Markdown fidelity against golden fixture files."""
    html = _load_fixture(html_name)
    golden_name = html_name.replace(".html", ".md")
    expected = _load_markdown_golden(golden_name)
    engine = ExtractionEngine(content_selectors=selectors) if selectors else ExtractionEngine()

    page = await engine.extract(_fetch_result(html, url))

    assert _normalize_md(page.markdown) == _normalize_md(expected)
    assert page.title
    assert len(page.markdown) > 50


def test_markdown_goldens_cover_html_fixtures() -> None:
    """Every non-fixture-site HTML page has a corresponding markdown golden."""
    html_files = [
        p.name
        for p in FIXTURES_DIR.glob("*.html")
        if not p.name.startswith("fixture_site")
    ]
    missing = [name for name in html_files if not (MARKDOWN_DIR / name.replace(".html", ".md")).exists()]
    assert not missing, f"Missing markdown goldens for: {missing}"
    assert len(html_files) >= 20
