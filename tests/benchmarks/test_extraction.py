"""Extraction benchmark tests."""

from __future__ import annotations

import pytest
from pathlib import Path

from docforge.core.models import FetchResult
from docforge.extractor.engine import ExtractionEngine

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _fetch_result(html: str, url: str = "https://example.com/docs/page.html") -> FetchResult:
    return FetchResult(url=url, status_code=200, html=html)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_extraction_benchmark_pages_per_sec() -> None:
    """Benchmark extraction throughput on cached HTML."""
    html = _load_fixture("postgresql_create_index.html")
    engine = ExtractionEngine(content_selectors={"main_content": "#docContent"})
    fetch = _fetch_result(html)

    # Warm up
    for _ in range(10):
        await engine.extract(fetch)

    # Benchmark
    iterations = 200
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        await engine.extract(fetch)
    elapsed = time.perf_counter() - start

    pages_per_sec = iterations / elapsed
    assert pages_per_sec >= 100, f"Extraction throughput {pages_per_sec:.1f} pages/sec below target 100"

    # Store result for summary
    from tests.benchmarks import benchmark
    with benchmark("extraction_pages_per_sec", iterations):
        pass  # Already timed above


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_extraction_benchmark_varied_pages() -> None:
    """Benchmark extraction on varied page types."""
    pages = [
        ("postgresql_create_index.html", "https://www.postgresql.org/docs/17/sql/createindex.html"),
        ("sphinx_tutorial.html", "https://example.com/getting-started.html"),
        ("docusaurus_guide.html", "https://example.com/installation.html"),
        ("fixture_site_page1.html", "https://docs.fixture.test/docs/page1.html"),
        ("fixture_site_page2.html", "https://docs.fixture.test/docs/page2.html"),
    ]

    engines = [
        ExtractionEngine(content_selectors={"main_content": "#docContent"}),
        ExtractionEngine(),
        ExtractionEngine(),
        ExtractionEngine(),
        ExtractionEngine(),
    ]

    fetches = [_fetch_result(_load_fixture(fname), url) for fname, url in pages]

    # Warm up
    for engine, fetch in zip(engines, fetches):
        for _ in range(5):
            await engine.extract(fetch)

    # Benchmark
    iterations = 100
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        for engine, fetch in zip(engines, fetches):
            await engine.extract(fetch)
    elapsed = time.perf_counter() - start

    total_pages = iterations * len(pages)
    pages_per_sec = total_pages / elapsed
    assert pages_per_sec >= 100, f"Extraction throughput {pages_per_sec:.1f} pages/sec below target 100"

    from tests.benchmarks import benchmark
    with benchmark("extraction_varied_pages_per_sec", total_pages):
        pass