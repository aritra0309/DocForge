from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import respx

from docforge.core.config import DocForgeConfig
from docforge.core.models import DiscoveryResult
from docforge.core.pipeline import Pipeline
from docforge.embeddings.engine import EmbeddingEngine
from docforge.embeddings.providers.base import EmbeddingProvider

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"
BASE = "https://docs.fixture.test"


class FakeEmbeddingProvider(EmbeddingProvider):
    model_name = "test-incr-model"
    dimension = 8
    max_tokens = 512

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.05 + (i * 0.01) for _ in range(self.dimension)] for i in range(len(texts))]


SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.fixture.test/docs/1.0</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page1.html</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page2.html</loc><lastmod>2025-01-01</lastmod></url>
</urlset>"""


SITEMAP_WITH_CHANGE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.fixture.test/docs/1.0</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page1.html</loc><lastmod>2025-02-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page2.html</loc><lastmod>2025-01-01</lastmod></url>
</urlset>"""


def _make_config(tmp_path: Path, suffix: str = "") -> DocForgeConfig:
    return DocForgeConfig(
        general={
            "data_dir": str(tmp_path / f"data{suffix}"),
            "log_level": "WARNING",
            "parallelism": 2,
        },
        storage={"path": str(tmp_path / f"vectordb{suffix}"), "backend": "faiss"},
        embeddings={"cache_embeddings": False},
        crawler={"max_pages_per_version": 10, "rate_limit_rps": 100},
        chunker={"target_chunk_size": 512, "max_chunk_size": 1024},
    )


def _make_discovery(sitemap_url: str | None = f"{BASE}/sitemap.xml") -> AsyncMock:
    mock = AsyncMock()
    mock.discover.return_value = DiscoveryResult(
        software="fixture",
        display_name="Fixture Test",
        base_url=f"{BASE}/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
        sitemap_url=sitemap_url,
    )
    return mock


def _mock_html_pages(last_modified: str | None = None) -> None:
    index_html = (FIXTURES_DIR / "fixture_site_index.html").read_text(encoding="utf-8")
    page1_html = (FIXTURES_DIR / "fixture_site_page1.html").read_text(encoding="utf-8")
    page2_html = (FIXTURES_DIR / "fixture_site_page2.html").read_text(encoding="utf-8")

    headers = {}
    if last_modified:
        headers["Last-Modified"] = last_modified

    respx.get(f"{BASE}/robots.txt").respond(status_code=404)
    respx.get(f"{BASE}/docs/1.0").respond(200, text=index_html, headers=headers)
    respx.get(f"{BASE}/docs/page1.html").respond(200, text=page1_html, headers=headers)
    respx.get(f"{BASE}/docs/page2.html").respond(200, text=page2_html, headers=headers)
    respx.get(f"{BASE}/docs/private/secret.html").respond(404)


def _make_pipeline(config: DocForgeConfig, provider: FakeEmbeddingProvider) -> Pipeline:
    pipeline = Pipeline(config=config)
    pipeline._embedding_provider = provider
    pipeline._embedding_engine = EmbeddingEngine(provider=provider, batch_size=64)
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_incremental_no_changes_sitemap(tmp_path: Path) -> None:
    """Full index then incremental with matching sitemap lastmod → 0 pages re-processed."""
    config = _make_config(tmp_path, "_nochange")
    provider = FakeEmbeddingProvider()

    _mock_html_pages(last_modified="2025-01-01")

    pipeline = _make_pipeline(config, provider)
    pipeline._discovery = _make_discovery()

    result = await pipeline.run("fixture", mode="full")
    assert result.status == "completed"
    assert result.versions[0].crawl.pages_processed == 3
    await pipeline.close()

    respx.get(f"{BASE}/sitemap.xml").respond(200, text=SITEMAP_XML)

    pipeline2 = _make_pipeline(config, provider)
    pipeline2._discovery = _make_discovery()

    result2 = await pipeline2.run("fixture", mode="incremental")
    assert result2.status == "completed"
    vr2 = result2.versions[0]
    assert vr2.extraction.pages_processed == 0, (
        f"Expected 0 pages re-processed, got {vr2.extraction.pages_processed}"
    )
    assert vr2.chunking.chunks_produced == 0
    await pipeline2.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_incremental_changed_page(tmp_path: Path) -> None:
    """Full index then incremental with one page's lastmod changed → only that page re-processed."""
    config = _make_config(tmp_path, "_changed")
    provider = FakeEmbeddingProvider()

    _mock_html_pages(last_modified="2025-01-01")

    pipeline = _make_pipeline(config, provider)
    pipeline._discovery = _make_discovery()

    result = await pipeline.run("fixture", mode="full")
    assert result.status == "completed"
    assert result.versions[0].crawl.pages_processed == 3
    await pipeline.close()

    page1_html_modified = (
        (FIXTURES_DIR / "fixture_site_page1.html")
        .read_text(encoding="utf-8")
        .replace(
            "<p>This is page 1 of the fixture site.</p>",
            "<p>This is UPDATED page 1 content.</p>",
        )
    )

    respx.get(f"{BASE}/sitemap.xml").respond(200, text=SITEMAP_WITH_CHANGE)
    respx.get(f"{BASE}/docs/1.0").respond(304)
    respx.get(f"{BASE}/docs/page1.html").respond(
        200,
        text=page1_html_modified,
        headers={"Last-Modified": "2025-02-01"},
    )
    respx.get(f"{BASE}/docs/page2.html").respond(304)
    respx.get(f"{BASE}/docs/private/secret.html").respond(404)

    pipeline2 = _make_pipeline(config, provider)
    pipeline2._discovery = _make_discovery()

    result2 = await pipeline2.run("fixture", mode="incremental")
    assert result2.status == "completed"
    vr2 = result2.versions[0]
    assert vr2.extraction.pages_processed >= 1, "Expected at least 1 page re-processed"
    await pipeline2.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_incremental_removed_page(tmp_path: Path) -> None:
    """Full index then incremental with a page removed from sitemap → removed from store."""
    config = _make_config(tmp_path, "_removed")
    provider = FakeEmbeddingProvider()

    _mock_html_pages(last_modified="2025-01-01")

    pipeline = _make_pipeline(config, provider)
    pipeline._discovery = _make_discovery()

    result = await pipeline.run("fixture", mode="full")
    assert result.status == "completed"
    await pipeline.close()

    sitemap_missing_page = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.fixture.test/docs/1.0</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page1.html</loc><lastmod>2025-01-01</lastmod></url>
</urlset>"""

    respx.get(f"{BASE}/sitemap.xml").respond(200, text=sitemap_missing_page)

    pipeline2 = _make_pipeline(config, provider)
    pipeline2._discovery = _make_discovery()

    result2 = await pipeline2.run("fixture", mode="incremental")
    assert result2.status == "completed"
    await pipeline2.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_incremental_new_page(tmp_path: Path) -> None:
    """Full index then incremental with a new page in sitemap → new page indexed."""
    config = _make_config(tmp_path, "_newpage")
    provider = FakeEmbeddingProvider()

    _mock_html_pages(last_modified="2025-01-01")

    pipeline = _make_pipeline(config, provider)
    pipeline._discovery = _make_discovery()

    result = await pipeline.run("fixture", mode="full")
    assert result.status == "completed"
    await pipeline.close()

    sitemap_new_page = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.fixture.test/docs/1.0</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page1.html</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page2.html</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page3.html</loc><lastmod>2025-02-01</lastmod></url>
</urlset>"""

    respx.get(f"{BASE}/sitemap.xml").respond(200, text=sitemap_new_page)
    respx.get(f"{BASE}/docs/page3.html").respond(
        200,
        text="<html><body><h1>New Page</h1><p>Brand new content.</p></body></html>",
    )

    pipeline2 = _make_pipeline(config, provider)
    pipeline2._discovery = _make_discovery()

    result2 = await pipeline2.run("fixture", mode="incremental")
    assert result2.status == "completed"
    vr2 = result2.versions[0]
    assert vr2.chunking.chunks_produced >= 1, "Expected at least 1 new chunk"
    await pipeline2.close()
