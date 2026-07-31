from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import respx

from docforge.core.config import DocForgeConfig
from docforge.core.models import DiscoveryResult
from docforge.core.pipeline import Pipeline
from docforge.embeddings.engine import EmbeddingEngine
from docforge.embeddings.providers.base import EmbeddingProvider
from docforge.storage.engine import StorageEngine
from tests.fixtures.site import (
    FIXTURE_SITE_BASE as BASE,
    FIXTURE_SITE_PAGE_COUNT,
    fixture_sitemap_xml,
    mock_fixture_site,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"


class FakeEmbeddingProvider(EmbeddingProvider):
    model_name = "test-e2e-model"
    dimension = 8
    max_tokens = 512

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.05 + (i * 0.01) for _ in range(self.dimension)] for i in range(len(texts))]


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_pipeline_full_e2e(tmp_path: Path) -> None:  # ruff: ignore[too-many-statements]
    """Full pipeline on the 20-page fixture site; assert chunk count and search."""
    mock_fixture_site(index_paths=("/docs/1.0",))

    config = DocForgeConfig(
        general={"data_dir": str(tmp_path / "data"), "log_level": "WARNING", "parallelism": 2},
        storage={"path": str(tmp_path / "vectordb"), "backend": "faiss"},
        embeddings={"cache_embeddings": False},
        crawler={"max_pages_per_version": 50, "rate_limit_rps": 100},
        chunker={"target_chunk_size": 512, "max_chunk_size": 1024},
    )

    mock_discovery = AsyncMock()
    mock_discovery.discover.return_value = DiscoveryResult(
        software="fixture",
        display_name="Fixture Test",
        base_url=f"{BASE}/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
    )

    provider = FakeEmbeddingProvider()
    pipeline = Pipeline(config=config)
    pipeline._discovery = mock_discovery
    pipeline._embedding_provider = provider
    pipeline._embedding_engine = EmbeddingEngine(provider=provider, batch_size=64)

    events_log: list[dict[str, Any]] = []

    def capture(event: Any) -> None:
        events_log.append({"type": event.type, "data": event.data})

    pipeline.events.on("pipeline.started", capture)
    pipeline.events.on("pipeline.completed", capture)
    pipeline.events.on("discovery.started", capture)
    pipeline.events.on("discovery.completed", capture)
    pipeline.events.on("crawl.started", capture)
    pipeline.events.on("crawl.completed", capture)
    pipeline.events.on("embedding.started", capture)
    pipeline.events.on("embedding.completed", capture)
    pipeline.events.on("storage.upserted", capture)
    pipeline.events.on("pipeline.version.completed", capture)

    result = await pipeline.run("fixture")

    assert result.status == "completed"
    assert len(result.versions) == 1
    vr = result.versions[0]

    # Index + 20 content pages
    assert vr.crawl.pages_processed >= FIXTURE_SITE_PAGE_COUNT
    assert vr.chunking.chunks_produced > 0
    assert vr.embedding.chunks_produced > 0
    assert vr.storage.chunks_produced > 0
    assert vr.total_duration_ms > 0

    event_types = [e["type"] for e in events_log]
    assert "pipeline.started" in event_types
    assert "pipeline.completed" in event_types
    assert "discovery.completed" in event_types
    assert "crawl.completed" in event_types
    assert "embedding.completed" in event_types
    assert "storage.upserted" in event_types
    assert "pipeline.version.completed" in event_types

    pipeline_start = next(e for e in events_log if e["type"] == "pipeline.started")
    assert pipeline_start["data"]["software"] == "fixture"

    pipeline_close = next(e for e in events_log if e["type"] == "pipeline.version.completed")
    assert pipeline_close["data"]["version"] == "1.0"

    # Search correctness against indexed fixture content
    storage = StorageEngine(config, software="fixture", version="1.0")
    await storage.initialize(dimension=provider.dimension, model_name=provider.model_name)
    query_vec = (await provider.embed_batch(["chunking overlapping semantic chunks"]))[0]
    hits = await storage.search(query_vec, k=5)
    assert len(hits) >= 1
    assert any("chunk" in h.content.lower() or "fixture" in h.content.lower() for h in hits)
    await storage.close()

    await pipeline.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_pipeline_e2e_no_chunks_when_no_pages(tmp_path: Path) -> None:
    respx.get(f"{BASE}/robots.txt").respond(status_code=404)
    respx.get(f"{BASE}/docs/1.0").respond(status_code=200, text="<html></html>")

    config = DocForgeConfig(
        general={"data_dir": str(tmp_path / "data_e2e_empty"), "log_level": "WARNING"},
        storage={"path": str(tmp_path / "vectordb_e2e_empty"), "backend": "faiss"},
        embeddings={"cache_embeddings": False},
        crawler={"max_pages_per_version": 10, "rate_limit_rps": 100},
    )

    mock_discovery = AsyncMock()
    mock_discovery.discover.return_value = DiscoveryResult(
        software="fixture",
        display_name="Fixture",
        base_url=f"{BASE}/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
    )

    provider = FakeEmbeddingProvider()
    pipeline = Pipeline(config=config)
    pipeline._discovery = mock_discovery
    pipeline._embedding_provider = provider
    pipeline._embedding_engine = EmbeddingEngine(provider=provider, batch_size=64)

    result = await pipeline.run("fixture")
    assert result.status == "completed"

    await pipeline.close()


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_pipeline_e2e_sitemap_lists_twenty_pages() -> None:
    """Fixture sitemap covers the full 20-page site used by e2e tests."""
    xml = fixture_sitemap_xml()
    assert xml.count("<url>") >= FIXTURE_SITE_PAGE_COUNT + 1
    assert f"{BASE}/docs/page20.html" in xml
