from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docforge.core import events
from docforge.core.config import DocForgeConfig
from docforge.core.models import (
    Chunk,
    ChunkMetadata,
    ClassifiedPage,
    DiscoveryResult,
    ExtractedPage,
    FetchResult,
    PageType,
)
from docforge.core.pipeline import Pipeline


def _make_fetch_result(url: str = "https://example.com/docs/1.0/") -> FetchResult:
    return FetchResult(
        url=url,
        status_code=200,
        html="<html><body><h1>Test</h1><p>Content</p></body></html>",
    )


def _make_extracted_page(url: str = "https://example.com/docs/1.0/") -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title="Test Page",
        markdown="# Test Page\n\nContent here.",
        headings=["Test Page"],
    )


def _make_classified_page(url: str = "https://example.com/docs/1.0/") -> ClassifiedPage:
    return ClassifiedPage(
        url=url,
        title="Test Page",
        markdown="# Test Page\n\nContent here.",
        headings=["Test Page"],
        page_type=PageType.GUIDE,
        confidence=0.95,
    )


def _make_chunk() -> Chunk:
    meta = ChunkMetadata(
        chunk_id="test_chunk_id",
        parent_page_id="test_page_id",
        software="test",
        version="1.0",
        url="https://example.com/docs/1.0/",
        title="Test Page",
        page_type=PageType.GUIDE,
        section_heading="Test",
        chunk_index=0,
        total_chunks=1,
        has_code=False,
        content_hash="abc123",
        crawl_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        embedding_model="test-model",
        embedding_dimension=4,
        docforge_version="0.1.0-dev",
    )
    return Chunk(content="# Test\n\nContent", metadata=meta)


def _make_discovery_result() -> DiscoveryResult:
    return DiscoveryResult(
        software="test",
        display_name="Test",
        base_url="https://example.com/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
    )


def _config(tmp_path: Any) -> DocForgeConfig:
    return DocForgeConfig(
        general={"data_dir": str(tmp_path / "data"), "log_level": "WARNING"},
        storage={"path": str(tmp_path / "vectordb"), "backend": "faiss"},
        embeddings={"cache_embeddings": False},
        crawler={"max_pages_per_version": 10, "rate_limit_rps": 100},
    )


# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------


class TestEventBus:
    @pytest.mark.asyncio
    async def test_on_and_emit_sync_handler(self) -> None:
        bus = events.EventBus()
        received: list[events.PipelineEvent] = []

        def handler(event: events.PipelineEvent) -> None:
            received.append(event)

        bus.on("test.event", handler)
        await bus.emit("test.event", key="value")
        assert len(received) == 1
        assert received[0].type == "test.event"
        assert received[0].data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_on_and_emit_async_handler(self) -> None:
        bus = events.EventBus()
        received: list[events.PipelineEvent] = []

        def handler(event: events.PipelineEvent) -> None:
            received.append(event)

        bus.on("test.event", handler)
        await bus.emit("test.event")
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_off_removes_handler(self) -> None:
        bus = events.EventBus()
        calls: list[str] = []

        def h1(event: events.PipelineEvent) -> None:
            calls.append("h1")

        def h2(event: events.PipelineEvent) -> None:
            calls.append("h2")

        bus.on("ev", h1)
        bus.on("ev", h2)
        bus.off("ev", h1)
        await bus.emit("ev")
        assert calls == ["h2"]

    @pytest.mark.asyncio
    async def test_off_removes_all_handlers(self) -> None:
        bus = events.EventBus()
        calls: list[str] = []

        def h1(event: events.PipelineEvent) -> None:
            calls.append("h1")

        bus.on("ev", h1)
        bus.off("ev")
        await bus.emit("ev")
        assert calls == []

    @pytest.mark.asyncio
    async def test_emit_ignores_unregistered_events(self) -> None:
        bus = events.EventBus()
        await bus.emit("nonexistent")

    @pytest.mark.asyncio
    async def test_inherits_timestamp(self) -> None:
        bus = events.EventBus()
        captured: list[events.PipelineEvent] = []

        def handler(event: events.PipelineEvent) -> None:
            captured.append(event)

        bus.on("ts", handler)
        await bus.emit("ts")
        assert captured[0].timestamp is not None


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.mark.asyncio
    @patch("docforge.core.pipeline.ExtractionEngine")
    @patch("docforge.core.pipeline.ClassificationEngine")
    @patch("docforge.core.pipeline.ChunkingEngine")
    @patch("docforge.core.pipeline.MetadataGenerator")
    @patch("docforge.core.pipeline.StorageEngine")
    async def test_run_full_success(  # ruff: ignore[too-many-locals, too-many-positional-arguments]
        self,
        mock_store_cls: MagicMock,
        mock_meta_cls: MagicMock,
        mock_chunk_cls: MagicMock,
        mock_class_cls: MagicMock,
        mock_ext_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        config = _config(tmp_path)
        discovery = _make_discovery_result()
        fetch_results = [_make_fetch_result()]
        extracted = _make_extracted_page()
        classified = _make_classified_page()
        chunk = _make_chunk()

        mock_discovery = AsyncMock()
        mock_discovery.discover.return_value = discovery

        mock_crawler = AsyncMock()
        mock_crawler.crawl.return_value = fetch_results

        mock_extractor = AsyncMock()
        mock_extractor.extract.return_value = extracted
        mock_ext_cls.return_value = mock_extractor

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = classified
        mock_class_cls.return_value = mock_classifier

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunk_cls.return_value = mock_chunker

        mock_meta = MagicMock()
        mock_meta.generate.return_value = [chunk]
        mock_meta_cls.return_value = mock_meta

        mock_store = AsyncMock()
        mock_store_cls.return_value = mock_store
        mock_metadata_store = MagicMock()
        mock_store.metadata_store = mock_metadata_store

        mock_embed_engine = AsyncMock()
        mock_embed_engine.embed.return_value = []

        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimension = 4

        pipeline = Pipeline(config=config)
        pipeline._discovery = mock_discovery
        pipeline._crawler = mock_crawler
        pipeline._embedding_engine = mock_embed_engine
        pipeline._embedding_provider = mock_provider

        result = await pipeline.run("test")

        assert result.status == "completed"
        assert len(result.versions) == 1
        assert result.versions[0].status == "completed"
        mock_discovery.discover.assert_awaited_once_with("test")
        mock_crawler.crawl.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("docforge.core.pipeline.ExtractionEngine")
    @patch("docforge.core.pipeline.ClassificationEngine")
    @patch("docforge.core.pipeline.ChunkingEngine")
    @patch("docforge.core.pipeline.MetadataGenerator")
    @patch("docforge.core.pipeline.StorageEngine")
    async def test_run_full_with_specific_version(  # ruff: ignore[too-many-positional-arguments]
        self,
        mock_store_cls: MagicMock,
        mock_meta_cls: MagicMock,
        mock_chunk_cls: MagicMock,
        mock_class_cls: MagicMock,
        mock_ext_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        config = _config(tmp_path)
        discovery = DiscoveryResult(
            software="test",
            display_name="Test",
            base_url="https://example.com/docs/",
            versions=["2.0", "1.0"],
            latest_version="2.0",
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover.return_value = discovery

        mock_crawler = AsyncMock()
        mock_crawler.crawl.return_value = []

        mock_extractor = AsyncMock()
        mock_extractor.extract.return_value = _make_extracted_page()
        mock_ext_cls.return_value = mock_extractor

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = _make_classified_page()
        mock_class_cls.return_value = mock_classifier

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []
        mock_chunk_cls.return_value = mock_chunker

        mock_meta = MagicMock()
        mock_meta.generate.return_value = []
        mock_meta_cls.return_value = mock_meta

        mock_store = AsyncMock()
        mock_store_cls.return_value = mock_store
        mock_store.metadata_store = MagicMock()

        mock_embed_engine = AsyncMock()
        mock_embed_engine.embed.return_value = []

        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimension = 4

        pipeline = Pipeline(config=config)
        pipeline._discovery = mock_discovery
        pipeline._crawler = mock_crawler
        pipeline._embedding_engine = mock_embed_engine
        pipeline._embedding_provider = mock_provider

        result = await pipeline.run("test", version="1.0")

        assert result.status == "completed"
        assert len(result.versions) == 1
        assert result.versions[0].version == "1.0"

    @pytest.mark.asyncio
    async def test_run_incremental_not_found_software(self, tmp_path: Any) -> None:
        config = _config(tmp_path)
        pipeline = Pipeline(config=config)

        result = await pipeline.run("nonexistent", mode="incremental")
        assert result.status == "failed"

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_run_reembed_raises_not_implemented(self, tmp_path: Any) -> None:
        config = _config(tmp_path)
        pipeline = Pipeline(config=config)

        with pytest.raises(NotImplementedError, match="Task 17"):
            await pipeline.run("test", mode="reembed")

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_run_unknown_mode_fails(self, tmp_path: Any) -> None:
        config = _config(tmp_path)
        pipeline = Pipeline(config=config)

        result = await pipeline.run("test", mode="unknown")
        assert result.status == "failed"
        assert result.error is not None

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_discovery_error_propagates(self, tmp_path: Any) -> None:
        config = _config(tmp_path)

        mock_discovery = AsyncMock()
        mock_discovery.discover.side_effect = ValueError("Not found")

        pipeline = Pipeline(config=config)
        pipeline._discovery = mock_discovery

        result = await pipeline.run("test")

        assert result.status == "failed"
        assert "Not found" in (result.error or "")

        await pipeline.close()

    @pytest.mark.asyncio
    @patch("docforge.core.pipeline.ExtractionEngine")
    @patch("docforge.core.pipeline.ClassificationEngine")
    @patch("docforge.core.pipeline.ChunkingEngine")
    @patch("docforge.core.pipeline.MetadataGenerator")
    @patch("docforge.core.pipeline.StorageEngine")
    async def test_page_error_skipped_pipeline_continues(  # ruff: ignore[too-many-positional-arguments]
        self,
        mock_store_cls: MagicMock,
        mock_meta_cls: MagicMock,
        mock_chunk_cls: MagicMock,
        mock_class_cls: MagicMock,
        mock_ext_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        config = _config(tmp_path)

        mock_discovery = AsyncMock()
        mock_discovery.discover.return_value = _make_discovery_result()

        mock_crawler = AsyncMock()
        good_page = _make_fetch_result(url="https://example.com/docs/1.0/good")
        bad_page = _make_fetch_result(url="https://example.com/docs/1.0/bad")
        mock_crawler.crawl.return_value = [good_page, bad_page]

        mock_extractor = AsyncMock()
        mock_extractor.extract.side_effect = [
            _make_extracted_page(url="https://example.com/docs/1.0/good"),
            ValueError("Bad page"),
        ]
        mock_ext_cls.return_value = mock_extractor

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = _make_classified_page()
        mock_class_cls.return_value = mock_classifier

        chunk = _make_chunk()
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunk_cls.return_value = mock_chunker

        mock_meta = MagicMock()
        mock_meta.generate.return_value = [chunk]
        mock_meta_cls.return_value = mock_meta

        mock_store = AsyncMock()
        mock_store_cls.return_value = mock_store
        mock_store.metadata_store = MagicMock()

        mock_embed_engine = AsyncMock()
        mock_embed_engine.embed.return_value = []

        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimension = 4

        pipeline = Pipeline(config=config)
        pipeline._discovery = mock_discovery
        pipeline._crawler = mock_crawler
        pipeline._embedding_engine = mock_embed_engine
        pipeline._embedding_provider = mock_provider

        result = await pipeline.run("test")

        assert result.status == "completed"
        assert result.versions[0].extraction.pages_processed == 1
        assert result.versions[0].extraction.pages_failed == 1

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, tmp_path: Any) -> None:
        config = _config(tmp_path)
        pipeline = Pipeline(config=config)
        await pipeline.close()
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_close_prevents_new_run(self, tmp_path: Any) -> None:
        config = _config(tmp_path)
        pipeline = Pipeline(config=config)
        await pipeline.close()

        with pytest.raises(RuntimeError, match="Pipeline is closed"):
            await pipeline.run("test")
