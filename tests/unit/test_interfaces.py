"""Unit tests for plugin ABC interfaces."""

from __future__ import annotations

from typing import Any

import pytest

from docforge.core.interfaces import (
    ChunkingStrategy,
    ContentExtractor,
    CrawlFetcher,
    DiscoveryProvider,
    EmbeddingProvider,
    PageClassifier,
    VectorStore,
)
from docforge.core.models import (
    Chunk,
    ChunkMetadata,
    ClassifiedPage,
    DiscoveryResult,
    ExtractedPage,
    FetchResult,
    PageType,
)

# ---------------------------------------------------------------------------
# Dummy implementations
# ---------------------------------------------------------------------------


class DummyDiscovery(DiscoveryProvider):
    async def discover(self, name: str) -> DiscoveryResult:
        return DiscoveryResult(
            software=name,
            display_name=name.title(),
            base_url=f"https://{name}.example.com/docs/",
            versions=["1"],
            latest_version="1",
        )


class DummyFetcher(CrawlFetcher):
    async def fetch(self, url: str) -> FetchResult:
        return FetchResult(url=url, status_code=200, html="<html></html>")


class DummyExtractor(ContentExtractor):
    async def extract(self, fetch_result: FetchResult) -> ExtractedPage:
        return ExtractedPage(
            url=fetch_result.url,
            title="Test",
            markdown="# Test",
        )


class DummyClassifier(PageClassifier):
    def classify(self, page: ExtractedPage) -> ClassifiedPage:
        return ClassifiedPage(
            url=page.url,
            title=page.title,
            markdown=page.markdown,
            page_type=PageType.UNKNOWN,
            confidence=0.5,
        )


class DummyChunker(ChunkingStrategy):
    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        meta = ChunkMetadata(
            chunk_id="a" * 64,
            parent_page_id="b" * 64,
            software="test",
            version="1",
            url=page.url,
            title=page.title,
            page_type=PageType.UNKNOWN,
            chunk_index=0,
            total_chunks=1,
            has_code=False,
            content_hash="c" * 64,
            crawl_timestamp="2026-01-01T00:00:00Z",
            embedding_model="test",
            embedding_dimension=8,
            docforge_version="0.1.0-dev",
        )
        return [Chunk(content=page.markdown, metadata=meta)]


class DummyEmbedder(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "dummy-model"

    @property
    def dimension(self) -> int:
        return 8

    @property
    def max_tokens(self) -> int:
        return 512

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


class DummyVectorStore(VectorStore):
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._initialized = False

    async def initialize(self, config: dict[str, Any]) -> None:
        self._initialized = True

    async def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._data[c.metadata.chunk_id] = c

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        return list(self._data.values())[:k]

    async def delete(self, filters: dict[str, Any]) -> None:
        self._data.clear()

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        return len(self._data)

    async def close(self) -> None:
        self._data.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoveryProvider:
    def test_dummy_instantiates(self) -> None:
        d = DummyDiscovery()
        assert isinstance(d, DiscoveryProvider)

    @pytest.mark.asyncio
    async def test_discover_returns_result(self) -> None:
        d = DummyDiscovery()
        result = await d.discover("postgresql")
        assert result.software == "postgresql"
        assert result.display_name == "Postgresql"

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            DiscoveryProvider()  # type: ignore[abstract]


class TestCrawlFetcher:
    def test_dummy_instantiates(self) -> None:
        f = DummyFetcher()
        assert isinstance(f, CrawlFetcher)

    @pytest.mark.asyncio
    async def test_fetch_returns_result(self) -> None:
        f = DummyFetcher()
        result = await f.fetch("https://example.com")
        assert result.status_code == 200

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            CrawlFetcher()  # type: ignore[abstract]


class TestContentExtractor:
    def test_dummy_instantiates(self) -> None:
        e = DummyExtractor()
        assert isinstance(e, ContentExtractor)

    @pytest.mark.asyncio
    async def test_extract_returns_result(self) -> None:
        e = DummyExtractor()
        fetch = FetchResult(url="https://x.com", status_code=200, html="<p>hi</p>")
        result = await e.extract(fetch)
        assert result.title == "Test"

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ContentExtractor()  # type: ignore[abstract]


class TestPageClassifier:
    def test_dummy_instantiates(self) -> None:
        c = DummyClassifier()
        assert isinstance(c, PageClassifier)

    def test_classify_returns_result(self) -> None:
        c = DummyClassifier()
        page = ExtractedPage(url="https://x.com", title="T", markdown="m")
        result = c.classify(page)
        assert result.page_type == PageType.UNKNOWN

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            PageClassifier()  # type: ignore[abstract]


class TestChunkingStrategy:
    def test_dummy_instantiates(self) -> None:
        cs = DummyChunker()
        assert isinstance(cs, ChunkingStrategy)

    def test_chunk_returns_list(self) -> None:
        cs = DummyChunker()
        page = ClassifiedPage(
            url="https://x.com",
            title="T",
            markdown="content",
            page_type=PageType.UNKNOWN,
            confidence=0.5,
        )
        chunks = cs.chunk(page)
        assert len(chunks) == 1
        assert chunks[0].content == "content"

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ChunkingStrategy()  # type: ignore[abstract]


class TestEmbeddingProvider:
    def test_dummy_instantiates(self) -> None:
        e = DummyEmbedder()
        assert isinstance(e, EmbeddingProvider)

    def test_properties(self) -> None:
        e = DummyEmbedder()
        assert e.model_name == "dummy-model"
        assert e.dimension == 8
        assert e.max_tokens == 512

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        e = DummyEmbedder()
        vectors = await e.embed_batch(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 8

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]


class TestVectorStore:
    def test_dummy_instantiates(self) -> None:
        vs = DummyVectorStore()
        assert isinstance(vs, VectorStore)

    @pytest.mark.asyncio
    async def test_initialize_and_upsert(self) -> None:
        vs = DummyVectorStore()
        await vs.initialize({"path": "/tmp/test"})
        assert vs._initialized

    @pytest.mark.asyncio
    async def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]
