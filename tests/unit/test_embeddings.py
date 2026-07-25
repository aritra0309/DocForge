"""Unit tests for the embedding layer — cache, providers, engine."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy
import pytest

from docforge.core.interfaces import EmbeddingProvider
from docforge.core.models import Chunk, ChunkMetadata, EmbeddedChunk, PageType
from docforge.embeddings.cache import EmbeddingCache
from docforge.embeddings.engine import EmbeddingEngine, EmbeddingProgress, TokenBucket
from docforge.embeddings.providers.openai import OpenAIEmbeddingProvider
from docforge.embeddings.providers.sentence_transformers import (
    SentenceTransformersProvider,
)
from docforge.embeddings.providers.voyage import VoyageEmbeddingProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, content_hash: str | None = None) -> Chunk:
    c_hash = content_hash or hashlib.sha256(text.encode()).hexdigest()
    meta = ChunkMetadata(
        chunk_id="chunk_test",
        parent_page_id="page_1",
        software="test",
        version="1.0",
        url="https://example.com/docs",
        title="Test Page",
        page_type=PageType.GUIDE,
        section_heading="Introduction",
        chunk_index=0,
        total_chunks=3,
        has_code=False,
        content_hash=c_hash,
        crawl_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        embedding_model="",
        embedding_dimension=0,
        docforge_version="0.1.0-dev",
    )
    return Chunk(content=text, metadata=meta)


def _dummy_provider(dim: int = 4) -> EmbeddingProvider:
    """Create a dummy embedding provider for testing."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.model_name = "test-model"
    provider.dimension = dim
    provider.max_tokens = 512
    mock_embed = AsyncMock(
        side_effect=lambda texts: [[float(i + j) for j in range(dim)] for i in range(len(texts))]
    )
    provider.embed_batch = mock_embed  # type: ignore[method-assign]
    return provider


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_does_not_block_with_tokens(self) -> None:
        bucket = TokenBucket(rate=100, burst=10)
        await bucket.acquire()

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_empty(self) -> None:
        bucket = TokenBucket(rate=1, burst=1)
        await bucket.acquire()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(bucket.acquire(), timeout=0.1)


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    @pytest.fixture
    def cache(self) -> EmbeddingCache:
        tmp = tempfile.mktemp(suffix=".db")
        return EmbeddingCache(tmp)

    def test_put_and_get(self, cache: EmbeddingCache) -> None:
        cache.put("m1", "hash1", [0.1, 0.2, 0.3])
        vec = cache.get("m1", "hash1")
        assert vec == [0.1, 0.2, 0.3]

    def test_cache_miss(self, cache: EmbeddingCache) -> None:
        vec = cache.get("m1", "nonexistent")
        assert vec is None

    def test_put_batch(self, cache: EmbeddingCache) -> None:
        cache.put_batch("m1", ["h1", "h2"], [[1.0, 2.0], [3.0, 4.0]])
        assert cache.get("m1", "h1") == [1.0, 2.0]
        assert cache.get("m1", "h2") == [3.0, 4.0]

    def test_clear_all(self, cache: EmbeddingCache) -> None:
        cache.put("m1", "hash1", [0.1])
        cache.put("m2", "hash2", [0.2])
        cache.clear()
        assert cache.count() == 0

    def test_clear_by_model(self, cache: EmbeddingCache) -> None:
        cache.put("m1", "h1", [0.1])
        cache.put("m2", "h2", [0.2])
        cache.clear(model_name="m1")
        assert cache.count() == 1
        assert cache.get("m2", "h2") == [0.2]

    def test_count(self, cache: EmbeddingCache) -> None:
        cache.put("m1", "h1", [0.1])
        cache.put("m1", "h2", [0.2])
        assert cache.count() == 2
        assert cache.count(model_name="m1") == 2
        assert cache.count(model_name="m2") == 0

    def test_update_existing(self, cache: EmbeddingCache) -> None:
        cache.put("m1", "h1", [0.1, 0.2])
        cache.put("m1", "h1", [9.9, 8.8])
        assert cache.get("m1", "h1") == [9.9, 8.8]

    def test_close(self, cache: EmbeddingCache) -> None:
        cache.put("m1", "h1", [0.1])
        cache.close()
        cache.put("m1", "h2", [0.2])
        assert cache.get("m1", "h2") == [0.2]

    def test_persistence(self) -> None:
        tmp = tempfile.mktemp(suffix=".db")
        c1 = EmbeddingCache(tmp)
        c1.put("m1", "h1", [0.5, 0.6])
        c1.close()
        c2 = EmbeddingCache(tmp)
        assert c2.get("m1", "h1") == [0.5, 0.6]
        c2.close()


# ---------------------------------------------------------------------------
# SentenceTransformersProvider
# ---------------------------------------------------------------------------


class TestSentenceTransformersProvider:
    def test_properties(self) -> None:
        provider = SentenceTransformersProvider(model_name="BAAI/bge-small-en-v1.5")
        assert provider.model_name == "BAAI/bge-small-en-v1.5"
        assert provider.dimension == 384
        assert provider.max_tokens == 512

    def test_default_properties(self) -> None:
        provider = SentenceTransformersProvider()
        assert provider.model_name == "BAAI/bge-base-en-v1.5"
        assert provider.dimension == 768

    @pytest.mark.asyncio
    async def test_embed_batch_mocked(self) -> None:
        provider = SentenceTransformersProvider()
        mock_model = MagicMock()
        mock_model.encode.return_value = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        provider._model = mock_model
        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 2
        mock_model.encode.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingProvider:
    def test_properties(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimension == 512
        assert provider.max_tokens == 8191

    def test_custom_dimensions(self) -> None:
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-large", dimensions=256, api_key="sk-test"
        )
        assert provider.dimension == 256

    @pytest.mark.asyncio
    async def test_embed_batch_mocked(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=[0.1, 0.2]),
            MagicMock(index=1, embedding=[0.3, 0.4]),
        ]
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client
        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# VoyageEmbeddingProvider
# ---------------------------------------------------------------------------


class TestVoyageEmbeddingProvider:
    def test_properties(self) -> None:
        provider = VoyageEmbeddingProvider(api_key="vo-test")
        assert provider.model_name == "voyage-3"
        assert provider.dimension == 1024
        assert provider.max_tokens == 32000

    @pytest.mark.asyncio
    async def test_embed_batch_mocked(self) -> None:
        provider = VoyageEmbeddingProvider(api_key="vo-test")
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        provider._client = mock_client
        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------


class TestEmbeddingEngine:
    @pytest.mark.asyncio
    async def test_embed_empty_list(self) -> None:
        provider = _dummy_provider()
        engine = EmbeddingEngine(provider)
        result = await engine.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_basic(self) -> None:
        provider = _dummy_provider(dim=2)
        chunks = [_make_chunk("hello"), _make_chunk("world")]
        engine = EmbeddingEngine(provider, batch_size=2)
        result = await engine.embed(chunks)
        assert len(result) == 2
        assert all(isinstance(e, EmbeddedChunk) for e in result)
        assert all(len(e.vector) == 2 for e in result)

    @pytest.mark.asyncio
    async def test_embed_with_cache_hits(self) -> None:
        tmp = tempfile.mktemp(suffix=".db")
        cache = EmbeddingCache(tmp)
        provider = _dummy_provider(dim=2)

        ch1 = _make_chunk("hello", content_hash="h1")
        ch2 = _make_chunk("world", content_hash="h2")
        cache.put("test-model", "h1", [9.9, 8.8])
        cache.put("test-model", "h2", [7.7, 6.6])

        engine = EmbeddingEngine(provider, cache=cache, batch_size=10)
        result = await engine.embed([ch1, ch2])
        assert result[0].vector == [9.9, 8.8]
        assert result[1].vector == [7.7, 6.6]
        embed_mock: AsyncMock = provider.embed_batch  # type: ignore[assignment]
        embed_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_cache_miss_triggers_provider(self) -> None:
        tmp = tempfile.mktemp(suffix=".db")
        cache = EmbeddingCache(tmp)
        provider = _dummy_provider(dim=2)
        chunks = [_make_chunk("hello", content_hash="h1")]
        engine = EmbeddingEngine(provider, cache=cache)
        result = await engine.embed(chunks)
        assert len(result) == 1
        embed_mock: AsyncMock = provider.embed_batch  # type: ignore[assignment]
        embed_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embed_persists_to_cache(self) -> None:
        tmp = tempfile.mktemp(suffix=".db")
        cache = EmbeddingCache(tmp)
        provider = _dummy_provider(dim=2)
        chunks = [_make_chunk("hello", content_hash="h1")]
        engine = EmbeddingEngine(provider, cache=cache)
        await engine.embed(chunks)
        cached = cache.get("test-model", "h1")
        assert cached is not None
        assert len(cached) == 2

    @pytest.mark.asyncio
    async def test_embed_batching(self) -> None:
        provider = _dummy_provider(dim=2)
        chunks = [_make_chunk(f"text_{i}") for i in range(5)]
        engine = EmbeddingEngine(provider, batch_size=2)
        result = await engine.embed(chunks)
        assert len(result) == 5
        embed_mock: AsyncMock = provider.embed_batch  # type: ignore[assignment]
        assert embed_mock.await_count == 3

    @pytest.mark.asyncio
    async def test_progress_callback(self) -> None:
        provider = _dummy_provider(dim=2)
        chunks = [_make_chunk(f"text_{i}") for i in range(3)]
        progress: list[EmbeddingProgress] = []

        engine = EmbeddingEngine(provider, batch_size=2, progress_callback=progress.append)
        await engine.embed(chunks)
        assert len(progress) == 2
        assert progress[0].total_chunks == 3
        assert progress[0].batch_index == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        provider = MagicMock(spec=EmbeddingProvider)
        provider.model_name = "test-model"
        provider.dimension = 2
        provider.max_tokens = 512
        mock_embed = AsyncMock(
            side_effect=[RuntimeError("API down"), RuntimeError("still down"), [[0.1, 0.2]]]
        )
        provider.embed_batch = mock_embed  # type: ignore[method-assign]
        chunks = [_make_chunk("hello")]
        engine = EmbeddingEngine(provider, max_retries=3)
        result = await engine.embed(chunks)
        assert len(result) == 1
        assert mock_embed.await_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self) -> None:
        provider = MagicMock(spec=EmbeddingProvider)
        provider.model_name = "test-model"
        provider.dimension = 2
        provider.max_tokens = 512
        mock_embed = AsyncMock(side_effect=RuntimeError("always down"))
        provider.embed_batch = mock_embed  # type: ignore[method-assign]
        chunks = [_make_chunk("hello")]
        engine = EmbeddingEngine(provider, max_retries=2)
        with pytest.raises(RuntimeError, match="failed after 2 retries"):
            await engine.embed(chunks)

    @pytest.mark.asyncio
    async def test_missing_content_hash(self) -> None:
        provider = _dummy_provider(dim=2)
        chunk = _make_chunk("no hash", content_hash="")
        engine = EmbeddingEngine(provider)
        result = await engine.embed([chunk])
        assert len(result) == 1
