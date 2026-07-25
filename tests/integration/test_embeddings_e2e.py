"""End-to-end integration tests for the embedding layer.

These tests require ``sentence-transformers`` and will download the
model on first run. Marked as ``slow`` because model loading is heavy.
"""

from __future__ import annotations

import tempfile

import pytest

from docforge.core.models import Chunk, ChunkMetadata, PageType
from docforge.embeddings.cache import EmbeddingCache
from docforge.embeddings.engine import EmbeddingEngine
from docforge.embeddings.providers.sentence_transformers import (
    SentenceTransformersProvider,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
]


def _chunk(text: str, idx: int) -> Chunk:
    meta = ChunkMetadata(
        chunk_id=f"e2e-{idx}",
        parent_page_id="e2e-page",
        software="test-e2e",
        version="1.0",
        url="https://example.com/docs",
        title="E2E Test",
        page_type=PageType.GUIDE,
        section_heading=f"Section {idx}",
        chunk_index=idx,
        total_chunks=10,
        has_code=False,
        content_hash=f"hash-{idx}",
        crawl_timestamp="2025-01-01T00:00:00Z",
        embedding_model="",
        embedding_dimension=0,
        docforge_version="0.1.0-dev",
    )
    return Chunk(content=text, metadata=meta)


@pytest.mark.asyncio
async def test_sentence_transformers_embeds_batch() -> None:
    provider = SentenceTransformersProvider(
        model_name="BAAI/bge-small-en-v1.5"
    )
    texts = [f"This is test document number {i}." for i in range(64)]
    vectors = await provider.embed_batch(texts)
    assert len(vectors) == 64
    assert all(len(v) == provider.dimension for v in vectors)
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


@pytest.mark.asyncio
async def test_engine_with_sentence_transformers() -> None:
    provider = SentenceTransformersProvider(
        model_name="BAAI/bge-small-en-v1.5"
    )
    tmp = tempfile.mktemp(suffix=".db")
    cache = EmbeddingCache(tmp)
    engine = EmbeddingEngine(provider, cache=cache, batch_size=16)

    chunks = [_chunk(f"Document chunk number {i}.", i) for i in range(10)]
    embedded = await engine.embed(chunks)

    assert len(embedded) == 10
    for e in embedded:
        assert len(e.vector) == provider.dimension
        assert isinstance(e.vector[0], float)

    embedded2 = await engine.embed(chunks)
    assert len(embedded2) == 10
    for e1, e2 in zip(embedded, embedded2, strict=False):
        assert e1.vector == e2.vector
    engine.close()


@pytest.mark.asyncio
async def test_engine_batches_large_corpus() -> None:
    provider = SentenceTransformersProvider(
        model_name="BAAI/bge-small-en-v1.5"
    )
    engine = EmbeddingEngine(provider, batch_size=32)

    chunks = [_chunk(f"Chunk {i}: {'hello world ' * 10}", i) for i in range(100)]
    embedded = await engine.embed(chunks)
    assert len(embedded) == 100
    engine.close()
