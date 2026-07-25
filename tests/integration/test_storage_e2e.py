"""End-to-end integration test for the storage layer.

Upserts 1,000 synthetic chunks, searches, and verifies result correctness.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docforge.core.config import load_config
from docforge.core.models import ChunkMetadata, EmbeddedChunk, PageType
from docforge.storage.engine import StorageEngine


def _make_chunk(text: str, idx: int, software: str = "test") -> EmbeddedChunk:
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    cid = hashlib.sha256(f"{software}|1.0|{text}".encode()).hexdigest()
    meta = ChunkMetadata(
        chunk_id=cid,
        parent_page_id="page_e2e",
        software=software,
        version="1.0",
        url="https://example.com/docs",
        title="E2E Test Page",
        page_type=PageType.GUIDE,
        section_heading=f"Section {idx // 10}",
        chunk_index=idx,
        total_chunks=1000,
        has_code=idx % 5 == 0,
        code_languages=["python"] if idx % 5 == 0 else [],
        content_hash=content_hash,
        crawl_timestamp=datetime(2025, 6, 1, tzinfo=UTC),
        embedding_model="test-model",
        embedding_dimension=8,
        breadcrumb=[],
        docforge_version="0.1.0-dev",
    )
    vec = [1.0 if i == idx % 8 else 0.0 for i in range(8)]
    return EmbeddedChunk(content=text, metadata=meta, vector=vec)


@pytest.mark.asyncio
async def test_upsert_1000_chunks_and_search() -> None:
    tmp_path = Path(tempfile.mkdtemp())
    config = load_config(overrides={"storage": {"backend": "faiss", "path": str(tmp_path)}})
    engine = StorageEngine(config, software="test", version="1.0")
    await engine.initialize(dimension=8, model_name="test-model")

    chunks = [_make_chunk(f"chunk content {i}", i) for i in range(1000)]
    await engine.upsert(chunks)
    assert await engine.count() == 1000

    query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = await engine.search(query, k=5)

    assert len(results) == 5
    assert results[0].score >= results[-1].score
    top = results[0]
    assert top.metadata.software == "test"
    assert top.metadata.version == "1.0"
    assert top.metadata.chunk_index % 8 == 0

    await engine.close()


@pytest.mark.asyncio
async def test_delete_filtered() -> None:
    tmp_path = Path(tempfile.mkdtemp())
    config = load_config(overrides={"storage": {"backend": "faiss", "path": str(tmp_path)}})
    engine = StorageEngine(config, software="test", version="1.0")
    await engine.initialize(dimension=8, model_name="test-model")

    chunks_a = [_make_chunk(f"a content {i}", i, software="alpha") for i in range(50)]
    chunks_b = [_make_chunk(f"b content {i}", i, software="beta") for i in range(50)]
    await engine.upsert(chunks_a + chunks_b)
    assert await engine.count() == 100

    await engine.delete({"software": "alpha"})
    assert await engine.count() == 50

    remaining = await engine.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=100)
    assert all(r.metadata.software == "beta" for r in remaining)

    await engine.close()
