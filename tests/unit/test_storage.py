"""Unit tests for the storage layer — MetadataStore, backends, and StorageEngine."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import numpy
import pytest

from docforge.core.config import load_config
from docforge.core.models import ChunkMetadata, EmbeddedChunk, PageType
from docforge.storage.backends.chromadb import ChromaDBStore
from docforge.storage.backends.faiss import FAISSStore
from docforge.storage.engine import StorageEngine, _collection_name
from docforge.storage.metadata_store import MetadataStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedded_chunk(
    text: str,
    vector: list[float] | None = None,
    software: str = "test",
    version: str = "1.0",
    chunk_index: int = 0,
    chunk_id: str | None = None,
) -> EmbeddedChunk:
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    cid = chunk_id or hashlib.sha256(f"{software}|{version}|{text}".encode()).hexdigest()
    meta = ChunkMetadata(
        chunk_id=cid,
        parent_page_id="page_test",
        software=software,
        version=version,
        url="https://example.com/docs",
        title="Test Page",
        page_type=PageType.GUIDE,
        section_heading="Introduction",
        chunk_index=chunk_index,
        total_chunks=3,
        has_code=False,
        code_languages=[],
        content_hash=content_hash,
        crawl_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        embedding_model="test-model",
        embedding_dimension=len(vector) if vector else 4,
        breadcrumb=[],
        docforge_version="0.1.0-dev",
    )
    vec = vector or [float(chunk_index), 0.0, 0.0, 1.0]
    return EmbeddedChunk(content=text, metadata=meta, vector=vec)


def _normalize(v: list[float]) -> list[float]:
    arr = numpy.array(v, dtype=numpy.float32)
    norm = numpy.linalg.norm(arr)
    result: list[float] = (arr / norm).tolist() if norm > 0 else v
    return result


# ---------------------------------------------------------------------------
# MetadataStore
# ---------------------------------------------------------------------------


class TestMetadataStore:
    @pytest.fixture
    def store(self) -> MetadataStore:
        tmp = tempfile.mktemp(suffix=".db")
        return MetadataStore(tmp)

    def test_upsert_and_get_software(self, store: MetadataStore) -> None:
        store.upsert_software("postgresql", "PostgreSQL", '{"key": "val"}')
        result = store.get_software("postgresql")
        assert result is not None
        assert result["software"] == "postgresql"
        assert result["display_name"] == "PostgreSQL"

    def test_get_software_not_found(self, store: MetadataStore) -> None:
        assert store.get_software("nonexistent") is None

    def test_list_software(self, store: MetadataStore) -> None:
        store.upsert_software("a", "A")
        store.upsert_software("b", "B")
        items = store.list_software()
        assert len(items) == 2

    def test_delete_software(self, store: MetadataStore) -> None:
        store.upsert_software("test", "Test")
        store.upsert_version("test", "1.0")
        store.delete_software("test")
        assert store.get_software("test") is None
        assert store.get_version("test", "1.0") is None

    def test_upsert_and_get_version(self, store: MetadataStore) -> None:
        store.upsert_version("postgresql", "17", page_count=10, chunk_count=500)
        result = store.get_version("postgresql", "17")
        assert result is not None
        assert result["page_count"] == 10
        assert result["chunk_count"] == 500

    def test_list_versions(self, store: MetadataStore) -> None:
        store.upsert_version("p", "17")
        store.upsert_version("p", "16")
        versions = store.list_versions("p")
        assert len(versions) == 2

    def test_delete_version(self, store: MetadataStore) -> None:
        store.upsert_version("p", "17")
        store.upsert_page_state("https://example.com", "p", "17")
        store.delete_version("p", "17")
        assert store.get_version("p", "17") is None

    def test_page_state_lifecycle(self, store: MetadataStore) -> None:
        store.upsert_page_state("https://example.com/page", "test", "1.0", content_hash="abc")
        state = store.get_page_state("https://example.com/page")
        assert state is not None
        assert state["content_hash"] == "abc"
        assert state["software"] == "test"

        store.delete_page_state("https://example.com/page")
        assert store.get_page_state("https://example.com/page") is None

    def test_list_page_states(self, store: MetadataStore) -> None:
        store.upsert_page_state("https://example.com/1", "test", "1.0")
        store.upsert_page_state("https://example.com/2", "test", "1.0")
        pages = store.list_page_states("test", "1.0")
        assert len(pages) == 2

    def test_pipeline_run_lifecycle(self, store: MetadataStore) -> None:
        run_id = store.create_run("test", "1.0", mode="full")
        assert run_id > 0

        run = store.get_run(run_id)
        assert run is not None
        assert run["status"] == "running"

        store.complete_run(run_id, status="completed", page_count=5, chunk_count=100)
        run = store.get_run(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert run["page_count"] == 5
        assert run["chunk_count"] == 100

    def test_list_runs(self, store: MetadataStore) -> None:
        store.create_run("a", "1.0")
        store.create_run("b", "2.0")
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_filtered(self, store: MetadataStore) -> None:
        store.create_run("a", "1.0")
        store.create_run("a", "2.0")
        store.create_run("b", "1.0")
        runs = store.list_runs(software="a")
        assert len(runs) == 2

    def test_get_software_stats(self, store: MetadataStore) -> None:
        store.upsert_software("test", "Test")
        store.upsert_version("test", "1.0", page_count=5, chunk_count=100)
        store.upsert_page_state("https://example.com/p1", "test", "1.0")
        store.upsert_page_state("https://example.com/p2", "test", "1.0")

        stats = store.get_software_stats("test")
        assert stats["software"] == "test"
        assert stats["version_count"] == 1
        assert stats["page_count"] == 2
        assert stats["chunk_count"] == 100

    def test_persistence(self) -> None:
        tmp = tempfile.mktemp(suffix=".db")
        m1 = MetadataStore(tmp)
        m1.upsert_software("test", "Test")
        m1.close()

        m2 = MetadataStore(tmp)
        result = m2.get_software("test")
        assert result is not None
        assert result["display_name"] == "Test"
        m2.close()

    def test_close_reopen(self, store: MetadataStore) -> None:
        store.upsert_software("test", "Test")
        store.close()
        store.upsert_software("test2", "Test2")
        assert store.get_software("test2") is not None


# ---------------------------------------------------------------------------
# ChromaDBStore
# ---------------------------------------------------------------------------


class TestChromaDBStore:
    @pytest.fixture
    async def store(self) -> AsyncGenerator[ChromaDBStore, None]:
        tmp_path = Path(tempfile.mkdtemp())
        s = ChromaDBStore()
        await s.initialize(
            {
                "path": str(tmp_path),
                "collection_name": "test_collection",
                "dimension": 4,
            }
        )
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, store: ChromaDBStore) -> None:
        chunks = [
            _make_embedded_chunk("hello", [0.1, 0.2, 0.3, 0.4], chunk_index=0),
            _make_embedded_chunk("world", [0.5, 0.6, 0.7, 0.8], chunk_index=1),
        ]
        await store.upsert(chunks)
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, store: ChromaDBStore) -> None:
        cid = "dedup_test"
        chunk = _make_embedded_chunk("data", [0.1, 0.2, 0.3, 0.4], chunk_id=cid)
        await store.upsert([chunk])
        await store.upsert([chunk])
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_search_returns_top_k(self, store: ChromaDBStore) -> None:
        chunks = [
            _make_embedded_chunk("cat", [1.0, 0.0, 0.0, 0.0], chunk_index=0),
            _make_embedded_chunk("dog", [0.0, 1.0, 0.0, 0.0], chunk_index=1),
            _make_embedded_chunk("bird", [0.0, 0.0, 1.0, 0.0], chunk_index=2),
        ]
        await store.upsert(chunks)
        results = await store.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].chunk_id == chunks[0].metadata.chunk_id

    @pytest.mark.asyncio
    async def test_delete_with_filters(self, store: ChromaDBStore) -> None:
        chunks = [
            _make_embedded_chunk("a", [0.1, 0.0, 0.0, 0.0], software="s1", chunk_index=0),
            _make_embedded_chunk("b", [0.0, 0.1, 0.0, 0.0], software="s2", chunk_index=1),
        ]
        await store.upsert(chunks)
        await store.delete({"software": "s1"})
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, store: ChromaDBStore) -> None:
        chunks = [
            _make_embedded_chunk("pg", [0.1, 0.0, 0.0, 0.0], software="postgresql", chunk_index=0),
            _make_embedded_chunk("my", [0.0, 0.1, 0.0, 0.0], software="mysql", chunk_index=1),
        ]
        await store.upsert(chunks)
        results = await store.search([0.1, 0.0, 0.0, 0.0], k=5, filters={"software": "postgresql"})
        assert len(results) == 1
        assert results[0].metadata.software == "postgresql"

    @pytest.mark.asyncio
    async def test_empty_count(self, store: ChromaDBStore) -> None:
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_count_with_filters(self, store: ChromaDBStore) -> None:
        chunks = [
            _make_embedded_chunk("a", [0.1, 0.0, 0.0, 0.0], software="s1", chunk_index=0),
            _make_embedded_chunk("b", [0.0, 0.1, 0.0, 0.0], software="s2", chunk_index=1),
        ]
        await store.upsert(chunks)
        assert await store.count(filters={"software": "s1"}) == 1


# ---------------------------------------------------------------------------
# FAISSStore
# ---------------------------------------------------------------------------


class TestFAISSStore:
    @pytest.fixture
    async def store(self) -> AsyncGenerator[FAISSStore, None]:
        tmp_path = Path(tempfile.mkdtemp())
        s = FAISSStore()
        await s.initialize(
            {
                "path": str(tmp_path),
                "collection_name": "test_faiss",
                "dimension": 4,
            }
        )
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, store: FAISSStore) -> None:
        chunks = [
            _make_embedded_chunk("hello", [1.0, 0.0, 0.0, 0.0], chunk_index=0),
            _make_embedded_chunk("world", [0.0, 1.0, 0.0, 0.0], chunk_index=1),
        ]
        await store.upsert(chunks)
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, store: FAISSStore) -> None:
        cid = "faiss_dedup"
        chunk = _make_embedded_chunk("data", [0.1, 0.2, 0.3, 0.4], chunk_id=cid)
        await store.upsert([chunk])
        await store.upsert([chunk])
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_search_returns_top_k(self, store: FAISSStore) -> None:
        chunks = [
            _make_embedded_chunk("cat", [1.0, 0.0, 0.0, 0.0], chunk_index=0),
            _make_embedded_chunk("dog", [0.0, 1.0, 0.0, 0.0], chunk_index=1),
        ]
        await store.upsert(chunks)
        results = await store.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].metadata.software == "test"

    @pytest.mark.asyncio
    async def test_correct_ranking(self, store: FAISSStore) -> None:
        cat_vec = _normalize([1.0, 0.0, 0.0, 0.0])
        dog_vec = _normalize([0.0, 1.0, 0.0, 0.0])
        chunks = [
            _make_embedded_chunk("cat", cat_vec, chunk_index=0),
            _make_embedded_chunk("dog", dog_vec, chunk_index=1),
        ]
        await store.upsert(chunks)
        query = _normalize([1.0, 0.0, 0.0, 0.0])
        results = await store.search(query, k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_delete_with_filters(self, store: FAISSStore) -> None:
        chunks = [
            _make_embedded_chunk("a", [0.1, 0.0, 0.0, 0.0], software="s1", chunk_index=0),
            _make_embedded_chunk("b", [0.0, 0.1, 0.0, 0.0], software="s2", chunk_index=1),
        ]
        await store.upsert(chunks)
        await store.delete({"software": "s1"})
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, store: FAISSStore) -> None:
        chunks = [
            _make_embedded_chunk("pg", [0.1, 0.0, 0.0, 0.0], software="postgresql", chunk_index=0),
            _make_embedded_chunk("my", [0.0, 0.1, 0.0, 0.0], software="mysql", chunk_index=1),
        ]
        await store.upsert(chunks)
        results = await store.search([0.1, 0.0, 0.0, 0.0], k=5, filters={"software": "postgresql"})
        assert len(results) == 1
        assert results[0].metadata.software == "postgresql"

    @pytest.mark.asyncio
    async def test_persistence(self) -> None:
        tmp_path = Path(tempfile.mkdtemp())
        s1 = FAISSStore()
        await s1.initialize(
            {
                "path": str(tmp_path),
                "collection_name": "persist_test",
                "dimension": 4,
            }
        )
        chunk = _make_embedded_chunk("persist me", [0.5, 0.5, 0.0, 0.0], chunk_index=0)
        await s1.upsert([chunk])
        await s1.close()

        s2 = FAISSStore()
        await s2.initialize(
            {
                "path": str(tmp_path),
                "collection_name": "persist_test",
                "dimension": 4,
            }
        )
        assert await s2.count() == 1
        results = await s2.search([0.5, 0.5, 0.0, 0.0], k=1)
        assert len(results) == 1
        await s2.close()

    @pytest.mark.asyncio
    async def test_empty_search(self, store: FAISSStore) -> None:
        results = await store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_count(self, store: FAISSStore) -> None:
        assert await store.count() == 0


# ---------------------------------------------------------------------------
# StorageEngine
# ---------------------------------------------------------------------------


class TestStorageEngine:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self) -> None:
        tmp_path = Path(tempfile.mkdtemp())
        config = load_config(overrides={"storage": {"backend": "faiss", "path": str(tmp_path)}})
        engine = StorageEngine(config, software="test", version="1.0")
        await engine.initialize(dimension=4)
        assert await engine.count() == 0
        await engine.close()

    @pytest.mark.asyncio
    async def test_upsert_and_search(self) -> None:
        tmp_path = Path(tempfile.mkdtemp())
        config = load_config(overrides={"storage": {"backend": "faiss", "path": str(tmp_path)}})
        engine = StorageEngine(config, software="test", version="1.0")
        await engine.initialize(dimension=4, model_name="test-model")

        chunks = [
            _make_embedded_chunk("alpha", [1.0, 0.0, 0.0, 0.0], chunk_index=0),
            _make_embedded_chunk("beta", [0.0, 1.0, 0.0, 0.0], chunk_index=1),
        ]
        await engine.upsert(chunks)
        assert await engine.count() == 2

        results = await engine.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self) -> None:
        config = load_config(overrides={"storage": {"backend": "nonexistent"}})
        engine = StorageEngine(config)
        with pytest.raises(ValueError, match="Unknown storage backend"):
            await engine.initialize()

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self) -> None:
        config = load_config()
        engine = StorageEngine(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.search([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Collection name helper
# ---------------------------------------------------------------------------


class TestCollectionNaming:
    def test_collection_name_format(self) -> None:
        name = _collection_name("postgresql", "17", "BAAI/bge-base-en-v1.5")
        assert name.startswith("docforge_postgresql_17_")
        assert len(name) > len("docforge_postgresql_17_")
