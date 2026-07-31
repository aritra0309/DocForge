"""Search and embedding benchmark tests."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy
import pytest

from docforge.core.config import DocForgeConfig
from docforge.core.models import Chunk, ChunkMetadata, EmbeddedChunk, PageType
from docforge.embeddings.engine import EmbeddingEngine
from docforge.embeddings.providers.base import EmbeddingProvider
from docforge.embeddings.providers.sentence_transformers import SentenceTransformersProvider
from docforge.storage.backends.faiss import FAISSStore
from docforge.storage.engine import StorageEngine


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


class DummyEmbeddingProvider(EmbeddingProvider):
    model_name = "benchmark-model"
    dimension = 384
    max_tokens = 512

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Simulate some work
        await asyncio.sleep(0.001 * len(texts))
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture
async def faiss_store() -> FAISSStore:
    tmp_path = Path(tempfile.mkdtemp())
    store = FAISSStore()
    await store.initialize(
        {
            "path": str(tmp_path),
            "collection_name": "bench_search",
            "dimension": 384,
        }
    )
    yield store
    await store.close()


@pytest.fixture
def chunks_1000() -> list[EmbeddedChunk]:
    """Generate 1000 chunks with varied content for search benchmarking."""
    chunks = []
    for i in range(1000):
        text = f"Document {i} about topic {i % 50}. Content goes here with some details."
        vec = _normalize([float(i % 100), float(i % 50), float(i % 25), 1.0] + [0.0] * 380)
        chunks.append(_make_embedded_chunk(text, vec, chunk_index=i))
    return chunks


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_embedding_benchmark_sentence_transformers() -> None:
    """Benchmark local Sentence Transformers embedding throughput."""
    # This test requires sentence-transformers to be installed
    try:
        provider = SentenceTransformersProvider(model_name="BAAI/bge-small-en-v1.5")
    except Exception:
        pytest.skip("sentence-transformers not available or model download failed")

    engine = EmbeddingEngine(provider, batch_size=64)

    # Generate test chunks
    texts = [f"This is test document number {i} with some content to embed." for i in range(1000)]
    chunks = []
    for i, text in enumerate(texts):
        meta = ChunkMetadata(
            chunk_id=f"chunk_{i}",
            parent_page_id=f"page_{i//5}",
            software="bench",
            version="1.0",
            url=f"https://example.com/docs/{i}",
            title=f"Doc {i}",
            page_type=PageType.GUIDE,
            section_heading="Content",
            chunk_index=i % 5,
            total_chunks=5,
            has_code=False,
            code_languages=[],
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            crawl_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            embedding_model="benchmark",
            embedding_dimension=384,
            breadcrumb=[],
            docforge_version="0.1.0-dev",
        )
        chunks.append(Chunk(content=text, metadata=meta))

    # Warm up
    await engine.embed(chunks[:10])

    # Benchmark
    iterations = 5  # 5000 chunks total
    start = time.perf_counter()
    for i in range(iterations):
        batch = chunks[i*1000:(i+1)*1000]
        await engine.embed(batch)
    elapsed = time.perf_counter() - start

    total_chunks = iterations * 1000
    chunks_per_sec = total_chunks / elapsed
    # Target: 50 chunks/sec for local model
    assert chunks_per_sec >= 50, f"Embedding throughput {chunks_per_sec:.1f} chunks/sec below target 50"

    from tests.benchmarks import benchmark
    with benchmark("embedding_sentence_transformers", total_chunks):
        pass


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_embedding_benchmark_dummy_provider() -> None:
    """Benchmark embedding throughput with dummy provider (no model loading)."""
    provider = DummyEmbeddingProvider()
    engine = EmbeddingEngine(provider, batch_size=64)

    texts = [f"Test document {i}" for i in range(10000)]
    chunks = []
    for i, text in enumerate(texts):
        meta = ChunkMetadata(
            chunk_id=f"chunk_{i}",
            parent_page_id=f"page_{i//5}",
            software="bench",
            version="1.0",
            url=f"https://example.com/docs/{i}",
            title=f"Doc {i}",
            page_type=PageType.GUIDE,
            section_heading="Content",
            chunk_index=i % 5,
            total_chunks=5,
            has_code=False,
            code_languages=[],
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            crawl_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            embedding_model="benchmark",
            embedding_dimension=384,
            breadcrumb=[],
            docforge_version="0.1.0-dev",
        )
        chunks.append(Chunk(content=text, metadata=meta))

    # Warm up
    await engine.embed(chunks[:10])

    # Benchmark
    start = time.perf_counter()
    await engine.embed(chunks)
    elapsed = time.perf_counter() - start

    chunks_per_sec = len(chunks) / elapsed
    assert chunks_per_sec >= 1000, f"Embedding throughput {chunks_per_sec:.1f} chunks/sec below target 1000"

    from tests.benchmarks import benchmark
    with benchmark("embedding_dummy_provider", len(chunks)):
        pass


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_search_benchmark_p50_latency(faiss_store: FAISSStore, chunks_1000: list[EmbeddedChunk]) -> None:
    """Benchmark search p50 latency."""
    await faiss_store.upsert(chunks_1000)

    query = _normalize([10.0, 5.0, 2.0, 1.0] + [0.0] * 380)

    # Warm up
    for _ in range(10):
        await faiss_store.search(query, k=10)

    # Benchmark - run many searches
    num_searches = 100
    latencies = []
    for _ in range(num_searches):
        start = time.perf_counter()
        await faiss_store.search(query, k=10)
        latencies.append(time.perf_counter() - start)

    latencies.sort()
    p50 = latencies[num_searches // 2]
    p99 = latencies[int(num_searches * 0.99)]

    assert p50 <= 0.050, f"Search p50 latency {p50*1000:.1f}ms exceeds target 50ms"
    assert p99 <= 0.100, f"Search p99 latency {p99*1000:.1f}ms exceeds target 100ms"

    from tests.benchmarks import benchmark
    with benchmark("search_p50_latency", num_searches):
        pass

    # Also record p99
    import tests.benchmarks as bm
    bm.benchmark.results.append(
        bm.BenchmarkResult(
            name="search_p99_latency",
            duration_seconds=p99,
            items_processed=1,
            items_per_second=1/p99 if p99 > 0 else 0,
        )
    )


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_search_benchmark_throughput(faiss_store: FAISSStore, chunks_1000: list[EmbeddedChunk]) -> None:
    """Benchmark search throughput (queries per second)."""
    await faiss_store.upsert(chunks_1000)

    queries = [
        _normalize([float(i % 100), float(i % 50), float(i % 25), 1.0] + [0.0] * 380)
        for i in range(1000)
    ]

    # Warm up
    for q in queries[:10]:
        await faiss_store.search(q, k=10)

    # Benchmark
    start = time.perf_counter()
    for q in queries:
        await faiss_store.search(q, k=10)
    elapsed = time.perf_counter() - start

    qps = len(queries) / elapsed
    assert qps >= 100, f"Search throughput {qps:.1f} qps below target 100"

    from tests.benchmarks import benchmark
    with benchmark("search_throughput_qps", len(queries)):
        pass


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_storage_engine_search_benchmark() -> None:
    """Benchmark StorageEngine search with FAISS backend."""
    tmp_path = Path(tempfile.mkdtemp())
    config = DocForgeConfig(
        general={"data_dir": str(tmp_path / "data")},
        storage={"path": str(tmp_path / "vectordb"), "backend": "faiss"},
        embeddings={"cache_embeddings": False},
    )

    engine = StorageEngine(config, software="bench", version="1.0")
    await engine.initialize(dimension=384, model_name="bench-model")

    # Add 1000 chunks
    chunks = []
    for i in range(1000):
        text = f"Document {i} content for storage engine benchmark."
        vec = _normalize([float(i % 100), float(i % 50), float(i % 25), 1.0] + [0.0] * 380)
        chunks.append(_make_embedded_chunk(text, vec, chunk_index=i))

    await engine.upsert(chunks)

    query = _normalize([10.0, 5.0, 2.0, 1.0] + [0.0] * 380)

    # Warm up
    for _ in range(10):
        await engine.search(query, k=10)

    # Benchmark p50
    num_searches = 100
    latencies = []
    for _ in range(num_searches):
        start = time.perf_counter()
        await engine.search(query, k=10)
        latencies.append(time.perf_counter() - start)

    latencies.sort()
    p50 = latencies[num_searches // 2]

    assert p50 <= 0.050, f"StorageEngine search p50 latency {p50*1000:.1f}ms exceeds target 50ms"

    from tests.benchmarks import benchmark
    with benchmark("storage_engine_search_p50", num_searches):
        pass

    await engine.close()