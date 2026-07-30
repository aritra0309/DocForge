"""Storage orchestrator — resolves the correct backend from config and delegates."""

from __future__ import annotations

import hashlib
from typing import Any

from docforge.core.config import DocForgeConfig
from docforge.core.interfaces import VectorStore
from docforge.core.models import EmbeddedChunk, SearchResult
from docforge.storage.backends.chromadb import ChromaDBStore
from docforge.storage.backends.faiss import FAISSStore
from docforge.storage.backends.lancedb import LanceDBStore
from docforge.storage.backends.qdrant import QdrantStore
from docforge.storage.backends.weaviate import WeaviateStore
from docforge.storage.metadata_store import MetadataStore

BACKENDS: dict[str, type[VectorStore]] = {
    "chromadb": ChromaDBStore,
    "faiss": FAISSStore,
    "lancedb": LanceDBStore,
    "qdrant": QdrantStore,
    "weaviate": WeaviateStore,
}


def _collection_name(software: str, version: str, model_name: str) -> str:
    raw = model_name.lower().replace("/", "_").replace("-", "_").replace(".", "_")
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"docforge_{software}_{version}_{short_hash}"


class StorageEngine:
    """Resolves the vector store backend from config and delegates all calls.

    Usage::

        engine = StorageEngine(config)
        await engine.initialize()
        await engine.upsert(embedded_chunks)
        results = await engine.search(query_vector, k=10, filters={...})
        await engine.close()
    """

    def __init__(
        self,
        config: DocForgeConfig,
        software: str = "",
        version: str = "",
    ) -> None:
        self._config = config
        self._software = software
        self._version = version
        self._store: VectorStore | None = None
        self._metadata_store: MetadataStore | None = None
        self._initialized = False

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            raise RuntimeError("StorageEngine not initialized")
        return self._store

    @property
    def metadata_store(self) -> MetadataStore:
        if self._metadata_store is None:
            raise RuntimeError("StorageEngine not initialized")
        return self._metadata_store

    async def initialize(
        self,
        software: str | None = None,
        version: str | None = None,
        dimension: int = 768,
        model_name: str = "",
    ) -> None:
        if software:
            self._software = software
        if version:
            self._version = version

        store_cfg = self._config.storage
        backend_name = store_cfg.backend

        backend_cls = BACKENDS.get(backend_name)
        if backend_cls is None:
            msg = f"Unknown storage backend: {backend_name!r}. Available: {list(BACKENDS)}"
            raise ValueError(msg)

        coll_name = _collection_name(self._software, self._version, model_name or "default")

        self._store = backend_cls()
        await self._store.initialize(
            {
                "path": str(store_cfg.path),
                "collection_name": coll_name,
                "dimension": dimension,
            }
        )

        meta_db_path = store_cfg.path / "metadata.db"
        self._metadata_store = MetadataStore(meta_db_path)
        self._initialized = True

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        self._ensure_initialized()
        await self.store.upsert(chunks)

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self._ensure_initialized()
        return await self.store.search(query_vector, k, filters)

    async def delete(self, filters: dict[str, Any]) -> None:
        self._ensure_initialized()
        await self.store.delete(filters)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        self._ensure_initialized()
        return await self.store.count(filters)

    async def close(self) -> None:
        if self._store is not None:
            await self._store.close()
        if self._metadata_store is not None:
            self._metadata_store.close()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "StorageEngine not initialized. Call await engine.initialize() first."
            )


__all__ = ["BACKENDS", "StorageEngine", "_collection_name"]
