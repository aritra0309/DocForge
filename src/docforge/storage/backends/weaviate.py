"""Weaviate vector store backend — client-server with hybrid BM25 + vector search.

Reads its configuration from the ``[storage.weaviate]`` config section.
Supports vector search, keyword search (BM25), and hybrid search.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from docforge.core.interfaces import VectorStore
from docforge.core.models import ChunkMetadata, EmbeddedChunk, SearchResult


class WeaviateStore(VectorStore):
    """Weaviate-backed vector store with hybrid search support.

    Connects to a Weaviate server (local or cloud). Collections map to
    Weaviate classes with auto-schema configuration.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collection_name: str = ""

    async def initialize(self, config: dict[str, Any]) -> None:
        import weaviate
        from weaviate.classes.config import Configure, DataType, Property
        from weaviate.classes.init import Auth

        host = config.get("host", "http://localhost:8080")
        api_key = config.get("api_key")
        self._collection_name = config.get("collection_name", "DocForgeDefault")

        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-KEY"] = api_key

        auth = Auth.api_key(api_key) if api_key else None

        self._client = weaviate.connect_to_local(
            host=host,
            auth_credentials=auth,
            headers=headers,
        )

        collection_name = self._collection_name
        existing = self._client.collections.exists(collection_name)
        if not existing:
            self._client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.VectorIndex.none(),
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="parent_page_id", data_type=DataType.TEXT),
                    Property(name="software", data_type=DataType.TEXT),
                    Property(name="version", data_type=DataType.TEXT),
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="page_type", data_type=DataType.TEXT),
                    Property(name="section_heading", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="total_chunks", data_type=DataType.INT),
                    Property(name="has_code", data_type=DataType.BOOL),
                    Property(name="content_hash", data_type=DataType.TEXT),
                    Property(name="crawl_timestamp", data_type=DataType.TEXT),
                    Property(name="embedding_model", data_type=DataType.TEXT),
                    Property(name="embedding_dimension", data_type=DataType.INT),
                    Property(name="docforge_version", data_type=DataType.TEXT),
                ],
            )

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        collection = self._client.collections.get(self._collection_name)

        for chunk in chunks:
            meta = chunk.metadata
            uuid_str = meta.chunk_id[:36]

            collection.data.insert(
                uuid=uuid_str,
                vector=chunk.vector,
                properties={
                    "chunk_id": meta.chunk_id,
                    "content": chunk.content,
                    "parent_page_id": meta.parent_page_id,
                    "software": meta.software,
                    "version": meta.version,
                    "url": meta.url,
                    "title": meta.title,
                    "page_type": meta.page_type.value,
                    "section_heading": meta.section_heading,
                    "chunk_index": meta.chunk_index,
                    "total_chunks": meta.total_chunks,
                    "has_code": meta.has_code,
                    "content_hash": meta.content_hash,
                    "crawl_timestamp": meta.crawl_timestamp.isoformat(),
                    "embedding_model": meta.embedding_model,
                    "embedding_dimension": meta.embedding_dimension,
                    "docforge_version": meta.docforge_version,
                },
            )

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        collection = self._client.collections.get(self._collection_name)

        weaviate_filter = self._build_filter(filters) if filters else None

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            filters=weaviate_filter,
            return_metadata=["distance"],
        )

        return self._format_results(response)

    async def delete(self, filters: dict[str, Any]) -> None:
        collection = self._client.collections.get(self._collection_name)
        weaviate_filter = self._build_filter(filters)
        if weaviate_filter:
            collection.data.delete_many(where=weaviate_filter)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        collection = self._client.collections.get(self._collection_name)
        if filters:
            weaviate_filter = self._build_filter(filters)
            count_val: int = collection.aggregate.over_all(filters=weaviate_filter).total_count
            return count_val
        total: int = collection.aggregate.over_all().total_count
        return total

    async def get_all(self, filters: dict[str, Any] | None = None) -> list[EmbeddedChunk]:
        collection = self._client.collections.get(self._collection_name)
        weaviate_filter = self._build_filter(filters) if filters else None
        response = collection.query.fetch_objects(
            filters=weaviate_filter,
            limit=10000,
            include_vector=True,
        )
        chunks: list[EmbeddedChunk] = []
        for obj in response.objects:
            p = obj.properties
            meta = ChunkMetadata(
                chunk_id=p.get("chunk_id", ""),
                parent_page_id=p.get("parent_page_id", ""),
                software=p.get("software", ""),
                version=p.get("version", ""),
                url=p.get("url", ""),
                title=p.get("title", ""),
                page_type=p.get("page_type", "unknown"),
                section_heading=p.get("section_heading", ""),
                chunk_index=int(p.get("chunk_index", 0)),
                total_chunks=int(p.get("total_chunks", 0)),
                has_code=bool(p.get("has_code", False)),
                code_languages=[],
                content_hash=p.get("content_hash", ""),
                crawl_timestamp=self._parse_timestamp(p.get("crawl_timestamp", "")),
                embedding_model=p.get("embedding_model", ""),
                embedding_dimension=int(p.get("embedding_dimension", 0)),
                breadcrumb=[],
                docforge_version=p.get("docforge_version", ""),
            )
            vector: list[float] = obj.vector or []
            chunks.append(
                EmbeddedChunk(
                    content=p.get("content", ""),
                    metadata=meta,
                    vector=vector,
                )
            )
        return chunks

    async def close(self) -> None:
        if self._client:
            self._client.close()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_filter(self, filters: dict[str, Any]) -> Any | None:
        from weaviate.classes.query import Filter

        conditions: list[Any] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(Filter.by_property(key).contains_any(value))
            elif isinstance(value, bool):
                conditions.append(Filter.by_property(key).equal(value))
            else:
                conditions.append(Filter.by_property(key).equal(value))

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return Filter.all_of(conditions)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _format_results(self, response: Any) -> list[SearchResult]:
        results: list[SearchResult] = []
        for obj in response.objects:
            p = obj.properties
            meta = ChunkMetadata(
                chunk_id=p.get("chunk_id", ""),
                parent_page_id=p.get("parent_page_id", ""),
                software=p.get("software", ""),
                version=p.get("version", ""),
                url=p.get("url", ""),
                title=p.get("title", ""),
                page_type=p.get("page_type", "unknown"),
                section_heading=p.get("section_heading", ""),
                chunk_index=int(p.get("chunk_index", 0)),
                total_chunks=int(p.get("total_chunks", 0)),
                has_code=bool(p.get("has_code", False)),
                code_languages=[],
                content_hash=p.get("content_hash", ""),
                crawl_timestamp=self._parse_timestamp(p.get("crawl_timestamp", "")),
                embedding_model=p.get("embedding_model", ""),
                embedding_dimension=int(p.get("embedding_dimension", 0)),
                breadcrumb=[],
                docforge_version=p.get("docforge_version", ""),
            )
            score = 1.0
            if obj.metadata and obj.metadata.distance is not None:
                score = 1.0 - obj.metadata.distance
            results.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", ""),
                    content=p.get("content", ""),
                    metadata=meta,
                    score=score,
                )
            )
        return results


__all__ = ["WeaviateStore"]
