"""Qdrant vector store backend — client-server production backend with full metadata filtering."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from docforge.core.interfaces import VectorStore
from docforge.core.models import ChunkMetadata, EmbeddedChunk, SearchResult


class QdrantStore(VectorStore):
    """Qdrant-backed vector store for production deployments.

    Connects to a Qdrant server (local or remote). Collections are created
    with cosine distance and full metadata filtering support.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collection_name: str = ""

    async def initialize(self, config: dict[str, Any]) -> None:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
        from qdrant_client.http import models  # type: ignore[import-not-found]

        host = config.get("host", "localhost")
        port = config.get("port", 6333)
        grpc_port = config.get("grpc_port", 6334)
        prefer_grpc = config.get("prefer_grpc", False)
        api_key = config.get("api_key")
        self._collection_name = config.get("collection_name", "docforge_default")
        dimension = config.get("dimension", 768)

        self._client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
        )

        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        from qdrant_client.http import models as m

        if not chunks:
            return

        points: list[m.PointStruct] = []
        for chunk in chunks:
            meta = chunk.metadata
            points.append(
                m.PointStruct(
                    id=meta.chunk_id,
                    vector=chunk.vector,
                    payload={
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
                        "code_languages": meta.code_languages,
                        "content_hash": meta.content_hash,
                        "crawl_timestamp": meta.crawl_timestamp.isoformat(),
                        "embedding_model": meta.embedding_model,
                        "embedding_dimension": meta.embedding_dimension,
                        "docforge_version": meta.docforge_version,
                    },
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:

        qdrant_filter = self._build_filter(filters) if filters else None

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return self._format_results(results)

    async def delete(self, filters: dict[str, Any]) -> None:
        from qdrant_client.http import models as m

        qdrant_filter = self._build_filter(filters)
        if qdrant_filter:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=m.FilterSelector(filter=qdrant_filter),
                wait=True,
            )

    async def count(self, filters: dict[str, Any] | None = None) -> int:

        qdrant_filter = self._build_filter(filters) if filters else None

        result = self._client.count(
            collection_name=self._collection_name,
            count_filter=qdrant_filter,
            exact=True,
        )
        count_val: int = result.count
        return count_val

    async def get_all(self, filters: dict[str, Any] | None = None) -> list[EmbeddedChunk]:
        from qdrant_client.http import models as m

        qdrant_filter = self._build_filter(filters) if filters else None
        scroll_result = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=qdrant_filter,
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )
        points, _ = scroll_result
        chunks: list[EmbeddedChunk] = []
        for point in points:
            p = point.payload or {}
            meta = ChunkMetadata(
                chunk_id=p.get("chunk_id", point.id),
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
                code_languages=p.get("code_languages", []),
                content_hash=p.get("content_hash", ""),
                crawl_timestamp=self._parse_timestamp(p.get("crawl_timestamp", "")),
                embedding_model=p.get("embedding_model", ""),
                embedding_dimension=int(p.get("embedding_dimension", 0)),
                breadcrumb=[],
                docforge_version=p.get("docforge_version", ""),
            )
            vector: list[float] = point.vector or []
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

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Any:
        from qdrant_client.http import models as m

        must_conditions: list[m.FieldCondition | m.IsEmptyCondition] = []
        for key, value in filters.items():
            if isinstance(value, list):
                must_conditions.append(
                    m.FieldCondition(
                        key=key,
                        match=m.MatchAny(any=value),
                    )
                )
            elif isinstance(value, bool):
                must_conditions.append(
                    m.FieldCondition(
                        key=key,
                        match=m.MatchValue(value=value),
                    )
                )
            else:
                must_conditions.append(
                    m.FieldCondition(
                        key=key,
                        match=m.MatchValue(value=value),
                    )
                )
        return m.Filter(must=must_conditions) if must_conditions else None

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _format_results(self, raw: list[Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for scored_point in raw:
            p = scored_point.payload or {}
            meta = ChunkMetadata(
                chunk_id=p.get("chunk_id", scored_point.id),
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
                code_languages=p.get("code_languages", []),
                content_hash=p.get("content_hash", ""),
                crawl_timestamp=self._parse_timestamp(p.get("crawl_timestamp", "")),
                embedding_model=p.get("embedding_model", ""),
                embedding_dimension=int(p.get("embedding_dimension", 0)),
                breadcrumb=[],
                docforge_version=p.get("docforge_version", ""),
            )
            results.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", scored_point.id),
                    content=p.get("content", ""),
                    metadata=meta,
                    score=scored_point.score,
                )
            )
        return results


__all__ = ["QdrantStore"]
