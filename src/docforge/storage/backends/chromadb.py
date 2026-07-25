"""ChromaDB vector store backend — embedded, zero-config, default backend."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docforge.core.interfaces import VectorStore
from docforge.core.models import ChunkMetadata, EmbeddedChunk, SearchResult


class ChromaDBStore(VectorStore):
    """ChromaDB-backed vector store using the embedded (in-process) client.

    Persists data to the path specified in config. Collection names follow
    the pattern ``docforge_{software}_{version}_{model_hash[:8]}``.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collection: Any = None
        self._collection_name: str = ""

    async def initialize(self, config: dict[str, Any]) -> None:
        import chromadb
        from chromadb.config import Settings

        db_path = config.get("path", str(Path.home() / ".docforge" / "vectordb"))
        self._collection_name = config.get("collection_name", "docforge_default")

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        ids = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for chunk in chunks:
            ids.append(chunk.metadata.chunk_id)
            embeddings.append(chunk.vector)
            documents.append(chunk.content)
            metadatas.append(self._chunk_metadata_to_dict(chunk.metadata))

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        where = self._metadata_filters_to_where(filters) if filters else None

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
        )

        return self._format_results(results)

    async def delete(self, filters: dict[str, Any]) -> None:
        where = self._metadata_filters_to_where(filters)
        if where:
            self._collection.delete(where=where)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        if filters:
            where = self._metadata_filters_to_where(filters)
            result = self._collection.get(where=where)
            filtered_count: int = len(result["ids"]) if result and result.get("ids") else 0
            return filtered_count
        total_count: int = self._collection.count()
        return total_count

    async def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_metadata_to_dict(meta: ChunkMetadata) -> dict[str, Any]:
        return {
            "parent_page_id": meta.parent_page_id,
            "software": meta.software,
            "version": meta.version,
            "url": meta.url,
            "title": meta.title,
            "page_type": meta.page_type.value,
            "section_heading": meta.section_heading,
            "chunk_index": meta.chunk_index,
            "total_chunks": meta.total_chunks,
            "has_code": int(meta.has_code),
            "content_hash": meta.content_hash,
            "embedding_model": meta.embedding_model,
            "embedding_dimension": meta.embedding_dimension,
        }

    @staticmethod
    def _metadata_filters_to_where(filters: dict[str, Any]) -> dict[str, Any] | None:
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: value}
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _format_results(self, raw: Any) -> list[SearchResult]:
        results: list[SearchResult] = []
        if not raw or not raw["ids"] or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        distances = raw["distances"][0] if raw.get("distances") else [0.0] * len(ids)
        documents = raw["documents"][0] if raw.get("documents") else [""] * len(ids)
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)

        for i in range(len(ids)):
            meta_dict = metadatas[i] if i < len(metadatas) else {}
            meta = ChunkMetadata(
                chunk_id=ids[i],
                parent_page_id=meta_dict.get("parent_page_id", ""),
                software=meta_dict.get("software", ""),
                version=meta_dict.get("version", ""),
                url=meta_dict.get("url", ""),
                title=meta_dict.get("title", ""),
                page_type=meta_dict.get("page_type", "unknown"),
                section_heading=meta_dict.get("section_heading", ""),
                chunk_index=int(meta_dict.get("chunk_index", 0)),
                total_chunks=int(meta_dict.get("total_chunks", 0)),
                has_code=bool(meta_dict.get("has_code", False)),
                code_languages=[],
                content_hash=meta_dict.get("content_hash", ""),
                crawl_timestamp=self._parse_timestamp(meta_dict.get("crawl_timestamp", "")),
                embedding_model=meta_dict.get("embedding_model", ""),
                embedding_dimension=int(meta_dict.get("embedding_dimension", 0)),
                breadcrumb=[],
                docforge_version="",
            )
            results.append(
                SearchResult(
                    chunk_id=ids[i],
                    content=documents[i],
                    metadata=meta,
                    score=1.0 - distances[i],
                )
            )

        return results


__all__ = ["ChromaDBStore"]
