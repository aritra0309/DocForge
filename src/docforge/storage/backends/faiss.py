"""FAISS vector store backend — high-performance local search with file-based persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy

from docforge.core.interfaces import VectorStore
from docforge.core.models import ChunkMetadata, EmbeddedChunk, SearchResult


class FAISSStore(VectorStore):
    """FAISS-backed vector store using ``IndexFlatIP`` (inner product).

    Since vectors are L2-normalised before indexing, inner product is
    equivalent to cosine similarity. Metadata is persisted as a JSON file
    alongside the FAISS index.
    """

    def __init__(self) -> None:
        self._index: Any = None
        self._dimension: int = 0
        self._metadata: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, str] = {}
        self._numeric_to_chunk: dict[int, str] = {}
        self._index_path: Path | None = None
        self._metadata_path: Path | None = None
        self._collection_name: str = ""

    async def initialize(self, config: dict[str, Any]) -> None:
        import faiss

        db_path = Path(config.get("path", str(Path.home() / ".docforge" / "vectordb")))
        self._collection_name = config.get("collection_name", "docforge_default")

        db_path.mkdir(parents=True, exist_ok=True)
        self._index_path = db_path / f"{self._collection_name}.faiss"
        self._metadata_path = db_path / f"{self._collection_name}.json"

        if self._index_path.exists() and self._metadata_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._dimension = self._index.d
            self._load_metadata()
        else:
            self._dimension = config.get("dimension", 768)
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dimension))
            self._metadata = {}
            self._documents = {}

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        import faiss

        if not chunks:
            return

        vectors = numpy.array([chunk.vector for chunk in chunks], dtype=numpy.float32)
        faiss.normalize_L2(vectors)

        new_ids: list[int] = []
        for chunk in chunks:
            meta = chunk.metadata
            chunk_id = meta.chunk_id
            if chunk_id in self._metadata:
                old_numeric = hash(chunk_id)
                id_selector = faiss.IDSelectorArray(numpy.array([old_numeric], dtype=numpy.int64))
                self._index.remove_ids(id_selector)
            numeric_id = hash(chunk_id)
            new_ids.append(numeric_id)

            self._metadata[chunk_id] = {
                "chunk_id": chunk_id,
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
            }
            self._documents[chunk_id] = chunk.content
            self._numeric_to_chunk[numeric_id] = chunk_id

        if not self._index.is_trained:
            self._index.train(vectors)

        self._index.add_with_ids(vectors, numpy.array(new_ids, dtype=numpy.int64))
        self._persist()

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        import faiss

        query = numpy.array([query_vector], dtype=numpy.float32)
        faiss.normalize_L2(query)

        actual_k = min(k, self._index.ntotal)
        if actual_k == 0:
            return []

        distances, indices = self._index.search(query, actual_k)

        results: list[SearchResult] = []
        id_to_meta = self._metadata
        id_to_doc = self._documents

        for i in range(actual_k):
            idx = indices[0][i]
            if idx == -1:
                continue
            chunk_id = self._numeric_to_chunk.get(idx)
            if chunk_id is None or chunk_id not in id_to_meta:
                continue

            if filters and not self._matches_filters(id_to_meta[chunk_id], filters):
                continue

            meta_dict = id_to_meta[chunk_id]
            meta = ChunkMetadata(
                chunk_id=meta_dict["chunk_id"],
                parent_page_id=meta_dict["parent_page_id"],
                software=meta_dict["software"],
                version=meta_dict["version"],
                url=meta_dict["url"],
                title=meta_dict["title"],
                page_type=meta_dict["page_type"],
                section_heading=meta_dict["section_heading"],
                chunk_index=meta_dict["chunk_index"],
                total_chunks=meta_dict["total_chunks"],
                has_code=meta_dict["has_code"],
                code_languages=meta_dict.get("code_languages", []),
                content_hash=meta_dict["content_hash"],
                crawl_timestamp=self._parse_timestamp(meta_dict["crawl_timestamp"]),
                embedding_model=meta_dict["embedding_model"],
                embedding_dimension=meta_dict["embedding_dimension"],
                docforge_version=meta_dict.get("docforge_version", ""),
            )
            results.append(
                SearchResult(
                    chunk_id=meta_dict["chunk_id"],
                    content=id_to_doc.get(chunk_id, ""),
                    metadata=meta,
                    score=float(distances[0][i]),
                )
            )

        return results

    async def delete(self, filters: dict[str, Any]) -> None:
        import faiss

        ids_to_remove: list[int] = []
        keys_to_remove: list[str] = []
        for chunk_id_str, meta in self._metadata.items():
            if self._matches_filters(meta, filters):
                numeric_id = hash(chunk_id_str)
                ids_to_remove.append(numeric_id)
                self._numeric_to_chunk.pop(numeric_id, None)
                keys_to_remove.append(chunk_id_str)
        for key in keys_to_remove:
            self._metadata.pop(key, None)
            self._documents.pop(key, None)

        if ids_to_remove:
            id_selector = faiss.IDSelectorArray(numpy.array(ids_to_remove, dtype=numpy.int64))
            self._index.remove_ids(id_selector)
            self._persist()

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        if filters:
            return sum(
                1 for meta in self._metadata.values() if self._matches_filters(meta, filters)
            )
        ntotal: int = self._index.ntotal if self._index else 0
        return ntotal

    async def close(self) -> None:
        self._persist()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if self._index_path and self._metadata_path:
            import faiss

            faiss.write_index(self._index, str(self._index_path))
            self._save_metadata()

    def _save_metadata(self) -> None:
        data = {
            "metadata": self._metadata,
            "documents": self._documents,
        }
        path = self._metadata_path
        if path is not None:
            path.write_text(json.dumps(data, indent=2))

    def _load_metadata(self) -> None:
        path = self._metadata_path
        if path is not None:
            data = json.loads(path.read_text())
        else:
            data = {}
        self._metadata = data.get("metadata", {})
        self._documents = data.get("documents", {})
        self._numeric_to_chunk = {}
        for cid in self._metadata:
            self._numeric_to_chunk[hash(cid)] = cid

    def _parse_timestamp(self, value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        if value:
            return datetime.fromisoformat(value)
        return datetime(1970, 1, 1, tzinfo=UTC)

    @staticmethod
    def _matches_filters(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, list):
                if meta[key] not in value:
                    return False
            elif meta[key] != value:
                return False
        return True


__all__ = ["FAISSStore"]
