"""LanceDB vector store backend — columnar, file-based, no server required.

Good for large batch operations and analytics. Persists data to the
configured path using LanceDB's native columnar format.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy

from docforge.core.interfaces import VectorStore
from docforge.core.models import ChunkMetadata, EmbeddedChunk, SearchResult


class LanceDBStore(VectorStore):
    """LanceDB-backed vector store using the embedded (local) format.

    Uses LanceDB's columnar storage format for efficient batch operations
    and analytics. Collections map to LanceDB tables.
    """

    def __init__(self) -> None:
        self._db: Any = None
        self._table: Any = None
        self._collection_name: str = ""

    async def initialize(self, config: dict[str, Any]) -> None:
        import lancedb

        db_path = config.get("path", str(Path.home() / ".docforge" / "vectordb"))
        self._collection_name = config.get("collection_name", "docforge_default")
        dimension = config.get("dimension", 768)

        db_path_obj = Path(db_path)
        db_path_obj.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(db_path_obj))

        table_names = await self._db.table_names()
        if self._collection_name in table_names:
            self._table = await self._db.open_table(self._collection_name)
        else:
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), dimension)),
                    pa.field("chunk_id", pa.utf8()),
                    pa.field("content", pa.utf8()),
                    pa.field("metadata_json", pa.utf8()),
                ]
            )
            self._table = await self._db.create_table(
                self._collection_name,
                schema=schema,
                exist_ok=True,
            )

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        import pyarrow as pa

        existing = await self._table.search().limit(0).to_list()
        existing_ids = {row["chunk_id"] for row in existing}

        new_rows = []
        for chunk in chunks:
            cid = chunk.metadata.chunk_id
            if cid in existing_ids:
                await self._table.delete(f"chunk_id = '{cid}'")
            new_rows.append(
                {
                    "vector": chunk.vector,
                    "chunk_id": cid,
                    "content": chunk.content,
                    "metadata_json": self._metadata_to_json(chunk.metadata),
                }
            )

        if new_rows:
            await self._table.add(pa.Table.from_pylist(new_rows))

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query = numpy.array([query_vector], dtype=numpy.float32)

        search_builder = self._table.search(query[0]).limit(k)

        if filters:
            filter_expr = self._build_filter_expr(filters)
            if filter_expr:
                search_builder = search_builder.where(filter_expr)

        results = await search_builder.to_list()
        return self._format_results(results)

    async def delete(self, filters: dict[str, Any]) -> None:
        filter_expr = self._build_filter_expr(filters)
        if filter_expr:
            await self._table.delete(filter_expr)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        if filters:
            filter_expr = self._build_filter_expr(filters)
            if filter_expr:
                results = await self._table.search().where(filter_expr).limit(0).to_list()
                return len(results)
        count_val: int = await self._table.count_rows()
        return count_val

    async def get_all(self, filters: dict[str, Any] | None = None) -> list[EmbeddedChunk]:
        search_builder = self._table.search().limit(0)
        if filters:
            filter_expr = self._build_filter_expr(filters)
            if filter_expr:
                search_builder = search_builder.where(filter_expr)
        raw = await search_builder.to_list()
        chunks: list[EmbeddedChunk] = []
        for row in raw:
            meta_dict = json.loads(row.get("metadata_json", "{}"))
            meta = self._dict_to_metadata(meta_dict, row.get("chunk_id", ""))
            vector: list[float] = row.get("vector") or []
            chunks.append(
                EmbeddedChunk(
                    content=row.get("content", ""),
                    metadata=meta,
                    vector=vector,
                )
            )
        return chunks

    async def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_filter_expr(self, filters: dict[str, Any]) -> str:
        conditions: list[str] = []
        for key, value in filters.items():
            meta_key = "metadata_json"
            if isinstance(value, str):
                conditions.append(f"json_extract({meta_key}, '$.{key}') = '{value}'")
            elif isinstance(value, bool):
                bool_str = "true" if value else "false"
                conditions.append(f"json_extract({meta_key}, '$.{key}') = {bool_str}")
            elif isinstance(value, (int, float)):
                conditions.append(f"json_extract({meta_key}, '$.{key}') = {value}")
            elif isinstance(value, list):
                vals = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                conditions.append(f"json_extract({meta_key}, '$.{key}') IN ({', '.join(vals)})")
        return " AND ".join(conditions) if conditions else ""

    def _format_results(self, raw: list[dict[str, Any]]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for row in raw:
            meta_dict = json.loads(row.get("metadata_json", "{}"))
            meta = self._dict_to_metadata(meta_dict, row.get("chunk_id", ""))
            results.append(
                SearchResult(
                    chunk_id=meta.chunk_id,
                    content=row.get("content", ""),
                    metadata=meta,
                    score=float(1.0 - row.get("_distance", 0)),
                )
            )
        return results

    @staticmethod
    def _metadata_to_json(meta: ChunkMetadata) -> str:
        return json.dumps(
            {
                "parent_page_id": meta.parent_page_id,
                "software": meta.software,
                "version": meta.version,
                "url": meta.url,
                "title": meta.title,
                "page_type": meta.page_type.value,
                "breadcrumb": meta.breadcrumb,
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
        )

    @staticmethod
    def _dict_to_metadata(d: dict[str, Any], chunk_id: str) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_id=chunk_id,
            parent_page_id=d.get("parent_page_id", ""),
            software=d.get("software", ""),
            version=d.get("version", ""),
            url=d.get("url", ""),
            title=d.get("title", ""),
            page_type=d.get("page_type", "unknown"),
            breadcrumb=d.get("breadcrumb", []),
            section_heading=d.get("section_heading", ""),
            chunk_index=int(d.get("chunk_index", 0)),
            total_chunks=int(d.get("total_chunks", 0)),
            has_code=bool(d.get("has_code")),
            code_languages=d.get("code_languages", []),
            content_hash=d.get("content_hash", ""),
            crawl_timestamp=LanceDBStore._parse_timestamp(d.get("crawl_timestamp", "")),
            embedding_model=d.get("embedding_model", ""),
            embedding_dimension=int(d.get("embedding_dimension", 0)),
            docforge_version=d.get("docforge_version", ""),
        )

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return datetime(1970, 1, 1, tzinfo=UTC)


__all__ = ["LanceDBStore"]
