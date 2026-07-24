"""Metadata generator — attaches rich metadata to every chunk."""

from __future__ import annotations

from datetime import UTC, datetime

from docforge.core.models import (
    Chunk,
    ChunkMetadata,
    ClassifiedPage,
    generate_chunk_id,
    generate_page_id,
)
from docforge.metadata.hasher import compute_content_hash


class MetadataGenerator:
    """Assembles fully-populated ``ChunkMetadata`` for every chunk.

    Takes the partial chunks produced by a chunking strategy (which contain
    section_heading, chunk_index, total_chunks, has_code, code_languages)
    and fills in all remaining fields from pipeline context and computed values.
    """

    def __init__(
        self,
        software: str,
        version: str,
        embedding_model: str = "",
        embedding_dimension: int = 0,
        docforge_version: str = "0.1.0-dev",
    ) -> None:
        self.software = software
        self.version = version
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.docforge_version = docforge_version

    def generate(self, chunks: list[Chunk], page: ClassifiedPage) -> list[Chunk]:
        """Populate metadata for all chunks from a single page.

        Args:
            chunks: Partial chunks from a chunking strategy (metadata fields
                like chunk_id, parent_page_id, software, etc. may be empty).
            page: The classified page these chunks were produced from.

        Returns:
            New Chunk objects with fully-populated ChunkMetadata.
        """
        parent_page_id = self._compute_page_id(page)
        now = datetime.now(UTC)

        return [self._enrich(chunk, page, parent_page_id, now, i) for i, chunk in enumerate(chunks)]

    @staticmethod
    def _compute_page_id(page: ClassifiedPage) -> str:
        return generate_page_id(page.url, page.title)

    def _enrich(
        self,
        chunk: Chunk,
        page: ClassifiedPage,
        parent_page_id: str,
        crawl_timestamp: datetime,
        index: int,
    ) -> Chunk:
        meta = chunk.metadata
        section_heading = meta.section_heading
        chunk_index = meta.chunk_index if meta.chunk_index >= 0 else index
        total_chunks = meta.total_chunks

        content_hash = compute_content_hash(chunk.content)
        chunk_id = self._compute_chunk_id(
            section_heading,
            chunk_index,
        )

        new_meta = ChunkMetadata(
            chunk_id=chunk_id,
            parent_page_id=parent_page_id,
            software=self.software,
            version=self.version,
            url=page.url,
            title=page.title,
            page_type=page.page_type,
            breadcrumb=page.breadcrumb,
            section_heading=section_heading,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            has_code=meta.has_code,
            code_languages=meta.code_languages,
            content_hash=content_hash,
            crawl_timestamp=crawl_timestamp,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            docforge_version=self.docforge_version,
        )
        return Chunk(content=chunk.content, metadata=new_meta)

    def _compute_chunk_id(
        self,
        section_heading: str,
        chunk_index: int,
    ) -> str:
        return generate_chunk_id(
            software=self.software,
            version=self.version,
            # Use url + section_heading as canonical identifier
            canonical_url="" if not section_heading else section_heading,
            section_heading=section_heading,
            chunk_index=chunk_index,
        )


__all__ = ["MetadataGenerator"]
