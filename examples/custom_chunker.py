"""Minimal custom ChunkingStrategy that chunks a page by paragraphs."""

import hashlib
from datetime import UTC, datetime
from typing import override

from docforge.core.interfaces import ChunkingStrategy
from docforge.core.models import (
    Chunk,
    ChunkMetadata,
    ClassifiedPage,
    generate_chunk_id,
    generate_page_id,
)


class ParagraphChunker(ChunkingStrategy):
    """Create one chunk for each non-empty Markdown paragraph."""

    @override
    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        paragraphs = [part.strip() for part in page.markdown.split("\n\n") if part.strip()]
        page_id = generate_page_id(page.url, page.title)
        timestamp = datetime.now(UTC)
        total = len(paragraphs)

        return [
            Chunk(
                content=paragraph,
                metadata=ChunkMetadata(
                    chunk_id=generate_chunk_id("example", "1", page.url, "", index),
                    parent_page_id=page_id,
                    software="example",
                    version="1",
                    url=page.url,
                    title=page.title,
                    page_type=page.page_type,
                    breadcrumb=page.breadcrumb,
                    section_heading="",
                    chunk_index=index,
                    total_chunks=total,
                    has_code=False,
                    code_languages=[],
                    content_hash=hashlib.sha256(paragraph.encode()).hexdigest(),
                    crawl_timestamp=timestamp,
                    embedding_model="not-embedded",
                    embedding_dimension=0,
                    docforge_version="0.1.0",
                ),
            )
            for index, paragraph in enumerate(paragraphs)
        ]


# Pass ParagraphChunker to your pipeline integration where a ChunkingStrategy is accepted.
