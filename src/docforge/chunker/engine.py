"""Chunking orchestrator — selects strategy based on page type and dispatches."""

from __future__ import annotations

from docforge.chunker.overlap import apply_overlap
from docforge.chunker.strategies import (
    ApiRefChunker,
    CodeChunker,
    HeadingChunker,
    TutorialChunker,
)
from docforge.core.interfaces import ChunkingStrategy
from docforge.core.models import Chunk, ClassifiedPage, PageType


class ChunkingEngine:
    """Selects the chunking strategy based on ``ClassifiedPage.page_type``."""

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
        overlap_tokens: int = 64,
    ) -> None:
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
        self._strategies: dict[PageType, ChunkingStrategy] = {}

    def _get_strategy(self, page_type: PageType) -> ChunkingStrategy:
        if page_type in self._strategies:
            return self._strategies[page_type]

        if page_type in {
            PageType.API_REFERENCE,
            PageType.FUNCTION_REFERENCE,
            PageType.CONFIGURATION,
        }:
            strategy: ChunkingStrategy = ApiRefChunker(
                target_chunk_size=self.target_chunk_size,
                max_chunk_size=self.max_chunk_size,
                min_chunk_size=self.min_chunk_size,
            )
        elif page_type in {PageType.TUTORIAL, PageType.GETTING_STARTED}:
            strategy = TutorialChunker(
                target_chunk_size=self.target_chunk_size,
                max_chunk_size=self.max_chunk_size,
                min_chunk_size=self.min_chunk_size,
            )
        elif page_type == PageType.EXAMPLES:
            strategy = CodeChunker(
                target_chunk_size=self.target_chunk_size,
                max_chunk_size=self.max_chunk_size,
                min_chunk_size=self.min_chunk_size,
            )
        else:
            strategy = HeadingChunker(
                target_chunk_size=self.target_chunk_size,
                max_chunk_size=self.max_chunk_size,
                min_chunk_size=self.min_chunk_size,
            )

        self._strategies[page_type] = strategy
        return strategy

    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        """Chunk a classified page using the appropriate strategy."""
        strategy = self._get_strategy(page.page_type)
        chunks = strategy.chunk(page)
        if self.overlap_tokens > 0 and len(chunks) > 1:
            texts = [c.content for c in chunks]
            overlapped = apply_overlap(texts, self.overlap_tokens)
            chunks = [
                Chunk(content=t, metadata=c.metadata)
                for t, c in zip(overlapped, chunks, strict=True)
            ]
        return chunks


__all__ = ["ChunkingEngine"]
