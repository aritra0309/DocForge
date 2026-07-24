"""TutorialChunker — one chunk per tutorial step."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from docforge.chunker.strategies.base import (
    BaseChunkingStrategy,
    merge_small_chunks,
)
from docforge.core.models import Chunk, ChunkMetadata, ClassifiedPage

_STEP_PAT = re.compile(r"^(#{1,3}\s+)?(step\s+\d+|step\s+|exercise\s+\d+)", re.IGNORECASE)
_STEP_NUM_PAT = re.compile(r"^\d+[\.\)]\s+")


class TutorialChunker(BaseChunkingStrategy):
    """Splits tutorial pages into one chunk per step."""

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
    ) -> None:
        super().__init__(target_chunk_size, max_chunk_size, min_chunk_size)

    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        sections = self._split_at_step_boundaries(page.markdown, page)
        texts = merge_small_chunks(
            [s["content"] for s in sections], self.min_chunk_size
        )
        return self._to_chunks(page, texts, sections)

    @staticmethod
    def _split_at_step_boundaries(markdown: str, page: ClassifiedPage) -> list[dict]:
        lines = markdown.split("\n")
        sections: list[dict] = []
        current_lines: list[str] = []
        current_heading = ""
        in_code = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code = not in_code
                current_lines.append(line)
                continue

            if in_code:
                current_lines.append(line)
                continue

            is_step = bool(_STEP_PAT.match(stripped) or _STEP_NUM_PAT.match(stripped))

            if is_step and current_lines:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines),
                })
                current_heading = stripped.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_lines),
            })

        if not sections and markdown.strip():
            sections.append({
                "heading": page.title,
                "content": markdown,
            })

        return sections

    def _to_chunks(
        self, page: ClassifiedPage, texts: list[str], sections: list[dict]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        total = len(texts)
        for i, text in enumerate(texts):
            section_heading = sections[i].get("heading", "") if i < len(sections) else ""
            has_code, code_langs = self._detect_code(text)
            chunks.append(Chunk(
                content=text,
                metadata=ChunkMetadata(
                    chunk_id="",
                    parent_page_id="",
                    software="",
                    version="",
                    url=page.url,
                    title=page.title,
                    page_type=page.page_type,
                    breadcrumb=page.breadcrumb,
                    section_heading=section_heading,
                    chunk_index=i,
                    total_chunks=total,
                    has_code=has_code,
                    code_languages=code_langs,
                    content_hash="",
                    crawl_timestamp=datetime.now(UTC),
                    embedding_model="",
                    embedding_dimension=0,
                    docforge_version="",
                ),
            ))
        return chunks

    @staticmethod
    def _detect_code(text: str) -> tuple[bool, list[str]]:
        languages: list[str] = []
        lines = text.split("\n")
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                if in_code:
                    lang = line[3:].strip()
                    if lang and lang not in languages:
                        languages.append(lang)
        return (len(languages) > 0, languages)


__all__ = ["TutorialChunker"]
