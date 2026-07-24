"""CodeChunker — keeps code blocks with their surrounding explanation."""

from __future__ import annotations

from datetime import UTC, datetime

from docforge.chunker.strategies.base import (
    BaseChunkingStrategy,
    count_tokens,
    merge_small_chunks,
    split_by_tokens,
)
from docforge.core.models import Chunk, ChunkMetadata, ClassifiedPage


class CodeChunker(BaseChunkingStrategy):
    """Keeps each code block + its surrounding explanation as one chunk."""

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
    ) -> None:
        super().__init__(target_chunk_size, max_chunk_size, min_chunk_size)

    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        sections = self._split_at_code_boundaries(page.markdown, page)
        texts = merge_small_chunks([s["content"] for s in sections], self.min_chunk_size)
        return self._to_chunks(page, texts, sections)

    def _split_at_code_boundaries(self, markdown: str, page: ClassifiedPage) -> list[dict]:
        lines = markdown.split("\n")
        sections: list[dict] = []
        current_lines: list[str] = []
        in_code = False
        code_opened_at = 0

        for line in lines:
            if line.startswith("```"):
                if not in_code:
                    in_code = True
                    code_opened_at = len(current_lines)
                    current_lines.append(line)
                else:
                    current_lines.append(line)
                    in_code = False
                    sections.append(
                        {
                            "heading": "",
                            "content": "\n".join(current_lines[code_opened_at:]),
                        }
                    )
                    current_lines = current_lines[:code_opened_at]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "heading": "",
                    "content": "\n".join(current_lines),
                }
            )

        if not sections and markdown.strip():
            sections.append(
                {
                    "heading": page.title,
                    "content": markdown,
                }
            )

        result: list[dict] = []
        for sec in sections:
            if count_tokens(sec["content"]) <= self.max_chunk_size:
                result.append(sec)
            else:
                sub_texts = split_by_tokens(sec["content"], self.target_chunk_size)
                for t in sub_texts:
                    result.append(
                        {
                            "heading": sec["heading"],
                            "content": t,
                        }
                    )
        return result

    def _to_chunks(
        self, page: ClassifiedPage, texts: list[str], sections: list[dict]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        total = len(texts)
        for i, text in enumerate(texts):
            section_heading = sections[i].get("heading", "") if i < len(sections) else ""
            has_code, code_langs = self._detect_code(text)
            chunks.append(
                Chunk(
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
                )
            )
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


__all__ = ["CodeChunker"]
