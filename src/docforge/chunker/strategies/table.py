"""TableChunker — handles tables as chunkable units."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from docforge.chunker.strategies.base import (
    BaseChunkingStrategy,
    count_tokens,
    merge_small_chunks,
)
from docforge.core.models import Chunk, ChunkMetadata, ClassifiedPage

_MIN_TABLE_LINES = 3


class TableChunker(BaseChunkingStrategy):
    """Splits pages with tables into table-centric chunks."""

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
        large_table_row_threshold: int = 20,
    ) -> None:
        super().__init__(target_chunk_size, max_chunk_size, min_chunk_size)
        self.large_table_row_threshold = large_table_row_threshold

    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        sections = self._parse_sections(page.markdown, page)
        texts = merge_small_chunks(
            [s["content"] for s in sections], self.min_chunk_size
        )
        return self._to_chunks(page, texts, sections)

    def _parse_sections(self, markdown: str, page: ClassifiedPage) -> list[dict]:
        lines = markdown.split("\n")
        sections: list[dict] = []
        current_lines: list[str] = []
        in_table = False
        table_lines: list[str] = []

        def flush_current() -> None:
            if current_lines:
                sections.append({
                    "heading": "",
                    "content": "\n".join(current_lines).strip(),
                })
                current_lines.clear()

        for line in lines:
            stripped = line.strip()
            is_table_row = "|" in stripped and (
                stripped.startswith("|") or stripped.startswith("+-")
            )
            is_separator = re.match(r"^[\|\s\-\+:]+$", stripped) if "|" in stripped else False

            if is_table_row and is_separator:
                pass
            elif is_table_row:
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table:
                    flush_current()
                    self._chunk_table(table_lines, sections)
                    table_lines.clear()
                    in_table = False
                current_lines.append(line)

        if in_table:
            flush_current()
            self._chunk_table(table_lines, sections)

        if current_lines:
            sections.append({
                "heading": "",
                "content": "\n".join(current_lines).strip(),
            })

        if not sections and markdown.strip():
            sections.append({
                "heading": page.title,
                "content": markdown,
            })

        return sections

    def _chunk_table(self, table_lines: list[str], sections: list[dict]) -> None:
        if len(table_lines) < _MIN_TABLE_LINES:
            sections.append({
                "heading": "",
                "content": "\n".join(table_lines),
            })
            return

        header = table_lines[0]
        separator = table_lines[1] if len(table_lines) > 1 else ""
        data_rows = table_lines[2:]

        full_table = "\n".join(table_lines)
        if count_tokens(full_table) <= self.max_chunk_size:
            sections.append({
                "heading": "",
                "content": full_table,
            })
            return

        header_tokens = max(count_tokens(header + separator + "|  |"), 1)
        rows_per_chunk = max(1, self.max_chunk_size // header_tokens)
        for start in range(0, len(data_rows), rows_per_chunk):
            chunk_rows = [header, separator, *data_rows[start:start + rows_per_chunk]]
            sections.append({
                "heading": "",
                "content": "\n".join(chunk_rows),
            })

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


__all__ = ["TableChunker"]
