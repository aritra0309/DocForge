"""HeadingChunker — splits on heading boundaries (H2 → H3 → paragraph)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from docforge.chunker.strategies.base import (
    BaseChunkingStrategy,
    count_tokens,
    merge_small_chunks,
    split_by_tokens,
)
from docforge.core.models import Chunk, ChunkMetadata, ClassifiedPage

_MAX_H1 = 1
_MAX_H2 = 2
_MAX_H3 = 3


class HeadingChunker(BaseChunkingStrategy):
    """Default chunker: splits on H2 boundaries, then H3, then paragraph."""

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
    ) -> None:
        super().__init__(target_chunk_size, max_chunk_size, min_chunk_size)

    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        sections = self._build_sections(page.markdown)
        sections = self._split_oversized(sections)
        texts = merge_small_chunks([s["content"] for s in sections], self.min_chunk_size)
        return self._to_chunks(page, texts, sections)

    def _build_sections(self, markdown: str) -> list[dict[str, Any]]:
        lines = markdown.split("\n")
        sections: list[dict[str, Any]] = []
        current_h1 = ""
        current_h2 = ""
        current_lines: list[str] = []

        for line in lines:
            heading_level, heading_text = self._parse_heading(line)

            if heading_level == _MAX_H1:
                if current_lines:
                    sections.append(self._make_section(current_h1, current_h2, current_lines))
                current_h1 = heading_text
                current_h2 = ""
                current_lines = []
            elif heading_level == _MAX_H2:
                if current_lines:
                    sections.append(self._make_section(current_h1, current_h2, current_lines))
                current_h2 = heading_text
                current_lines = []
            elif heading_level >= _MAX_H3:
                if current_lines:
                    sections.append(self._make_section(current_h1, current_h2, current_lines))
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(self._make_section(current_h1, current_h2, current_lines))

        if not sections and markdown.strip():
            sections.append(self._make_section("", "", markdown.split("\n")))

        return sections

    @staticmethod
    def _parse_heading(line: str) -> tuple[int, str]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            return (0, "")
        level = 0
        for ch in stripped:
            if ch == "#":
                level += 1
            else:
                break
        if level == 0 or len(stripped) <= level or stripped[level] != " ":
            return (0, "")
        text = stripped[level:].strip()
        return (level, text)

    @staticmethod
    def _make_section(h1: str, h2: str, lines: list[str]) -> dict[str, Any]:
        content = "\n".join(lines).strip()
        prefix_bits = []
        if h1:
            prefix_bits.append(f"# {h1}")
        if h2:
            prefix_bits.append(f"## {h2}")
        prefix = "\n".join(prefix_bits)
        if prefix:
            content = prefix + "\n\n" + content
        return {
            "h1": h1,
            "h2": h2,
            "content": content,
        }

    def _split_oversized(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for sec in sections:
            if count_tokens(sec["content"]) <= self.max_chunk_size:
                result.append(sec)
            else:
                sub_texts = split_by_tokens(sec["content"], self.target_chunk_size)
                for t in sub_texts:
                    result.append(
                        {
                            "h1": sec["h1"],
                            "h2": sec["h2"],
                            "content": t,
                        }
                    )
        return result

    def _to_chunks(
        self, page: ClassifiedPage, texts: list[str], sections: list[dict[str, Any]]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        total = len(texts)
        for i, text in enumerate(texts):
            section_heading = ""
            if i < len(sections):
                section_heading = sections[i].get("h2") or sections[i].get("h1") or ""
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


__all__ = ["HeadingChunker"]
