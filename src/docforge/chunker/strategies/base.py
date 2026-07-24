"""Token counting and shared chunking utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken

from docforge.core.interfaces import ChunkingStrategy
from docforge.core.models import Chunk, ClassifiedPage

_ENCODING = "cl100k_base"


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING)


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def split_by_tokens(
    text: str, max_tokens: int, separator: str = "\n\n"
) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    parts = text.split(separator)
    chunks: list[str] = []
    current: list[str] = []

    for part in parts:
        candidate = separator.join([*current, part])
        if count_tokens(candidate) <= max_tokens:
            current.append(part)
        else:
            if current:
                chunks.append(separator.join(current))
            current = [part]

    if current:
        chunks.append(separator.join(current))

    return chunks


def merge_small_chunks(
    chunks: list[str], min_tokens: int
) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if count_tokens(merged[-1]) < min_tokens:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


class BaseChunkingStrategy(ChunkingStrategy, ABC):
    """Base class for all chunking strategies with shared helpers."""

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
    ) -> None:
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        ...


__all__ = [
    "BaseChunkingStrategy",
    "count_tokens",
    "merge_small_chunks",
    "split_by_tokens",
]
