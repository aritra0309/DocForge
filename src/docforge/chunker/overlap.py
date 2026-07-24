"""Overlap injection — adds token overlap between adjacent chunks."""

from __future__ import annotations

import tiktoken

_ENCODING = "cl100k_base"


def apply_overlap(texts: list[str], overlap_tokens: int) -> list[str]:
    if not texts or overlap_tokens <= 0:
        return texts[:]

    result: list[str] = [texts[0]]
    for i in range(1, len(texts)):
        prev = texts[i - 1]
        overlap_text = _extract_tail(prev, overlap_tokens)
        if overlap_text:
            result.append(overlap_text + "\n\n" + texts[i])
        else:
            result.append(texts[i])
    return result


def _extract_tail(text: str, n_tokens: int) -> str:
    enc = tiktoken.get_encoding(_ENCODING)
    tokens = enc.encode(text)
    if len(tokens) <= n_tokens:
        return text
    tail_tokens = tokens[-n_tokens:]
    return enc.decode(tail_tokens)


__all__ = ["apply_overlap"]
