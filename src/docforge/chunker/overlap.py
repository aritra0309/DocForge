"""Overlap injection — adds token overlap between adjacent chunks."""

from __future__ import annotations

import re

from docforge.chunker.strategies.base import _get_encoder


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
    encoder = _get_encoder()
    if encoder is not None:
        tokens = encoder.encode(text)
        if len(tokens) <= n_tokens:
            return text
        return encoder.decode(tokens[-n_tokens:])

    words = re.findall(r"\w+|[^\w\s]", text)
    return text if len(words) <= n_tokens else " ".join(words[-n_tokens:])


__all__ = ["apply_overlap"]
