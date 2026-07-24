"""Content hashing utilities for chunk deduplication and change detection."""

from __future__ import annotations

import hashlib
import re


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalised chunk content.

    Normalisation:
        1. Strip leading/trailing whitespace
        2. Collapse multiple whitespace chars (including newlines) to single space
        3. Lowercase

    This ensures the hash only changes when the *semantic content* changes,
    not when whitespace formatting or casing is adjusted.

    Args:
        text: The raw chunk content.

    Returns:
        Hex-encoded SHA-256 digest (64 characters).
    """
    normalised = _normalise(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _normalise(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text


__all__ = ["compute_content_hash"]
