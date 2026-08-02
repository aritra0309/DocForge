"""Minimal deterministic EmbeddingProvider for tests and prototypes."""

import hashlib

from docforge.core.interfaces import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Map text to stable eight-dimensional vectors; not for production retrieval."""

    @property
    def model_name(self) -> str:
        return "example-hash-v1"

    @property
    def dimension(self) -> int:
        return 8

    @property
    def max_tokens(self) -> int:
        return 8192

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [
            [byte / 255 for byte in hashlib.sha256(text.encode()).digest()[: self.dimension]]
            for text in texts
        ]


# Pass HashEmbeddingProvider() to EmbeddingEngine(provider=...) in custom integrations.
