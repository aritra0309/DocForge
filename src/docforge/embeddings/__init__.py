"""Embedding layer — produce dense vector representations of every chunk."""

from docforge.embeddings.cache import EmbeddingCache
from docforge.embeddings.engine import EmbeddingEngine

__all__ = ["EmbeddingCache", "EmbeddingEngine"]
