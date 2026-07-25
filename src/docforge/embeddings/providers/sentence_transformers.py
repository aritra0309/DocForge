"""Sentence Transformers embedding provider.

Runs locally via ``sentence-transformers`` — no API key required.
Default model is ``BAAI/bge-base-en-v1.5`` (768-dimensional).
"""

from __future__ import annotations

from typing import Any

from docforge.core.interfaces import EmbeddingProvider

_MODEL_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}

_MODEL_MAX_TOKENS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 512,
    "BAAI/bge-base-en-v1.5": 512,
    "BAAI/bge-large-en-v1.5": 512,
    "all-MiniLM-L6-v2": 256,
    "all-mpnet-base-v2": 384,
}


class SentenceTransformersProvider(EmbeddingProvider):
    """Local embedding provider using ``sentence-transformers``.

    The underlying model is lazy-loaded on first call to avoid import-time
    overhead and memory usage.

    .. note::
        The model is loaded into CPU memory. For GPU support, set the
        ``torch_device`` argument to ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        torch_device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._torch_device = torch_device
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return _MODEL_DIMS.get(self._model_name, 768)

    @property
    def max_tokens(self) -> int:
        return _MODEL_MAX_TOKENS.get(self._model_name, 512)

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, device=self._torch_device
            )
        return self._model

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        result: list[list[float]] = embeddings.tolist()
        return result


__all__ = ["SentenceTransformersProvider"]
