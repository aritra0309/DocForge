"""BGE embedding provider — wraps sentence-transformers with named presets.

Supports named presets:
- ``bge-small-en`` → ``BAAI/bge-small-en-v1.5`` (384d)
- ``bge-base-en``  → ``BAAI/bge-base-en-v1.5`` (768d) — default
- ``bge-large-en`` → ``BAAI/bge-large-en-v1.5`` (1024d)
"""

from __future__ import annotations

from typing import Any

from docforge.core.interfaces import EmbeddingProvider

_PRESETS: dict[str, str] = {
    "bge-small-en": "BAAI/bge-small-en-v1.5",
    "bge-base-en": "BAAI/bge-base-en-v1.5",
    "bge-large-en": "BAAI/bge-large-en-v1.5",
}

_MODEL_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

_MODEL_MAX_TOKENS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 512,
    "BAAI/bge-base-en-v1.5": 512,
    "BAAI/bge-large-en-v1.5": 512,
}


def _resolve_model(name: str) -> str:
    return _PRESETS.get(name, name)


class BgeProvider(EmbeddingProvider):
    """BGE embedding provider using ``sentence-transformers``.

    Accepts short preset names (``bge-small-en``, ``bge-base-en``,
    ``bge-large-en``) or any HuggingFace model identifier.

    The underlying model is lazy-loaded on first call.
    """

    def __init__(
        self,
        model_name: str = "bge-base-en",
        torch_device: str = "cpu",
    ) -> None:
        self._model_name = _resolve_model(model_name)
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

            self._model = SentenceTransformer(self._model_name, device=self._torch_device)
        return self._model

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        result: list[list[float]] = embeddings.tolist()
        return result


__all__ = ["BgeProvider"]
