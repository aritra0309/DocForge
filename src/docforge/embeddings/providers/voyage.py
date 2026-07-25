"""Voyage AI embedding provider.

Supports ``voyage-3`` and ``voyage-code-3``.
Reads the API key from the ``VOYAGE_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any

from docforge.core.interfaces import EmbeddingProvider

_MODEL_INFO: dict[str, tuple[int, int]] = {
    "voyage-3": (1024, 32000),
    "voyage-3-lite": (512, 32000),
    "voyage-code-3": (1024, 32000),
    "voyage-2": (1024, 4000),
    "voyage-code-2": (1536, 4000),
}


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the Voyage AI Embeddings API.

    Requires ``VOYAGE_API_KEY`` to be set in the environment or passed
    explicitly.
    """

    def __init__(
        self,
        model_name: str = "voyage-3",
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return _MODEL_INFO.get(self._model_name, (1024, 32000))[0]

    @property
    def max_tokens(self) -> int:
        return _MODEL_INFO.get(self._model_name, (1024, 32000))[1]

    def _get_client(self) -> Any:
        if self._client is None:
            import voyageai

            self._client = voyageai.AsyncClient(api_key=self._api_key)
        return self._client

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = await client.embed(
            texts=texts,
            model=self._model_name,
            input_type="document",
        )
        result: list[list[float]] = response.embeddings
        return result


__all__ = ["VoyageEmbeddingProvider"]
