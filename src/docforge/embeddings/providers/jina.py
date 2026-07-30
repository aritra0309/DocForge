"""Jina Embeddings v3 provider.

Reads the API key from the ``JINA_API_KEY`` environment variable.
Supports ``jina-embeddings-v3`` and other Jina embedding models.
"""

from __future__ import annotations

import os
from typing import Any

from docforge.core.interfaces import EmbeddingProvider

_MODEL_INFO: dict[str, tuple[int, int]] = {
    "jina-embeddings-v3": (1024, 8192),
    "jina-embeddings-v2-base-en": (768, 8192),
    "jina-embeddings-v2-small-en": (512, 8192),
}

_DEFAULT_MODEL = "jina-embeddings-v3"
_DEFAULT_DIM = 1024
_DEFAULT_MAX_TOKENS = 8192


class JinaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the Jina Embeddings API.

    Requires ``JINA_API_KEY`` to be set in the environment or passed
    explicitly. Uses the ``jina`` Python client library.

    The default model is ``jina-embeddings-v3`` (1024-dimensional).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return _MODEL_INFO.get(self._model_name, (_DEFAULT_DIM, _DEFAULT_MAX_TOKENS))[0]

    @property
    def max_tokens(self) -> int:
        return _MODEL_INFO.get(self._model_name, (_DEFAULT_DIM, _DEFAULT_MAX_TOKENS))[1]

    def _get_client(self) -> Any:
        if self._client is None:
            from jina import Clients

            base_url = os.environ.get("JINA_BASE_URL", "https://api.jina.ai")
            self._client = Clients(
                base_url=base_url,
                token=self._api_key,
            )
        return self._client

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = await self._run_embedding(client, texts)
        response.data.sort(key=lambda d: d.index)
        result: list[list[float]] = [d.embedding for d in response.data]
        return result

    async def _run_embedding(self, client: Any, texts: list[str]) -> Any:
        from jina import AsyncEmbeddings

        embeddings = AsyncEmbeddings(client=client)
        response = await embeddings.run(
            input=texts,
            model=self._model_name,
        )
        return response


__all__ = ["JinaEmbeddingProvider"]
