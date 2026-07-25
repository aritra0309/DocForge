"""OpenAI embedding provider.

Supports ``text-embedding-3-small`` (512d) and ``text-embedding-3-large`` (256…3072d).
Reads the API key from the ``OPENAI_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any

from docforge.core.interfaces import EmbeddingProvider

_MODEL_INFO: dict[str, tuple[int, int]] = {
    "text-embedding-3-small": (512, 8191),
    "text-embedding-3-large": (3072, 8191),
    "text-embedding-ada-002": (1536, 8191),
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the OpenAI Embeddings API.

    Requires ``OPENAI_API_KEY`` to be set in the environment or passed
    explicitly.

    The default model is ``text-embedding-3-small``. Use ``dimensions``
    to reduce vector size (only for ``text-embedding-3-*`` models).
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._dimensions = dimensions
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        return _MODEL_INFO.get(self._model_name, (1536, 8191))[0]

    @property
    def max_tokens(self) -> int:
        return _MODEL_INFO.get(self._model_name, (1536, 8191))[1]

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self._model_name, "input": texts}
        if self._dimensions is not None and "3-" in self._model_name:
            kwargs["dimensions"] = self._dimensions
        response = await client.embeddings.create(**kwargs)
        response.data.sort(key=lambda d: d.index)
        return [d.embedding for d in response.data]


__all__ = ["OpenAIEmbeddingProvider"]
