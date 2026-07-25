"""Embedding orchestrator — batches chunks, caches vectors, manages rate limits."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from docforge.core.interfaces import EmbeddingProvider
from docforge.core.models import Chunk, EmbeddedChunk
from docforge.embeddings.cache import EmbeddingCache


@dataclass
class EmbeddingProgress:
    """Progress information emitted after each batch is processed."""

    batch_index: int
    total_batches: int
    chunks_processed: int
    total_chunks: int
    cache_hits: int
    batch_time_ms: float


class TokenBucket:
    """Simple token bucket rate limiter for API providers."""

    def __init__(self, rate: float, burst: int | None = None) -> None:
        self._rate = rate
        self._burst = burst or int(rate)
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
        await asyncio.sleep(wait)
        async with self._lock:
            self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        elapsed = time.monotonic() - self._last_refill
        self._tokens = min(float(self._burst), self._tokens + elapsed * self._rate)
        self._last_refill = time.monotonic()


class EmbeddingEngine:
    """Orchestrates chunk embedding with batching, caching, and rate limiting.

    Usage::

        engine = EmbeddingEngine(provider, cache=my_cache, batch_size=64)
        embedded = await engine.embed(chunks)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
        progress_callback: Callable[[EmbeddingProgress], Any] | None = None,
        api_rate_limit: float | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._progress_callback = progress_callback
        self._rate_limiter: TokenBucket | None = (
            TokenBucket(rate=api_rate_limit) if api_rate_limit else None
        )

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    async def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks and return ``EmbeddedChunk`` objects.

        Chunks must have their ``metadata.content_hash`` populated
        (typically by ``MetadataGenerator``).
        """
        if not chunks:
            return []

        model_name = self._provider.model_name
        total = len(chunks)
        hits = 0
        cached: dict[int, list[float]] = {}
        uncached: list[int] = []
        uncached_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            content_hash = chunk.metadata.content_hash
            if self._cache and content_hash:
                vector = self._cache.get(model_name, content_hash)
                if vector is not None:
                    cached[i] = vector
                    hits += 1
                    continue
            uncached.append(i)
            uncached_texts.append(chunk.content)

        batches = [
            uncached_texts[j : j + self._batch_size]
            for j in range(0, len(uncached_texts), self._batch_size)
        ]
        total_batches = len(batches)
        new_vecs: dict[int, list[float]] = {}

        for batch_idx, batch_texts in enumerate(batches):
            t0 = time.monotonic()
            vectors = await self._embed_with_retries(batch_texts)

            if self._rate_limiter:
                await self._rate_limiter.acquire()

            elapsed = (time.monotonic() - t0) * 1000

            start = batch_idx * self._batch_size
            for offset, vector in enumerate(vectors):
                idx = uncached[start + offset]
                new_vecs[idx] = vector
                chk = chunks[idx]
                ch_hash = chk.metadata.content_hash
                if self._cache and ch_hash:
                    self._cache.put(model_name, ch_hash, vector)

            processed = hits + len(new_vecs)
            if self._progress_callback:
                self._progress_callback(
                    EmbeddingProgress(
                        batch_index=batch_idx,
                        total_batches=total_batches,
                        chunks_processed=processed,
                        total_chunks=total,
                        cache_hits=hits,
                        batch_time_ms=elapsed,
                    )
                )

        results: list[EmbeddedChunk] = []
        for i, chunk in enumerate(chunks):
            vector = cached.get(i) or new_vecs.get(i)
            if vector is None:
                continue
            results.append(
                EmbeddedChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    vector=vector,
                )
            )

        return results

    async def _embed_with_retries(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._provider.embed_batch(texts)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    wait = 2**attempt * 0.5
                    await asyncio.sleep(wait)
        msg = f"Embedding failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    async def close(self) -> None:
        """Close the embedding cache if one was provided."""
        if self._cache:
            self._cache.close()


__all__ = ["EmbeddingEngine", "EmbeddingProgress", "TokenBucket"]
