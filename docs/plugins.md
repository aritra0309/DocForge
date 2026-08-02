# Plugins

DocForge contracts live in `docforge.core.interfaces`.

## Custom chunker

Implement `ChunkingStrategy.chunk(page) -> list[Chunk]`. Return chunks with complete `ChunkMetadata`; see [`examples/custom_chunker.py`](https://github.com/aritra0309/DocForge/blob/main/examples/custom_chunker.py).

## Custom embedding provider

Implement `EmbeddingProvider` properties `model_name`, `dimension`, `max_tokens`, plus async `embed_batch(texts)`. Register provider with `EmbeddingEngine`; see [`examples/custom_embedding_provider.py`](https://github.com/aritra0309/DocForge/blob/main/examples/custom_embedding_provider.py).

## Vector backend

Implement `VectorStore`: `initialize`, `upsert`, `search`, `delete`, `count`, and `close`. Backends must preserve chunk metadata and return highest-score search results first.

## Guidelines

- Keep public methods fully typed and documented.
- Keep `upsert` idempotent by `chunk_id`.
- Match `EmbeddingProvider.dimension` to vector length.
- Add unit tests, then document any credentials or optional extras.
