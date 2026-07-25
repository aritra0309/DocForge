"""Abstract base classes defining the plugin contract for every pipeline stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from docforge.core.models import (
    Chunk,
    ClassifiedPage,
    DiscoveryResult,
    EmbeddedChunk,
    ExtractedPage,
    FetchResult,
    SearchResult,
)


class DiscoveryProvider(ABC):
    """Discover a software project's documentation source from a name string.

    Implementations check the curated registry first, then fall back to
    heuristic URL probing for unknown software.
    """

    @abstractmethod
    async def discover(self, name: str) -> DiscoveryResult:
        """Given a software name, return its documentation discovery result.

        Args:
            name: Canonical software identifier (e.g. 'postgresql').

        Returns:
            A DiscoveryResult with the base URL, available versions, and
            content selector hints.

        Raises:
            DiscoveryError: If the software cannot be found via any strategy.
        """


class CrawlFetcher(ABC):
    """Fetch individual documentation pages over HTTP.

    Implementations handle rate limiting, retries with exponential backoff,
    conditional requests (ETag / Last-Modified), and response caching.
    """

    @abstractmethod
    async def fetch(self, url: str) -> FetchResult:
        """Fetch a single URL and return the raw response.

        Args:
            url: The canonical URL of the documentation page.

        Returns:
            A FetchResult containing the HTTP status, HTML body, and headers.

        Raises:
            FetchError: If the URL cannot be fetched after all retries.
        """


class ContentExtractor(ABC):
    """Convert raw HTML documentation pages into clean structured Markdown.

    Implementations strip navigation, headers, footers, and ads, then
    convert the main content area to Markdown while preserving code blocks,
    tables, callouts, and link integrity.
    """

    @abstractmethod
    async def extract(self, fetch_result: FetchResult) -> ExtractedPage:
        """Extract structured Markdown content from a fetched HTML page.

        Args:
            fetch_result: The raw fetch result containing HTML.

        Returns:
            An ExtractedPage with clean Markdown, extracted headings, code
            blocks, and breadcrumb information.
        """


class PageClassifier(ABC):
    """Classify a documentation page into a semantic PageType.

    Implementations use a combination of URL path patterns, title keywords,
    heading analysis, and code-to-text ratio to determine the page type.
    """

    @abstractmethod
    def classify(self, page: ExtractedPage) -> ClassifiedPage:
        """Assign a semantic page type to an extracted page.

        Args:
            page: The extracted documentation page.

        Returns:
            A ClassifiedPage that inherits all ExtractedPage fields plus
            the assigned page_type and confidence score.
        """


class ChunkingStrategy(ABC):
    """Split a classified documentation page into retrieval-sized chunks.

    Each strategy is type-aware — API reference pages, tutorials, and
    generic documentation each use different boundary detection logic.
    """

    @abstractmethod
    def chunk(self, page: ClassifiedPage) -> list[Chunk]:
        """Split a classified page into one or more chunks.

        Args:
            page: The classified documentation page.

        Returns:
            A list of Chunks, each with content and full metadata. No chunk
            exceeds the configured max_chunk_size in tokens.
        """


class EmbeddingProvider(ABC):
    """Generate dense vector embeddings for chunks of text.

    Implementations wrap local models (e.g. sentence-transformers) or
    remote API providers (e.g. OpenAI, Voyage) behind a uniform interface.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model (e.g. 'BAAI/bge-base-en-v1.5')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum number of tokens the model accepts per input."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vector representations.

        Args:
            texts: List of text strings to embed (in order).

        Returns:
            A list of embedding vectors, one per input text, where each
            vector is a list of floats of length ``dimension``.
        """


class VectorStore(ABC):
    """Persist embedded chunks and enable semantic similarity search.

    Implementations wrap vector database backends (ChromaDB, FAISS, Qdrant,
    LanceDB, Weaviate) behind a uniform interface.
    """

    @abstractmethod
    async def initialize(self, config: dict[str, Any]) -> None:
        """Set up the vector store connection or create the database.

        Args:
            config: Backend-specific configuration dictionary (e.g. path,
                host, port, collection name).
        """

    @abstractmethod
    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or update chunks in the vector store.

        Upsert is idempotent — inserting the same chunk twice does not
        create duplicates. Chunks are matched by their chunk_id.

        Args:
            chunks: List of embedded chunks with vectors and metadata.
        """

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the top-k most similar chunks to a query vector.

        Args:
            query_vector: The query embedding vector.
            k: Number of results to return (default 10).
            filters: Optional metadata filters to narrow the search
                (e.g. {'software': 'postgresql', 'version': '17'}).

        Returns:
            A list of SearchResults ranked by similarity (highest score first).
        """

    @abstractmethod
    async def delete(self, filters: dict[str, Any]) -> None:
        """Delete all chunks matching the given metadata filters.

        Args:
            filters: Metadata key-value pairs identifying chunks to remove.
        """

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Return the number of chunks matching the given filters.

        Args:
            filters: Optional metadata filters. If None, count all chunks.

        Returns:
            Total count of matching chunks.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the vector store (connections, file handles)."""
