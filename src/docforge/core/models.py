"""Core data models that flow through every pipeline stage."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PageType(StrEnum):
    """Classification of documentation page types."""

    TUTORIAL = "tutorial"
    API_REFERENCE = "api_reference"
    FUNCTION_REFERENCE = "function_reference"
    GUIDE = "guide"
    CONCEPTS = "concepts"
    EXAMPLES = "examples"
    RELEASE_NOTES = "release_notes"
    FAQ = "faq"
    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    GETTING_STARTED = "getting_started"
    MIGRATION = "migration"
    UNKNOWN = "unknown"


class DiscoveryResult(BaseModel):
    """Result of discovering a software's documentation source."""

    model_config = {"frozen": True}

    software: str = Field(description="Canonical software identifier (e.g. 'postgresql')")
    display_name: str = Field(description="Human-readable name (e.g. 'PostgreSQL')")
    base_url: str = Field(description="Root URL of the documentation site")
    versions: list[str] = Field(description="Available documentation versions, newest first")
    latest_version: str = Field(description="Latest stable version string")
    sitemap_url: str | None = Field(default=None, description="URL of the sitemap XML if known")
    content_selectors: dict[str, str] = Field(
        default_factory=dict,
        description="CSS selectors for content regions (e.g. {'main_content': '#docContent'})",
    )
    url_filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Include/exclude URL patterns (e.g. {'include': ['/docs/**'], 'exclude': []})",
    )


class FetchResult(BaseModel):
    """Result of fetching a single documentation page."""

    model_config = {"frozen": True}

    url: str = Field(description="Canonical URL of the fetched page")
    status_code: int = Field(description="HTTP status code")
    html: str = Field(description="Raw HTML content of the page")
    headers: dict[str, str] = Field(default_factory=dict, description="Response headers")
    etag: str | None = Field(default=None, description="ETag header value for conditional requests")
    last_modified: str | None = Field(
        default=None, description="Last-Modified header value for conditional requests"
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the page was fetched",
    )


class ExtractedPage(BaseModel):
    """A documentation page after HTML-to-Markdown extraction."""

    model_config = {"frozen": True}

    url: str = Field(description="Canonical URL of the page")
    title: str = Field(description="Page title extracted from <h1> or <title>")
    markdown: str = Field(description="Clean Markdown content of the page body")
    headings: list[str] = Field(
        default_factory=list, description="All heading texts in document order"
    )
    code_blocks: list[dict[str, str]] = Field(
        default_factory=list,
        description="Extracted code blocks, each with 'language' and 'content' keys",
    )
    breadcrumb: list[str] = Field(default_factory=list, description="Navigation breadcrumb path")
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Original HTML metadata (OpenGraph, etc.)"
    )


class ClassifiedPage(ExtractedPage):
    """An extracted page with its semantic classification."""

    page_type: PageType = Field(description="Detected semantic page type")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification confidence score (0.0-1.0)",
    )


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk for retrieval filtering and provenance."""

    model_config = {"frozen": True}

    chunk_id: str = Field(description="Deterministic chunk identifier (SHA-256 based)")
    parent_page_id: str = Field(description="Hash-based identifier of the source page")
    software: str = Field(description="Software identifier (e.g. 'postgresql')")
    version: str = Field(description="Documentation version (e.g. '17')")
    url: str = Field(description="Source page URL")
    title: str = Field(description="Source page title")
    page_type: PageType = Field(description="Semantic type of the source page")
    breadcrumb: list[str] = Field(
        default_factory=list, description="Navigation breadcrumb at time of indexing"
    )
    section_heading: str = Field(
        default="", description="Heading of the section this chunk belongs to"
    )
    chunk_index: int = Field(description="Position of this chunk within its page (0-based)")
    total_chunks: int = Field(description="Total number of chunks from this page")
    has_code: bool = Field(description="Whether this chunk contains code blocks")
    code_languages: list[str] = Field(
        default_factory=list, description="Programming languages found in code blocks"
    )
    content_hash: str = Field(description="SHA-256 hash of normalised chunk content")
    crawl_timestamp: datetime = Field(description="When this page was crawled")
    embedding_model: str = Field(description="Name of the embedding model used")
    embedding_dimension: int = Field(description="Dimension of the embedding vector")
    docforge_version: str = Field(description="DocForge version that produced this chunk")


class Chunk(BaseModel):
    """A retrieval-sized unit of documentation content."""

    model_config = {"frozen": True}

    content: str = Field(description="Markdown content of this chunk")
    metadata: ChunkMetadata = Field(description="Associated metadata")


class EmbeddedChunk(Chunk):
    """A chunk with its dense vector representation."""

    vector: list[float] = Field(description="Dense embedding vector")


class SearchResult(BaseModel):
    """A single search result returned from the vector store."""

    model_config = {"frozen": True}

    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Markdown content of the matched chunk")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    score: float = Field(description="Similarity score (higher = more relevant)")


# ---------------------------------------------------------------------------
# ID generation helpers
# ---------------------------------------------------------------------------


def generate_chunk_id(
    software: str,
    version: str,
    canonical_url: str,
    section_heading: str,
    chunk_index: int,
) -> str:
    """Generate a deterministic chunk ID via SHA-256.

    The input is the pipe-delimited concatenation of:
        software | version | canonical_url | section_heading | chunk_index

    Returns the hex-encoded SHA-256 digest.
    """
    raw = f"{software}|{version}|{canonical_url}|{section_heading}|{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_page_id(url: str, title: str) -> str:
    """Generate a deterministic page identifier via SHA-256.

    The input is the pipe-delimited concatenation of url | title.
    """
    raw = f"{url}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
