"""Unit tests for core data models and ID generation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from docforge.core.models import (
    Chunk,
    ChunkMetadata,
    ClassifiedPage,
    DiscoveryResult,
    EmbeddedChunk,
    ExtractedPage,
    FetchResult,
    PageType,
    SearchResult,
    generate_chunk_id,
    generate_page_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_discovery() -> DiscoveryResult:
    return DiscoveryResult(
        software="postgresql",
        display_name="PostgreSQL",
        base_url="https://www.postgresql.org/docs/",
        versions=["17", "16", "15"],
        latest_version="17",
        sitemap_url="https://www.postgresql.org/sitemap.xml",
        content_selectors={"main_content": "#docContent"},
        url_filters={"include": ["/docs/{version}/**"], "exclude": []},
    )


@pytest.fixture
def sample_fetch() -> FetchResult:
    return FetchResult(
        url="https://www.postgresql.org/docs/17/tutorial-intro.html",
        status_code=200,
        html="<html><body><h1>Introduction</h1><p>Hello</p></body></html>",
        headers={"content-type": "text/html"},
        etag='"abc123"',
        last_modified="Wed, 21 Jul 2026 10:00:00 GMT",
    )


@pytest.fixture
def sample_extracted() -> ExtractedPage:
    return ExtractedPage(
        url="https://www.postgresql.org/docs/17/tutorial-intro.html",
        title="Introduction",
        markdown="# Introduction\n\nThis is a tutorial.",
        headings=["Introduction"],
        code_blocks=[{"language": "sql", "content": "SELECT 1;"}],
        breadcrumb=["Docs", "17", "Tutorial"],
        raw_metadata={"og:title": "Introduction"},
    )


@pytest.fixture
def sample_classified(sample_extracted: ExtractedPage) -> ClassifiedPage:
    return ClassifiedPage(
        url=sample_extracted.url,
        title=sample_extracted.title,
        markdown=sample_extracted.markdown,
        headings=sample_extracted.headings,
        code_blocks=sample_extracted.code_blocks,
        breadcrumb=sample_extracted.breadcrumb,
        raw_metadata=sample_extracted.raw_metadata,
        page_type=PageType.TUTORIAL,
        confidence=0.92,
    )


@pytest.fixture
def sample_chunk_metadata() -> ChunkMetadata:
    now = datetime.now(UTC)
    return ChunkMetadata(
        chunk_id="a" * 64,
        parent_page_id="b" * 64,
        software="postgresql",
        version="17",
        url="https://www.postgresql.org/docs/17/tutorial-intro.html",
        title="Introduction",
        page_type=PageType.TUTORIAL,
        breadcrumb=["Docs", "17", "Tutorial"],
        section_heading="Introduction",
        chunk_index=0,
        total_chunks=3,
        has_code=True,
        code_languages=["sql"],
        content_hash="c" * 64,
        crawl_timestamp=now,
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        docforge_version="0.1.0-dev",
    )


@pytest.fixture
def sample_chunk(sample_chunk_metadata: ChunkMetadata) -> Chunk:
    return Chunk(
        content="# Introduction\n\nThis is a tutorial.",
        metadata=sample_chunk_metadata,
    )


@pytest.fixture
def sample_embedded(sample_chunk: Chunk) -> EmbeddedChunk:
    return EmbeddedChunk(
        content=sample_chunk.content,
        metadata=sample_chunk.metadata,
        vector=[0.1] * 768,
    )


@pytest.fixture
def sample_search(sample_chunk_metadata: ChunkMetadata) -> SearchResult:
    return SearchResult(
        chunk_id=sample_chunk_metadata.chunk_id,
        content="# Introduction\n\nThis is a tutorial.",
        metadata=sample_chunk_metadata,
        score=0.95,
    )


# ---------------------------------------------------------------------------
# PageType enum
# ---------------------------------------------------------------------------


class TestPageType:
    def test_member_values(self) -> None:
        assert PageType.TUTORIAL.value == "tutorial"
        assert PageType.API_REFERENCE.value == "api_reference"
        assert PageType.UNKNOWN.value == "unknown"

    def test_all_members_exist(self) -> None:
        expected = {
            "tutorial",
            "api_reference",
            "function_reference",
            "guide",
            "concepts",
            "examples",
            "release_notes",
            "faq",
            "configuration",
            "troubleshooting",
            "getting_started",
            "migration",
            "unknown",
        }
        assert {pt.value for pt in PageType} == expected


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


class TestDiscoveryResult:
    def test_instantiate(self, sample_discovery: DiscoveryResult) -> None:
        assert sample_discovery.software == "postgresql"
        assert sample_discovery.latest_version == "17"

    def test_defaults(self) -> None:
        dr = DiscoveryResult(
            software="x",
            display_name="X",
            base_url="https://x.com",
            versions=["1"],
            latest_version="1",
        )
        assert dr.sitemap_url is None
        assert dr.content_selectors == {}
        assert dr.url_filters == {}

    def test_frozen(self, sample_discovery: DiscoveryResult) -> None:
        with pytest.raises(Exception):
            sample_discovery.software = "mysql"  # type: ignore[misc]


class TestFetchResult:
    def test_instantiate(self, sample_fetch: FetchResult) -> None:
        assert sample_fetch.status_code == 200
        assert isinstance(sample_fetch.fetched_at, datetime)

    def test_defaults(self) -> None:
        fr = FetchResult(url="https://x.com", status_code=404, html="")
        assert fr.etag is None
        assert fr.last_modified is None
        assert fr.headers == {}


class TestExtractedPage:
    def test_instantiate(self, sample_extracted: ExtractedPage) -> None:
        assert sample_extracted.title == "Introduction"
        assert len(sample_extracted.headings) == 1


class TestClassifiedPage:
    def test_inherits_extracted(self, sample_classified: ClassifiedPage) -> None:
        assert isinstance(sample_classified, ExtractedPage)
        assert sample_classified.page_type == PageType.TUTORIAL
        assert sample_classified.confidence == 0.92


class TestChunkMetadata:
    def test_instantiate(self, sample_chunk_metadata: ChunkMetadata) -> None:
        assert sample_chunk_metadata.software == "postgresql"
        assert sample_chunk_metadata.embedding_dimension == 768


class TestChunk:
    def test_instantiate(self, sample_chunk: Chunk) -> None:
        assert "tutorial" in sample_chunk.content.lower() or "Introduction" in sample_chunk.content
        assert sample_chunk.metadata.chunk_index == 0


class TestEmbeddedChunk:
    def test_inherits_chunk(self, sample_embedded: EmbeddedChunk) -> None:
        assert isinstance(sample_embedded, Chunk)
        assert len(sample_embedded.vector) == 768


class TestSearchResult:
    def test_instantiate(self, sample_search: SearchResult) -> None:
        assert sample_search.score == 0.95


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_discovery_round_trip(self, sample_discovery: DiscoveryResult) -> None:
        data = sample_discovery.model_dump()
        restored = DiscoveryResult.model_validate(data)
        assert restored == sample_discovery

    def test_fetch_round_trip(self, sample_fetch: FetchResult) -> None:
        data = sample_fetch.model_dump()
        restored = FetchResult.model_validate(data)
        assert restored == sample_fetch

    def test_extracted_round_trip(self, sample_extracted: ExtractedPage) -> None:
        data = sample_extracted.model_dump()
        restored = ExtractedPage.model_validate(data)
        assert restored == sample_extracted

    def test_classified_round_trip(self, sample_classified: ClassifiedPage) -> None:
        data = sample_classified.model_dump()
        restored = ClassifiedPage.model_validate(data)
        assert restored == sample_classified

    def test_chunk_metadata_round_trip(self, sample_chunk_metadata: ChunkMetadata) -> None:
        data = sample_chunk_metadata.model_dump()
        restored = ChunkMetadata.model_validate(data)
        assert restored == sample_chunk_metadata

    def test_chunk_round_trip(self, sample_chunk: Chunk) -> None:
        data = sample_chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored == sample_chunk

    def test_embedded_round_trip(self, sample_embedded: EmbeddedChunk) -> None:
        data = sample_embedded.model_dump()
        restored = EmbeddedChunk.model_validate(data)
        assert restored == sample_embedded

    def test_search_round_trip(self, sample_search: SearchResult) -> None:
        data = sample_search.model_dump()
        restored = SearchResult.model_validate(data)
        assert restored == sample_search

    def test_json_round_trip(self, sample_chunk_metadata: ChunkMetadata) -> None:
        json_str = sample_chunk_metadata.model_dump_json()
        restored = ChunkMetadata.model_validate_json(json_str)
        assert restored == sample_chunk_metadata


# ---------------------------------------------------------------------------
# ID generation determinism
# ---------------------------------------------------------------------------


class TestChunkIdGeneration:
    def test_deterministic(self) -> None:
        args = ("postgresql", "17", "https://example.com/page", "Section", 0)
        id1 = generate_chunk_id(*args)
        id2 = generate_chunk_id(*args)
        assert id1 == id2

    def test_different_inputs_yield_different_ids(self) -> None:
        base = ("postgresql", "17", "https://example.com/page", "Section", 0)
        id1 = generate_chunk_id(*base)
        id2 = generate_chunk_id("mysql", *base[1:])
        id3 = generate_chunk_id(base[0], "16", *base[2:])
        id4 = generate_chunk_id(base[0], base[1], "https://other.com", *base[3:])
        id5 = generate_chunk_id(base[0], base[1], base[2], "Other", base[4])
        id6 = generate_chunk_id(*base[:4], 1)
        assert len({id1, id2, id3, id4, id5, id6}) == 6

    def test_produces_64_char_hex(self) -> None:
        cid = generate_chunk_id("pg", "17", "https://x.com", "H", 0)
        assert len(cid) == 64
        assert all(c in "0123456789abcdef" for c in cid)


class TestPageIdGeneration:
    def test_deterministic(self) -> None:
        args = ("https://example.com/page", "Title")
        id1 = generate_page_id(*args)
        id2 = generate_page_id(*args)
        assert id1 == id2

    def test_different_inputs_yield_different_ids(self) -> None:
        id1 = generate_page_id("https://a.com", "A")
        id2 = generate_page_id("https://b.com", "A")
        id3 = generate_page_id("https://a.com", "B")
        assert id1 != id2 != id3

    def test_produces_64_char_hex(self) -> None:
        pid = generate_page_id("https://x.com", "X")
        assert len(pid) == 64
