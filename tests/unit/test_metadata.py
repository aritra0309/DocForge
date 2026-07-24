"""Unit tests for the metadata generator, hasher, and breadcrumb extractors."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from docforge.core.models import (
    Chunk,
    ChunkMetadata,
    ClassifiedPage,
    PageType,
)
from docforge.metadata.breadcrumbs import (
    extract_breadcrumb_from_html,
    extract_breadcrumb_from_url,
)
from docforge.metadata.generator import MetadataGenerator
from docforge.metadata.hasher import compute_content_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_page() -> ClassifiedPage:
    return ClassifiedPage(
        url="https://www.postgresql.org/docs/17/tutorial-intro.html",
        title="Introduction",
        markdown="# Introduction\n\nThis is a tutorial about PostgreSQL.",
        headings=["Introduction"],
        code_blocks=[{"language": "sql", "content": "SELECT 1;"}],
        breadcrumb=["Docs", "17", "Tutorial", "Introduction"],
        raw_metadata={},
        page_type=PageType.TUTORIAL,
        confidence=0.92,
    )


@pytest.fixture
def partial_chunks() -> list[Chunk]:
    """Chunks as produced by a chunking strategy (metadata partially filled)."""
    return [
        Chunk(
            content="# Introduction\n\nThis is the first section.",
            metadata=ChunkMetadata(
                chunk_id="",
                parent_page_id="",
                software="",
                version="",
                url="",
                title="",
                page_type=PageType.TUTORIAL,
                section_heading="Introduction",
                chunk_index=0,
                total_chunks=2,
                has_code=False,
                code_languages=[],
                content_hash="",
                crawl_timestamp=datetime.now(UTC),
                embedding_model="",
                embedding_dimension=0,
                docforge_version="",
            ),
        ),
        Chunk(
            content="# Setup\n\n```sql\nCREATE TABLE test;\n```\n\nThis is the setup section.",
            metadata=ChunkMetadata(
                chunk_id="",
                parent_page_id="",
                software="",
                version="",
                url="",
                title="",
                page_type=PageType.TUTORIAL,
                section_heading="Setup",
                chunk_index=1,
                total_chunks=2,
                has_code=True,
                code_languages=["sql"],
                content_hash="",
                crawl_timestamp=datetime.now(UTC),
                embedding_model="",
                embedding_dimension=0,
                docforge_version="",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Hasher tests
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_deterministic(self) -> None:
        text = "SELECT * FROM users;"
        h1 = compute_content_hash(text)
        h2 = compute_content_hash(text)
        assert h1 == h2

    def test_whitespace_insensitive(self) -> None:
        normal = "hello world"
        extra_space = "  hello   world  "
        spaced = "hello\n\nworld"
        assert compute_content_hash(normal) == compute_content_hash(extra_space)
        assert compute_content_hash(normal) == compute_content_hash(spaced)

    def test_case_insensitive(self) -> None:
        upper = "HELLO WORLD"
        lower = "hello world"
        assert compute_content_hash(upper) == compute_content_hash(lower)

    def test_different_content_different_hash(self) -> None:
        text1 = "apple banana"
        text2 = "apple cherry"
        assert compute_content_hash(text1) != compute_content_hash(text2)

    def test_produces_64_char_hex(self) -> None:
        h = compute_content_hash("some text")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_string(self) -> None:
        h = compute_content_hash("")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# MetadataGenerator tests
# ---------------------------------------------------------------------------


class TestMetadataGenerator:
    def test_populates_all_fields(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(
            software="postgresql",
            version="17",
            embedding_model="BAAI/bge-base-en-v1.5",
            embedding_dimension=768,
            docforge_version="0.1.0-dev",
        )
        result = generator.generate(partial_chunks, sample_page)

        assert len(result) == 2
        for chunk in result:
            meta = chunk.metadata
            assert meta.software == "postgresql"
            assert meta.version == "17"
            assert meta.url == sample_page.url
            assert meta.title == sample_page.title
            assert meta.page_type == PageType.TUTORIAL
            assert meta.breadcrumb == sample_page.breadcrumb
            assert meta.embedding_model == "BAAI/bge-base-en-v1.5"
            assert meta.embedding_dimension == 768
            assert meta.docforge_version == "0.1.0-dev"
            assert meta.content_hash
            assert len(meta.content_hash) == 64
            assert isinstance(meta.crawl_timestamp, datetime)

    def test_chunk_id_is_deterministic(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result1 = generator.generate(partial_chunks, sample_page)
        result2 = generator.generate(partial_chunks, sample_page)

        for c1, c2 in zip(result1, result2, strict=True):
            assert c1.metadata.chunk_id == c2.metadata.chunk_id

    def test_parent_page_id_is_deterministic(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result = generator.generate(partial_chunks, sample_page)

        for chunk in result:
            assert chunk.metadata.parent_page_id
            assert len(chunk.metadata.parent_page_id) == 64

    def test_different_chunks_have_different_ids(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result = generator.generate(partial_chunks, sample_page)

        assert result[0].metadata.chunk_id != result[1].metadata.chunk_id

    def test_content_hash_reflects_chunk_content(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result = generator.generate(partial_chunks, sample_page)

        for chunk in result:
            expected = compute_content_hash(chunk.content)
            assert chunk.metadata.content_hash == expected

    def test_code_metadata_preserved(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result = generator.generate(partial_chunks, sample_page)

        # Second chunk has code
        assert result[1].metadata.has_code is True
        assert "sql" in result[1].metadata.code_languages

    def test_section_heading_preserved(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result = generator.generate(partial_chunks, sample_page)

        assert result[0].metadata.section_heading == "Introduction"
        assert result[1].metadata.section_heading == "Setup"

    def test_chunk_counts_preserved(
        self, sample_page: ClassifiedPage, partial_chunks: list[Chunk]
    ) -> None:
        generator = MetadataGenerator(software="postgresql", version="17")
        result = generator.generate(partial_chunks, sample_page)

        assert result[0].metadata.chunk_index == 0
        assert result[1].metadata.chunk_index == 1
        assert result[0].metadata.total_chunks == 2


# ---------------------------------------------------------------------------
# Breadcrumb extraction tests
# ---------------------------------------------------------------------------


class TestExtractBreadcrumbFromUrl:
    def test_simple_path(self) -> None:
        url = "https://example.com/docs/17/tutorial/intro"
        result = extract_breadcrumb_from_url(url)
        assert result == ["docs", "17", "tutorial", "intro"]

    def test_with_file_extension(self) -> None:
        url = "https://example.com/docs/17/tutorial.html"
        result = extract_breadcrumb_from_url(url)
        assert result == ["docs", "17", "tutorial"]

    def test_root_url(self) -> None:
        url = "https://example.com/"
        result = extract_breadcrumb_from_url(url)
        assert result == []

    def test_no_path(self) -> None:
        url = "https://example.com"
        result = extract_breadcrumb_from_url(url)
        assert result == []

    def test_trailing_slash(self) -> None:
        url = "https://example.com/docs/api/"
        result = extract_breadcrumb_from_url(url)
        assert result == ["docs", "api"]


class TestExtractBreadcrumbFromHtml:
    def test_breadcrumb_class(self) -> None:
        html = """
        <html><body>
        <nav class="breadcrumb">
            <li class="breadcrumb-item"><a href="/">Home</a></li>
            <li class="breadcrumb-item"><a href="/docs/">Docs</a></li>
            <li class="breadcrumb-item active">Current</li>
        </nav>
        </body></html>
        """
        result = extract_breadcrumb_from_html(html)
        assert "Home" in result
        assert "Docs" in result
        assert "Current" in result

    def test_aria_label_breadcrumb(self) -> None:
        html = """
        <html><body>
        <nav aria-label="Breadcrumb">
            <a href="/">Home</a>
            <a href="/docs/">Docs</a>
            <span>Current</span>
        </nav>
        </body></html>
        """
        result = extract_breadcrumb_from_html(html)
        assert len(result) >= 1

    def test_empty_html(self) -> None:
        result = extract_breadcrumb_from_html("<html></html>")
        assert result == []

    def test_invalid_html(self) -> None:
        result = extract_breadcrumb_from_html("")
        assert result == []

    def test_with_selector(self) -> None:
        html = """
        <html><body>
        <div id="crumbs">
            <span>Home</span> / <span>Docs</span> / <span>Install</span>
        </div>
        </body></html>
        """
        result = extract_breadcrumb_from_html(html, selectors={"breadcrumb": "#crumbs"})
        assert len(result) >= 1


class TestExtractBreadcrumbFromHtmlEdgeCases:
    def test_breadcrumbs_class(self) -> None:
        html = """
        <html><body>
        <div class="breadcrumbs">
            <a href="/">Home</a>
            <a href="/guide/">Guide</a>
        </div>
        </body></html>
        """
        result = extract_breadcrumb_from_html(html)
        assert len(result) >= 1

    def test_ol_breadcrumb(self) -> None:
        html = """
        <html><body>
        <ol class="breadcrumb">
            <li><a href="/">Home</a></li>
            <li><a href="/api/">API</a></li>
        </ol>
        </body></html>
        """
        result = extract_breadcrumb_from_html(html)
        assert "Home" in result
        assert "API" in result
