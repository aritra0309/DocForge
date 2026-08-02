"""Integration tests for the CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from tests.fixtures.site import FIXTURE_SITE_BASE as BASE, mock_fixture_site
from typer.testing import CliRunner

from docforge.cli.main import app
from docforge.core.config import DocForgeConfig
from docforge.core.models import DiscoveryResult
from docforge.embeddings.providers.base import EmbeddingProvider

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"


class FakeEmbeddingProvider(EmbeddingProvider):
    model_name = "test-cli-model"
    dimension = 8
    max_tokens = 512

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.05 + (i * 0.01) for _ in range(self.dimension)] for i in range(len(texts))]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_config(tmp_path: Path) -> DocForgeConfig:
    return DocForgeConfig(
        general={"data_dir": str(tmp_path / "data"), "log_level": "WARNING", "parallelism": 2},
        storage={"path": str(tmp_path / "vectordb"), "backend": "faiss"},
        embeddings={"cache_embeddings": False},
        crawler={"max_pages_per_version": 10, "rate_limit_rps": 100},
        chunker={"target_chunk_size": 512, "max_chunk_size": 1024},
    )


@pytest.fixture
def mock_discovery() -> AsyncMock:
    mock = AsyncMock()
    mock.discover.return_value = DiscoveryResult(
        software="fixture",
        display_name="Fixture Test",
        base_url=f"{BASE}/docs/",
        versions=["1.0"],
        latest_version="1.0",
        url_filters={"include": ["/docs/**"]},
    )
    return mock


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _mock_html_pages() -> None:
    mock_fixture_site(index_paths=("/docs/1.0",))


class TestCLIConfig:
    def test_config_command_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "Show current configuration" in result.output

    def test_config_command_outputs_config(
        self, runner: CliRunner, mock_config: DocForgeConfig
    ) -> None:
        with patch("docforge.cli.main.load_config", return_value=mock_config):
            result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "general" in result.output
        assert "crawler" in result.output
        assert "storage" in result.output


class TestCLIIndex:
    @pytest.mark.asyncio
    @respx.mock
    async def test_index_command_full_mode(
        self,
        runner: CliRunner,
        mock_config: DocForgeConfig,
        mock_discovery: AsyncMock,
        provider: FakeEmbeddingProvider,
    ) -> None:
        _mock_html_pages()

        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.Pipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.software = "fixture"
                mock_result.total_duration_ms = 1000
                mock_result.versions = [
                    MagicMock(
                        version="1.0",
                        status="completed",
                        error=None,
                        discovery=MagicMock(pages_processed=1, duration_ms=100),
                        crawl=MagicMock(pages_processed=3, chunks_produced=0, duration_ms=200),
                        extraction=MagicMock(pages_processed=3, duration_ms=150),
                        classification=MagicMock(pages_processed=3, duration_ms=50),
                        chunking=MagicMock(chunks_produced=5, duration_ms=100),
                        metadata=MagicMock(chunks_produced=5, duration_ms=50),
                        embedding=MagicMock(chunks_produced=5, duration_ms=300),
                        storage=MagicMock(chunks_produced=5, duration_ms=100),
                        total_duration_ms=1000,
                    )
                ]
                mock_pipeline.run.return_value = mock_result
                mock_pipeline_class.return_value = mock_pipeline

                result = runner.invoke(app, ["index", "fixture"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Indexing fixture" in result.output or "completed" in result.output


class TestCLISearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_search_command(
        self, runner: CliRunner, mock_config: DocForgeConfig, provider: FakeEmbeddingProvider
    ) -> None:
        from docforge.core.models import ChunkMetadata, PageType, SearchResult

        mock_search_result = SearchResult(
            chunk_id="abc",
            content="# Test\n\nTest content about creating indexes.",
            metadata=ChunkMetadata(
                chunk_id="abc",
                parent_page_id="page1",
                software="fixture",
                version="1.0",
                url="https://example.com/docs/page1",
                title="Test Page",
                page_type=PageType.GUIDE,
                section_heading="Test",
                chunk_index=0,
                total_chunks=1,
                has_code=False,
                code_languages=[],
                content_hash="hash1",
                crawl_timestamp=__import__("datetime").datetime(2025, 1, 1),
                embedding_model="test-model",
                embedding_dimension=8,
                docforge_version="0.1.0-dev",
            ),
            score=0.95,
        )

        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.StorageEngine") as mock_storage_class:
                mock_storage = AsyncMock()
                mock_storage.search.return_value = [mock_search_result]
                mock_storage_class.return_value = mock_storage

                with patch("docforge.cli.main.EmbeddingEngine") as mock_embed_class:
                    mock_embed = AsyncMock()
                    mock_embed.embed.return_value = [
                        MagicMock(
                            content="# Test\n\nTest content about creating indexes.",
                            metadata=mock_search_result.metadata,
                            vector=[0.1] * 8,
                        )
                    ]
                    mock_embed_class.return_value = mock_embed

                    result = runner.invoke(
                        app, ["search", "how to create index", "--software", "fixture"]
                    )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Test content" in result.output or "creating indexes" in result.output


class TestCLIList:
    @pytest.mark.asyncio
    async def test_list_command(self, runner: CliRunner, mock_config: DocForgeConfig) -> None:
        from docforge.storage.metadata_store import MetadataStore

        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.MetadataStore") as mock_store_class:
                mock_store = MagicMock(spec=MetadataStore)
                mock_store.list_software.return_value = [
                    {
                        "software": "postgresql",
                        "display_name": "PostgreSQL",
                        "latest_version": "17",
                        "last_indexed_at": "2025-01-01T00:00:00Z",
                    },
                    {
                        "software": "redis",
                        "display_name": "Redis",
                        "latest_version": "7.2",
                        "last_indexed_at": "2025-01-01T00:00:00Z",
                    },
                ]
                # Mock list_versions for each software
                mock_store.list_versions.side_effect = [
                    [
                        {
                            "version": "17",
                            "page_count": 100,
                            "chunk_count": 500,
                            "indexed_at": "2025-01-01T00:00:00Z",
                        }
                    ],
                    [
                        {
                            "version": "7.2",
                            "page_count": 50,
                            "chunk_count": 200,
                            "indexed_at": "2025-01-01T00:00:00Z",
                        }
                    ],
                ]
                mock_store_class.return_value = mock_store

                result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "postgresql" in result.output
        assert "redis" in result.output


class TestCLIStats:
    @pytest.mark.asyncio
    async def test_stats_command(self, runner: CliRunner, mock_config: DocForgeConfig) -> None:
        from docforge.storage.metadata_store import MetadataStore

        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.MetadataStore") as mock_store_class:
                mock_store = MagicMock(spec=MetadataStore)
                mock_store.get_software_stats.return_value = {
                    "software": "fixture",
                    "display_name": "Fixture Test",
                    "version_count": 1,
                    "page_count": 10,
                    "chunk_count": 50,
                }
                mock_store.list_versions.return_value = [
                    {
                        "version": "1.0",
                        "page_count": 10,
                        "chunk_count": 50,
                        "indexed_at": "2025-01-01T00:00:00Z",
                    }
                ]
                mock_store_class.return_value = mock_store

                result = runner.invoke(app, ["stats", "fixture"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "fixture" in result.output
        assert "10" in result.output  # page count
        assert "50" in result.output  # chunk count


class TestCLIDelete:
    @pytest.mark.asyncio
    async def test_delete_command(self, runner: CliRunner, mock_config: DocForgeConfig) -> None:
        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.MetadataStore") as mock_store_class:
                mock_store = MagicMock()
                mock_store.get_software.return_value = {
                    "software": "fixture",
                    "display_name": "Fixture Test",
                }
                mock_store.list_versions.return_value = [{"version": "1.0"}]
                mock_store_class.return_value = mock_store

                with patch("docforge.cli.main.StorageEngine") as mock_storage_class:
                    mock_storage = AsyncMock()
                    mock_storage.delete.return_value = None
                    mock_storage_class.return_value = mock_storage

                    result = runner.invoke(app, ["delete", "fixture", "--version", "1.0", "--yes"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"


class TestCLIUpdate:
    @pytest.mark.asyncio
    @respx.mock
    async def test_update_command(
        self,
        runner: CliRunner,
        mock_config: DocForgeConfig,
        mock_discovery: AsyncMock,
        provider: FakeEmbeddingProvider,
    ) -> None:
        _mock_html_pages()

        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.fixture.test/docs/1.0</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page1.html</loc><lastmod>2025-01-01</lastmod></url>
  <url><loc>https://docs.fixture.test/docs/page2.html</loc><lastmod>2025-01-01</lastmod></url>
</urlset>"""
        respx.get(f"{BASE}/sitemap.xml").respond(200, text=sitemap_xml)

        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.Pipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.software = "fixture"
                mock_result.total_duration_ms = 500
                mock_result.versions = [
                    MagicMock(
                        version="1.0",
                        status="completed",
                        error=None,
                        discovery=MagicMock(pages_processed=0, duration_ms=100),
                        crawl=MagicMock(pages_processed=0, chunks_produced=0, duration_ms=0),
                        extraction=MagicMock(pages_processed=0, duration_ms=0),
                        classification=MagicMock(pages_processed=0, duration_ms=0),
                        chunking=MagicMock(chunks_produced=0, duration_ms=0),
                        metadata=MagicMock(chunks_produced=0, duration_ms=0),
                        embedding=MagicMock(chunks_produced=0, duration_ms=0),
                        storage=MagicMock(chunks_produced=0, duration_ms=0),
                        total_duration_ms=500,
                    )
                ]
                mock_pipeline.run.return_value = mock_result
                mock_pipeline_class.return_value = mock_pipeline

                result = runner.invoke(app, ["update", "fixture"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"


class TestCLIReembed:
    @pytest.mark.asyncio
    async def test_reembed_command(self, runner: CliRunner, mock_config: DocForgeConfig) -> None:
        with patch("docforge.cli.main.load_config", return_value=mock_config):
            with patch("docforge.cli.main.Pipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_pipeline.run.return_value = MagicMock(
                    status="completed",
                    versions=[
                        MagicMock(
                            crawl=MagicMock(pages_processed=0),
                            extraction=MagicMock(pages_processed=0),
                            chunking=MagicMock(chunks_produced=10),
                            embedding=MagicMock(chunks_produced=10),
                            storage=MagicMock(chunks_produced=10),
                            total_duration_ms=2000,
                        )
                    ],
                )
                mock_pipeline_class.return_value = mock_pipeline

                result = runner.invoke(app, ["reembed", "fixture", "--model", "test-model"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"


class TestCLIHelp:
    def test_root_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "index" in result.output
        assert "search" in result.output
        assert "update" in result.output
        assert "reembed" in result.output
        assert "list" in result.output
        assert "stats" in result.output
        assert "delete" in result.output
        assert "config" in result.output

    def test_index_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0
        assert "software" in result.output
        assert "--version" in result.output

    def test_search_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output
        assert "--software" in result.output
        assert "--version" in result.output
        assert "--k" in result.output

    def test_update_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["update", "--help"])
        assert result.exit_code == 0

    def test_list_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_stats_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_delete_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["delete", "--help"])
        assert result.exit_code == 0

    def test_config_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_reembed_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["reembed", "--help"])
        assert result.exit_code == 0
