"""Unit tests for the discovery engine and its components."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from docforge.discovery.engine import DiscoveryEngine, DiscoveryError
from docforge.discovery.heuristics import _build_candidate_urls, _score_candidate
from docforge.discovery.registry import Registry, RegistryEntry
from docforge.discovery.robots import RobotsPolicy, parse_robots_txt
from docforge.discovery.sitemap import parse_sitemap, parse_sitemap_index
from docforge.discovery.version_detector import (
    _explicit_strategy,
    _extract_versions_from_urls,
    _parse_version_dropdown,
    _version_sort_key,
    detect_versions,
    looks_like_version,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_entry() -> RegistryEntry:
    return RegistryEntry(
        {
            "name": "postgresql",
            "display_name": "PostgreSQL",
            "documentation": {
                "base_url": "https://www.postgresql.org/docs/",
                "version_pattern": "https://www.postgresql.org/docs/{version}/",
                "versions": {
                    "strategy": "url_enumeration",
                    "known_versions": ["17", "16", "15"],
                    "latest": "17",
                },
                "content_selectors": {"main_content": "#docContent"},
                "url_filters": {"include": ["/docs/{version}/**"], "exclude": []},
            },
        }
    )


@pytest.fixture
def registry(pg_entry: RegistryEntry) -> Registry:
    return Registry([pg_entry])


@pytest.fixture
def sample_sitemap_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.postgresql.org/docs/17/tutorial-intro.html</loc>
    <lastmod>2026-01-15</lastmod>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://www.postgresql.org/docs/17/ddl.html</loc>
    <lastmod>2026-01-10</lastmod>
  </url>
  <url>
    <loc>https://www.postgresql.org/docs/16/tutorial-intro.html</loc>
  </url>
</urlset>"""


@pytest.fixture
def sample_sitemap_index_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://www.postgresql.org/sitemaps/docs-17.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://www.postgresql.org/sitemaps/docs-16.xml</loc>
  </sitemap>
</sitemapindex>"""


@pytest.fixture
def sample_robots_txt() -> str:
    return """User-agent: *
Disallow: /private/
Disallow: /internal/
Crawl-delay: 2

User-agent: DocForge
Allow: /

Sitemap: https://www.postgresql.org/sitemap.xml
"""


# ---------------------------------------------------------------------------
# Engine URL helpers (tested via the public API indirectly)
# ---------------------------------------------------------------------------


class TestEngineUrlHelpers:
    """Test the URL detection and name extraction logic."""

    def test_valid_https(self) -> None:
        from docforge.discovery.engine import _is_url

        assert _is_url("https://www.postgresql.org/docs/") is True

    def test_valid_http(self) -> None:
        from docforge.discovery.engine import _is_url

        assert _is_url("http://example.com") is True

    def test_name_string(self) -> None:
        from docforge.discovery.engine import _is_url

        assert _is_url("postgresql") is False

    def test_empty_string(self) -> None:
        from docforge.discovery.engine import _is_url

        assert _is_url("") is False

    def test_extract_name_standard_docs(self) -> None:
        from docforge.discovery.engine import _extract_name_from_url

        assert _extract_name_from_url("https://docs.postgresql.com/") == "postgresql"

    def test_extract_name_readthedocs(self) -> None:
        from docforge.discovery.engine import _extract_name_from_url

        name = _extract_name_from_url("https://flask.readthedocs.io/en/latest/")
        assert name == "flask"

    def test_extract_name_www_org(self) -> None:
        from docforge.discovery.engine import _extract_name_from_url

        name = _extract_name_from_url("https://www.postgresql.org/docs/")
        assert name == "postgresql"


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------


class TestSitemap:
    def test_parse_flat_sitemap(self, sample_sitemap_xml: str) -> None:
        urls = parse_sitemap(sample_sitemap_xml)
        assert len(urls) == 3
        assert urls[0].loc == "https://www.postgresql.org/docs/17/tutorial-intro.html"
        assert urls[0].lastmod == "2026-01-15"
        assert urls[0].priority == 0.8
        assert urls[1].lastmod == "2026-01-10"
        assert urls[2].lastmod is None

    def test_parse_sitemap_index(self, sample_sitemap_index_xml: str) -> None:
        child_urls = parse_sitemap_index(sample_sitemap_index_xml)
        assert len(child_urls) == 2
        assert "docs-17.xml" in child_urls[0]
        assert "docs-16.xml" in child_urls[1]

    def test_parse_empty_sitemap(self) -> None:
        xml = (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "</urlset>"
        )
        with pytest.raises(Exception):
            parse_sitemap(xml)

    def test_parse_invalid_xml(self) -> None:
        with pytest.raises(Exception):
            parse_sitemap("<not-valid-xml")

    def test_parse_no_namespace(self) -> None:
        xml = """<urlset>
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
</urlset>"""
        urls = parse_sitemap(xml)
        assert len(urls) == 2


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


class TestRobots:
    def test_parse_robots(self, sample_robots_txt: str) -> None:
        policy = parse_robots_txt(sample_robots_txt)
        assert policy.crawl_delay == 2.0
        assert "/private/" in policy.disallowed_paths
        assert "/internal/" in policy.disallowed_paths
        assert len(policy.sitemap_urls) == 1

    def test_is_allowed(self) -> None:
        policy = RobotsPolicy(disallowed_paths=["/private/", "/internal/"])
        assert policy.is_allowed("/docs/") is True
        assert policy.is_allowed("/private/secret") is False
        assert policy.is_allowed("/internal/data") is False

    def test_empty_robots(self) -> None:
        policy = parse_robots_txt("")
        assert policy.crawl_delay is None
        assert policy.disallowed_paths == []


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersionDetection:
    def test_explicit_strategy(self) -> None:
        versions = _explicit_strategy(["17", "16", "15"])
        assert versions == ["17", "16", "15"]

    def test_explicit_empty(self) -> None:
        assert _explicit_strategy([]) == []

    def test_looks_like_version(self) -> None:
        assert looks_like_version("17") is True
        assert looks_like_version("3.12") is True
        assert looks_like_version("7.2.0") is True
        assert looks_like_version("abc") is False
        assert looks_like_version("") is False

    def test_version_sort_key(self) -> None:
        assert _version_sort_key("17") > _version_sort_key("16")
        assert _version_sort_key("3.12") > _version_sort_key("3.11")
        assert _version_sort_key("7.2.0") > _version_sort_key("7.1.0")

    def test_extract_versions_from_urls(self) -> None:
        urls = [
            "https://www.postgresql.org/docs/17/tutorial.html",
            "https://www.postgresql.org/docs/17/ddl.html",
            "https://www.postgresql.org/docs/16/tutorial.html",
        ]
        versions = _extract_versions_from_urls(urls, "https://www.postgresql.org/docs/")
        assert "17" in versions
        assert "16" in versions

    def test_parse_version_dropdown(self) -> None:
        html = """
        <select class="version-select">
            <option value="17/">17</option>
            <option value="16/">16</option>
            <option value="15/">15</option>
        </select>
        """
        versions = _parse_version_dropdown(html)
        assert "17" in versions
        assert "16" in versions

    def test_parse_version_dropdown_links(self) -> None:
        html = '<a href="/docs/17/">17</a> <a href="/docs/16/">16</a>'
        versions = _parse_version_dropdown(html)
        assert "17" in versions
        assert "16" in versions

    @pytest.mark.asyncio
    async def test_detect_versions_explicit(self) -> None:
        versions = await detect_versions(
            "explicit",
            base_url="https://example.com/",
            known_versions=["2.0", "1.0"],
        )
        assert versions == ["2.0", "1.0"]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


class TestHeuristics:
    def test_build_candidate_urls(self) -> None:
        urls = _build_candidate_urls("postgresql")
        assert len(urls) >= 5
        assert any("postgresql" in u for u in urls)

    def test_score_candidate_docs_site(self) -> None:
        html = """
        <html>
        <nav><a href="/docs">Docs</a></nav>
        <h1>PostgreSQL Documentation</h1>
        <pre><code>SELECT * FROM users;</code></pre>
        </html>
        """
        candidate = _score_candidate("https://docs.postgresql.com/", html)
        assert candidate.confidence > 0.5
        assert candidate.has_headings is True
        assert candidate.has_code_blocks is True
        assert candidate.has_sitemap is False

    def test_score_candidate_low_quality(self) -> None:
        html = "<html><body>Sign up for our enterprise plan!</body></html>"
        candidate = _score_candidate("https://example.com/", html)
        assert candidate.confidence < 0.3

    def test_score_candidate_with_sitemap(self) -> None:
        html = (
            "<html><head>"
            '<link rel="sitemap" href="/sitemap.xml"/>'
            "</head><body><h1>Docs</h1></body></html>"
        )
        candidate = _score_candidate("https://docs.example.com/", html)
        assert candidate.has_sitemap is True


# ---------------------------------------------------------------------------
# DiscoveryEngine — name-based
# ---------------------------------------------------------------------------


class TestDiscoveryEngineName:
    @pytest.mark.asyncio
    async def test_discover_from_registry(self, registry: Registry) -> None:
        engine = DiscoveryEngine(registry=registry)
        result = await engine.discover("postgresql")
        assert result.software == "postgresql"
        assert result.display_name == "PostgreSQL"
        assert result.base_url == "https://www.postgresql.org/docs/"
        assert "17" in result.versions

    @pytest.mark.asyncio
    async def test_discover_from_name_direct(self, registry: Registry) -> None:
        engine = DiscoveryEngine(registry=registry)
        result = await engine.discover_from_name("postgresql")
        assert result.software == "postgresql"

    @pytest.mark.asyncio
    async def test_discover_unknown_raises(self, registry: Registry) -> None:
        engine = DiscoveryEngine(registry=registry)
        with patch(
            "docforge.discovery.engine.probe_heuristics",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with pytest.raises(DiscoveryError, match="Could not discover"):
                await engine.discover("nonexistent-software")


# ---------------------------------------------------------------------------
# DiscoveryEngine — URL-based
# ---------------------------------------------------------------------------


class TestDiscoveryEngineUrl:
    @pytest.mark.asyncio
    async def test_discover_from_url(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://www.postgresql.org/docs/"
        mock_response.text = """
        <html>
        <a href="/docs/17/">17</a>
        <a href="/docs/16/">16</a>
        </html>
        """

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        engine = DiscoveryEngine(client=mock_client)
        result = await engine.discover("https://www.postgresql.org/docs/")
        assert result.software == "postgresql"
        assert "17" in result.versions
        assert result.latest_version == "17"

    @pytest.mark.asyncio
    async def test_discover_url_invalid_raises(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.aclose = AsyncMock()

        engine = DiscoveryEngine(client=mock_client)
        with pytest.raises(DiscoveryError, match="Failed to access"):
            await engine.discover("https://invalid.example.com/docs/")


# ---------------------------------------------------------------------------
# DiscoveryEngine — integration with version_detector
# ---------------------------------------------------------------------------


class TestDiscoveryEngineIntegration:
    @pytest.mark.asyncio
    async def test_discover_postgresql_default_registry(self) -> None:
        engine = DiscoveryEngine()
        result = await engine.discover("postgresql")
        assert result.software == "postgresql"
        assert result.display_name == "PostgreSQL"
        assert result.base_url == "https://www.postgresql.org/docs/"
        assert "17" in result.versions
        assert result.latest_version == "17"

    @pytest.mark.asyncio
    async def test_discover_fastapi_default_registry(self) -> None:
        engine = DiscoveryEngine()
        result = await engine.discover("fastapi")
        assert result.software == "fastapi"
        assert result.display_name == "FastAPI"
        assert result.base_url == "https://fastapi.tiangolo.com/"
        assert "latest" in result.versions
        assert result.latest_version == "latest"

    @pytest.mark.asyncio
    async def test_discover_with_explicit_versions(self) -> None:
        entry = RegistryEntry(
            {
                "name": "test-sw",
                "display_name": "Test Software",
                "documentation": {
                    "base_url": "https://test.example.com/docs/",
                    "versions": {
                        "strategy": "explicit",
                        "known_versions": ["3.0", "2.0", "1.0"],
                        "latest": "3.0",
                    },
                },
            }
        )
        registry = Registry([entry])
        engine = DiscoveryEngine(registry=registry)
        result = await engine.discover("test-sw")
        assert result.versions == ["3.0", "2.0", "1.0"]
        assert result.latest_version == "3.0"

    @pytest.mark.asyncio
    async def test_detect_versions_postgresql_url_enumeration(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_client.head = AsyncMock(return_value=mock_head_resp)

        versions = await detect_versions(
            "url_enumeration",
            base_url="https://www.postgresql.org/docs/",
            version_pattern="https://www.postgresql.org/docs/{version}/",
            known_versions=["17", "16", "15", "14", "13"],
            client=mock_client,
        )
        assert versions == ["17", "16", "15", "14", "13"]
        assert mock_client.head.call_count == 5

    @pytest.mark.asyncio
    async def test_fetch_sitemap_index_recursive(self) -> None:
        from docforge.discovery.sitemap import fetch_sitemap

        index_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
        </sitemapindex>"""

        child_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/docs/page1</loc></url>
        </urlset>"""

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if url == "https://example.com/sitemap.xml":
                resp.text = index_xml
            else:
                resp.text = child_xml
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=mock_get)

        urls = await fetch_sitemap("https://example.com/sitemap.xml", client=mock_client)
        assert len(urls) == 1
        assert urls[0].loc == "https://example.com/docs/page1"
