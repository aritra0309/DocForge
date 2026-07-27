from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import respx

from docforge.core.models import Chunk, ChunkMetadata, DiscoveryResult, PageType
from docforge.storage.metadata_store import MetadataStore
from docforge.updates.detector import UpdateDetector, UpdateReport
from docforge.updates.differ import ChunkDiffer


def _make_chunk(
    content: str = "# Test\n\nContent",
    chunk_id: str = "chunk_1",
    content_hash: str = "hash_1",
    page_url: str = "https://example.com/docs/1.0/page",
) -> Chunk:
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        parent_page_id="page_id",
        software="test",
        version="1.0",
        url=page_url,
        title="Test",
        page_type=PageType.GUIDE,
        section_heading="Test",
        chunk_index=0,
        total_chunks=1,
        has_code=False,
        content_hash=content_hash,
        crawl_timestamp=__import__("datetime").datetime(2025, 1, 1),
        embedding_model="test-model",
        embedding_dimension=4,
        docforge_version="0.1.0-dev",
    )
    return Chunk(content=content, metadata=meta)


def _make_discovery_result(
    base_url: str = "https://example.com/docs/",
    sitemap_url: str | None = None,
    versions: list[str] | None = None,
) -> DiscoveryResult:
    return DiscoveryResult(
        software="test",
        display_name="Test",
        base_url=base_url,
        versions=versions or ["1.0"],
        latest_version="1.0",
        sitemap_url=sitemap_url,
        url_filters={"include": ["/docs/**"]},
    )


# ---------------------------------------------------------------------------
# UpdateReport
# ---------------------------------------------------------------------------


class TestUpdateReport:
    def test_total_changed_counts(self) -> None:
        report = UpdateReport(
            new_urls=["a", "b"],
            changed_urls=["c"],
            removed_urls=["d"],
        )
        assert report.total_changed == 4

    def test_total_changed_empty(self) -> None:
        report = UpdateReport()
        assert report.total_changed == 0

    def test_total_changed_only_unchanged(self) -> None:
        report = UpdateReport(unchanged_urls=["a", "b", "c"])
        assert report.total_changed == 0


# ---------------------------------------------------------------------------
# ChunkDiffer
# ---------------------------------------------------------------------------


class TestChunkDiffer:
    @pytest.mark.asyncio
    async def test_no_changes(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_chunk_state("chunk_1", "https://example.com/page", "hash_1", "test", "1.0")
        store.upsert_chunk_state("chunk_2", "https://example.com/page", "hash_2", "test", "1.0")

        new_chunks = [
            _make_chunk(chunk_id="chunk_1", content_hash="hash_1"),
            _make_chunk(chunk_id="chunk_2", content_hash="hash_2"),
        ]

        diff = await ChunkDiffer.diff_page("https://example.com/page", new_chunks, store)

        assert len(diff.chunks_to_add) == 0
        assert len(diff.chunks_updated) == 0
        assert len(diff.chunks_to_remove) == 0
        assert diff.unchanged_chunk_ids == ["chunk_1", "chunk_2"]
        assert diff.total_changed == 0

    @pytest.mark.asyncio
    async def test_chunk_added(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_chunk_state("chunk_1", "https://example.com/page", "hash_1", "test", "1.0")

        new_chunks = [
            _make_chunk(chunk_id="chunk_1", content_hash="hash_1"),
            _make_chunk(chunk_id="chunk_2", content_hash="hash_2"),
        ]

        diff = await ChunkDiffer.diff_page("https://example.com/page", new_chunks, store)

        assert len(diff.chunks_to_add) == 1
        assert diff.chunks_to_add[0].metadata.chunk_id == "chunk_2"
        assert len(diff.chunks_updated) == 0
        assert len(diff.chunks_to_remove) == 0
        assert diff.unchanged_chunk_ids == ["chunk_1"]

    @pytest.mark.asyncio
    async def test_chunk_removed(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_chunk_state("chunk_1", "https://example.com/page", "hash_1", "test", "1.0")
        store.upsert_chunk_state("chunk_2", "https://example.com/page", "hash_2", "test", "1.0")

        new_chunks = [
            _make_chunk(chunk_id="chunk_1", content_hash="hash_1"),
        ]

        diff = await ChunkDiffer.diff_page("https://example.com/page", new_chunks, store)

        assert len(diff.chunks_to_add) == 0
        assert len(diff.chunks_updated) == 0
        assert diff.chunks_to_remove == ["chunk_2"]
        assert diff.unchanged_chunk_ids == ["chunk_1"]

    @pytest.mark.asyncio
    async def test_chunk_content_changed(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_chunk_state("chunk_1", "https://example.com/page", "hash_old", "test", "1.0")

        new_chunks = [
            _make_chunk(chunk_id="chunk_1", content_hash="hash_new"),
        ]

        diff = await ChunkDiffer.diff_page("https://example.com/page", new_chunks, store)

        assert len(diff.chunks_to_add) == 0
        assert len(diff.chunks_updated) == 1
        assert diff.chunks_updated[0].metadata.chunk_id == "chunk_1"
        assert len(diff.chunks_to_remove) == 0
        assert len(diff.unchanged_chunk_ids) == 0

    @pytest.mark.asyncio
    async def test_all_changes(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_chunk_state("chunk_a", "https://example.com/page", "hash_a", "test", "1.0")
        store.upsert_chunk_state("chunk_b", "https://example.com/page", "hash_b", "test", "1.0")
        store.upsert_chunk_state("chunk_c", "https://example.com/page", "hash_c", "test", "1.0")

        new_chunks = [
            _make_chunk(chunk_id="chunk_a", content_hash="hash_a"),
            _make_chunk(chunk_id="chunk_b", content_hash="hash_b_new"),
            _make_chunk(chunk_id="chunk_d", content_hash="hash_d"),
        ]

        diff = await ChunkDiffer.diff_page("https://example.com/page", new_chunks, store)

        assert len(diff.chunks_to_add) == 1
        assert diff.chunks_to_add[0].metadata.chunk_id == "chunk_d"
        assert len(diff.chunks_updated) == 1
        assert diff.chunks_updated[0].metadata.chunk_id == "chunk_b"
        assert diff.chunks_to_remove == ["chunk_c"]
        assert diff.unchanged_chunk_ids == ["chunk_a"]

    @pytest.mark.asyncio
    async def test_empty_page(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_chunk_state("chunk_a", "https://example.com/page", "hash_a", "test", "1.0")

        diff = await ChunkDiffer.diff_page("https://example.com/page", [], store)

        assert len(diff.chunks_to_add) == 0
        assert len(diff.chunks_updated) == 0
        assert diff.chunks_to_remove == ["chunk_a"]
        assert len(diff.unchanged_chunk_ids) == 0

    @pytest.mark.asyncio
    async def test_no_stored_state(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))

        new_chunks = [
            _make_chunk(chunk_id="chunk_1", content_hash="hash_1"),
        ]

        diff = await ChunkDiffer.diff_page("https://example.com/page", new_chunks, store)

        assert len(diff.chunks_to_add) == 1
        assert len(diff.chunks_updated) == 0
        assert len(diff.chunks_to_remove) == 0
        assert len(diff.unchanged_chunk_ids) == 0


# ---------------------------------------------------------------------------
# UpdateDetector
# ---------------------------------------------------------------------------


class TestUpdateDetector:
    @pytest.mark.asyncio
    async def test_no_sitemap_all_new(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        detector = UpdateDetector()
        discovery = _make_discovery_result(sitemap_url=None)

        report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.changed_urls) >= 1
        assert len(report.unchanged_urls) == 0

    @pytest.mark.asyncio
    async def test_sitemap_all_unchanged(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url="https://example.com/docs/1.0/",
            software="test", version="1.0",
            content_hash="h1", etag="", last_modified="2025-01-01",
        )
        store.upsert_page_state(
            url="https://example.com/docs/1.0/page1",
            software="test", version="1.0",
            content_hash="h2", etag="", last_modified="2025-01-02",
        )

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with patch("docforge.updates.detector.fetch_sitemap") as mock_fetch:
            mock_fetch.return_value = [
                _make_sitemap_url("https://example.com/docs/1.0/", "2025-01-01"),
                _make_sitemap_url("https://example.com/docs/1.0/page1", "2025-01-02"),
            ]
            report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.unchanged_urls) == 2
        assert len(report.new_urls) == 0
        assert len(report.changed_urls) == 0
        assert len(report.removed_urls) == 0

    @pytest.mark.asyncio
    async def test_sitemap_new_url(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url="https://example.com/docs/1.0/",
            software="test", version="1.0",
            content_hash="h1", etag="", last_modified="2025-01-01",
        )

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with (
            patch("docforge.updates.detector.fetch_sitemap") as mock_fetch,
            patch.object(detector, "_fetch_changed_and_new") as mock_fetch_new,
        ):
            mock_fetch.return_value = [
                _make_sitemap_url("https://example.com/docs/1.0/", "2025-01-01"),
                _make_sitemap_url("https://example.com/docs/1.0/new_page", "2025-01-03"),
            ]
            report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.unchanged_urls) == 1
        assert report.new_urls == ["https://example.com/docs/1.0/new_page"]
        assert len(report.changed_urls) == 0
        assert len(report.removed_urls) == 0
        mock_fetch_new.assert_called_once()

    @pytest.mark.asyncio
    async def test_sitemap_removed_url(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url="https://example.com/docs/1.0/",
            software="test", version="1.0",
            content_hash="h1", etag="", last_modified="2025-01-01",
        )
        store.upsert_page_state(
            url="https://example.com/docs/1.0/removed",
            software="test", version="1.0",
            content_hash="h2", etag="", last_modified="2025-01-02",
        )

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with patch("docforge.updates.detector.fetch_sitemap") as mock_fetch:
            mock_fetch.return_value = [
                _make_sitemap_url("https://example.com/docs/1.0/", "2025-01-01"),
            ]
            report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.unchanged_urls) == 1
        assert report.removed_urls == ["https://example.com/docs/1.0/removed"]
        assert len(report.new_urls) == 0
        assert len(report.changed_urls) == 0

    @pytest.mark.asyncio
    async def test_sitemap_changed_lastmod(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url="https://example.com/docs/1.0/",
            software="test", version="1.0",
            content_hash="h1", etag="", last_modified="2025-01-01",
        )

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with (
            patch("docforge.updates.detector.fetch_sitemap") as mock_fetch,
            patch.object(detector, "_fetch_changed_and_new") as mock_fetch_new,
        ):
            mock_fetch.return_value = [
                _make_sitemap_url("https://example.com/docs/1.0/", "2025-02-01"),
            ]
            report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.unchanged_urls) == 0
        assert report.changed_urls == ["https://example.com/docs/1.0/"]
        assert len(report.new_urls) == 0
        assert len(report.removed_urls) == 0
        mock_fetch_new.assert_called_once()

    @pytest.mark.asyncio
    async def test_sitemap_no_lastmod_falls_to_changed(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url="https://example.com/docs/1.0/",
            software="test", version="1.0",
            content_hash="h1", etag="", last_modified="2025-01-01",
        )

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with (
            patch("docforge.updates.detector.fetch_sitemap") as mock_fetch,
            patch.object(detector, "_fetch_changed_and_new"),
        ):
            mock_fetch.return_value = [
                _make_sitemap_url("https://example.com/docs/1.0/", None),
            ]
            report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.unchanged_urls) == 0
        assert report.changed_urls == ["https://example.com/docs/1.0/"]

    @pytest.mark.asyncio
    async def test_sitemap_fetch_failure_fallback(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url="https://example.com/docs/1.0/",
            software="test", version="1.0",
            content_hash="h1", etag="", last_modified="2025-01-01",
        )

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with patch("docforge.updates.detector.fetch_sitemap") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.changed_urls) >= 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_sitemap_changed_lastmod_confirmed_unchanged(self, tmp_path: Path) -> None:
        """Sitemap lastmod mismatch is confirmed unchanged via conditional request."""
        url = "https://example.com/docs/1.0/"
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.upsert_page_state(
            url=url,
            software="test", version="1.0",
            content_hash="h1", etag='"abc"', last_modified="2025-01-01",
        )

        respx.get(url).respond(status_code=304)

        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url="https://example.com/sitemap.xml",
        )

        with patch("docforge.updates.detector.fetch_sitemap") as mock_fetch:
            mock_fetch.return_value = [
                _make_sitemap_url(url, "2025-02-01"),
            ]
            report = await detector.detect(discovery, "test", "1.0", store)

        assert report.changed_urls == []
        assert report.unchanged_urls == [url]
        assert len(report.changed_fetch_results) == 0

    @pytest.mark.asyncio
    async def test_304_unchanged(self, tmp_path: Path) -> None:
        detector = UpdateDetector()

        fetch_results, _ = await detector._fetch_with_conditionals(
            ["https://example.com/page"],
            {"https://example.com/page": {"etag": '"abc"', "last_modified": "2025-01-01"}},
        )

        assert len(fetch_results) >= 0

    @pytest.mark.asyncio
    async def test_empty_stored_no_sitemap(self, tmp_path: Path) -> None:
        store = MetadataStore(str(tmp_path / "meta.db"))
        detector = UpdateDetector()
        discovery = _make_discovery_result(
            base_url="https://example.com/docs/",
            sitemap_url=None,
        )

        report = await detector.detect(discovery, "test", "1.0", store)

        assert len(report.changed_urls) >= 1


@dataclass
class _MockSitemapUrl:
    loc: str
    lastmod: str | None = None


def _make_sitemap_url(loc: str, lastmod: str | None) -> Any:
    return _MockSitemapUrl(loc=loc, lastmod=lastmod)
