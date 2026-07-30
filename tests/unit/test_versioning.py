from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock

import pytest

from docforge.core.config import DocForgeConfig
from docforge.storage.engine import StorageEngine
from docforge.storage.metadata_store import MetadataStore
from docforge.versioning import (
    VersionComparator,
    VersionInfo,
    VersionManager,
    compare_versions,
)
from docforge.versioning.aliases import resolve_latest

# ---------------------------------------------------------------------------
# Comparator tests
# ---------------------------------------------------------------------------


class TestVersionComparator:
    def test_numeric_ordering(self) -> None:
        assert VersionComparator("17") > VersionComparator("16")
        assert VersionComparator("16") > VersionComparator("15")
        assert VersionComparator("15") < VersionComparator("17")

    def test_semver_ordering(self) -> None:
        assert VersionComparator("7.2.0") > VersionComparator("7.1.0")
        assert VersionComparator("3.10.0") > VersionComparator("3.9.0")
        assert VersionComparator("1.0.0") < VersionComparator("2.0.0")

    def test_mixed_segment_counts(self) -> None:
        assert VersionComparator("17.0") > VersionComparator("16")
        assert VersionComparator("17") == VersionComparator("17.0.0")

    def test_equal_versions(self) -> None:
        assert VersionComparator("17") == VersionComparator("17")
        assert not (VersionComparator("17") != VersionComparator("17"))

    def test_short_versions(self) -> None:
        assert VersionComparator("5") < VersionComparator("6")
        assert VersionComparator("9") < VersionComparator("10")

    def test_sort_list(self) -> None:
        versions = ["17", "15", "16", "14", "13"]
        expected = ["13", "14", "15", "16", "17"]
        from docforge.versioning.comparator import sort_versions

        assert sort_versions(versions) == expected

    def test_sort_semver_list(self) -> None:
        versions = ["7.2.0", "7.1.0", "7.10.0", "7.1.5"]
        expected = ["7.1.0", "7.1.5", "7.2.0", "7.10.0"]
        from docforge.versioning.comparator import sort_versions

        assert sort_versions(versions) == expected

    def test_compare_versions(self) -> None:
        assert compare_versions("17", "16") == 1
        assert compare_versions("16", "17") == -1
        assert compare_versions("17", "17") == 0

    def test_non_numeric(self) -> None:
        assert VersionComparator("latest") == VersionComparator("latest")
        assert VersionComparator("alpha") < VersionComparator("beta")

    def test_edge_cases(self) -> None:
        assert VersionComparator("1.0.0-alpha") < VersionComparator("1.0.0")
        assert VersionComparator("0.0.1") < VersionComparator("0.0.2")


# ---------------------------------------------------------------------------
# Aliases tests
# ---------------------------------------------------------------------------


class TestResolveLatest:
    def test_returns_unchanged_if_not_latest(self) -> None:
        store = MagicMock(spec=MetadataStore)
        result = resolve_latest(store, "postgresql", "17")
        assert result == "17"
        store.list_versions.assert_not_called()

    def test_resolves_to_latest_indexed(self) -> None:
        store = MagicMock(spec=MetadataStore)
        store.list_versions.return_value = [
            {"version": "15"},
            {"version": "16"},
            {"version": "17"},
        ]
        result = resolve_latest(store, "postgresql", "latest")
        assert result == "17"

    def test_resolves_to_newest_with_semver(self) -> None:
        store = MagicMock(spec=MetadataStore)
        store.list_versions.return_value = [
            {"version": "7.1.0"},
            {"version": "7.2.0"},
            {"version": "7.10.0"},
        ]
        result = resolve_latest(store, "postgresql", "latest")
        assert result == "7.10.0"

    def test_raises_if_no_versions(self) -> None:
        store = MagicMock(spec=MetadataStore)
        store.list_versions.return_value = []
        with pytest.raises(ValueError, match="No indexed versions"):
            resolve_latest(store, "postgresql", "latest")


# ---------------------------------------------------------------------------
# VersionManager tests
# ---------------------------------------------------------------------------


class TestVersionManager:
    @pytest.fixture
    def metadata_store(self) -> MetadataStore:
        tmp = TemporaryDirectory()
        db_path = Path(tmp.name) / "metadata.db"
        store = MetadataStore(db_path)
        store.upsert_software("postgresql", "PostgreSQL")
        store.upsert_version("postgresql", "15", page_count=10, chunk_count=100)
        store.upsert_version("postgresql", "17", page_count=20, chunk_count=200)
        store.upsert_version("postgresql", "16", page_count=15, chunk_count=150)
        return store

    @pytest.fixture
    def version_manager(self, metadata_store: MetadataStore) -> VersionManager:
        storage = MagicMock(spec=StorageEngine)
        storage.metadata_store = metadata_store
        return VersionManager(storage, metadata_store=metadata_store)

    def test_list_versions_orders_correctly(self, version_manager: VersionManager) -> None:
        versions = version_manager.list_versions("postgresql")
        version_strings = [v.version for v in versions]
        assert version_strings == ["15", "16", "17"]

    def test_list_versions_returns_version_info(self, version_manager: VersionManager) -> None:
        versions = version_manager.list_versions("postgresql")
        for v in versions:
            assert isinstance(v, VersionInfo)
            assert v.software == "postgresql"

    def test_get_latest(self, version_manager: VersionManager) -> None:
        latest = version_manager.get_latest("postgresql")
        assert latest == "17"

    def test_get_latest_no_versions(self, version_manager: VersionManager) -> None:
        latest = version_manager.get_latest("unknown_software")
        assert latest is None

    def test_get_version_exists(self, version_manager: VersionManager) -> None:
        info = version_manager.get_version("postgresql", "16")
        assert info is not None
        assert info.version == "16"
        assert info.page_count == 15
        assert info.chunk_count == 150

    def test_get_version_not_found(self, version_manager: VersionManager) -> None:
        info = version_manager.get_version("postgresql", "99")
        assert info is None

    def test_version_exists_true(self, version_manager: VersionManager) -> None:
        assert version_manager.version_exists("postgresql", "17") is True

    def test_version_exists_false(self, version_manager: VersionManager) -> None:
        assert version_manager.version_exists("postgresql", "99") is False

    def test_set_latest(self, version_manager: VersionManager) -> None:
        version_manager.set_latest("postgresql", "16")
        entry = version_manager._metadata_store.get_software("postgresql")
        assert entry is not None
        import json

        config = json.loads(entry["config_snapshot"])
        assert config.get("latest_version") == "16"

    def test_delete_version(self, version_manager: VersionManager) -> None:
        storage = version_manager._storage
        storage.delete = AsyncMock()

        # Use asyncio to run the async method
        import asyncio

        asyncio.run(version_manager.delete_version("postgresql", "16"))

        storage.delete.assert_awaited_once_with(filters={"software": "postgresql", "version": "16"})
        assert version_manager.version_exists("postgresql", "16") is False
        # v17 should still exist
        assert version_manager.version_exists("postgresql", "17") is True

    def test_software_is_indexed(self, version_manager: VersionManager) -> None:
        assert version_manager.software_is_indexed("postgresql") is True

    def test_software_is_not_indexed(self, version_manager: VersionManager) -> None:
        assert version_manager.software_is_indexed("mongodb") is False


class TestVersionManagerRealStorage:
    @pytest.fixture
    def storage_engine(self) -> StorageEngine:
        config = DocForgeConfig()
        engine = StorageEngine(config, software="postgresql", version="17")
        return engine

    @pytest.fixture
    def metadata_store(self) -> MetadataStore:
        tmp = TemporaryDirectory()
        db_path = Path(tmp.name) / "metadata.db"
        store = MetadataStore(db_path)
        return store

    def test_construct_with_storage_engine(
        self, storage_engine: StorageEngine, metadata_store: MetadataStore
    ) -> None:
        storage_engine._metadata_store = metadata_store
        vm = VersionManager(storage_engine)
        assert vm._metadata_store is metadata_store
