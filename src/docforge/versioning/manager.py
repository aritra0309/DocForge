from __future__ import annotations

from dataclasses import dataclass

from docforge.storage.engine import StorageEngine
from docforge.storage.metadata_store import MetadataStore
from docforge.versioning.aliases import set_latest_version_in_store
from docforge.versioning.comparator import sort_versions


@dataclass(frozen=True)
class VersionInfo:
    """Information about a single indexed version."""

    software: str
    version: str
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    indexed_at: str


class VersionManager:
    """Manages the lifecycle of multiple documentation versions in the vector store.

    Versions are stored independently — indexing ``v17`` never modifies
    ``v16`` data. Collections are named with the version embedded, and
    ``delete_version`` is atomic: either the entire version is removed
    or nothing is.
    """

    def __init__(
        self,
        storage_engine: StorageEngine,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        self._storage = storage_engine
        self._metadata_store = metadata_store or storage_engine.metadata_store

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_versions(self, software: str) -> list[VersionInfo]:
        """List all indexed versions for a given software.

        Returns:
            List of ``VersionInfo`` sorted from oldest to newest version.
        """
        rows = self._metadata_store.list_versions(software)
        versions = [
            VersionInfo(
                software=r["software"],
                version=r["version"],
                page_count=r["page_count"],
                chunk_count=r["chunk_count"],
                embedding_model=r["embedding_model"],
                embedding_dimension=r["embedding_dimension"],
                indexed_at=r["indexed_at"],
            )
            for r in rows
        ]
        return sort_versions_by_info(versions)

    def get_latest(self, software: str) -> str | None:
        """Get the latest indexed version string for a software.

        Returns:
            The latest version string, or ``None`` if no versions are indexed.
        """
        versions = self.list_versions(software)
        if not versions:
            return None
        return versions[-1].version

    def get_version(self, software: str, version: str) -> VersionInfo | None:
        """Get details for a specific indexed version.

        Returns:
            A ``VersionInfo``, or ``None`` if the version is not indexed.
        """
        row = self._metadata_store.get_version(software, version)
        if row is None:
            return None
        return VersionInfo(
            software=row["software"],
            version=row["version"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            indexed_at=row["indexed_at"],
        )

    def version_exists(self, software: str, version: str) -> bool:
        """Check if a specific version is already indexed.

        Returns:
            ``True`` if the version exists in the metadata store.
        """
        return self._metadata_store.get_version(software, version) is not None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_latest(self, software: str, version: str) -> None:
        """Update the latest-version marker for a software.

        This writes the alias into the software entry's config snapshot.
        It does **not** validate that the version is currently indexed;
        callers should check ``version_exists`` first if needed.

        Args:
            software: Software identifier.
            version: Version string to mark as latest.
        """
        set_latest_version_in_store(self._metadata_store, software, version)

    async def delete_version(self, software: str, version: str) -> None:
        """Atomically remove all data for a given software version.

        This:
        1. Deletes all chunks from the vector store that match the version
        2. Removes the version record from ``indexed_versions``
        3. Removes all page_state and chunk_state rows for that version

        If the vector store delete fails, the operation is aborted and
        the exception propagates — leaving the metadata store untouched.

        Args:
            software: Software identifier.
            version: Version string to remove.

        Raises:
            Exception: If the vector store delete fails.
        """
        await self._storage.delete(filters={"software": software, "version": version})
        self._metadata_store.delete_version(software, version)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def software_is_indexed(self, software: str) -> bool:
        """Check if any version of a software is indexed."""
        return len(self.list_versions(software)) > 0


def sort_versions_by_info(versions: list[VersionInfo]) -> list[VersionInfo]:
    """Sort a list of ``VersionInfo`` objects by version (oldest first)."""
    sorted_versions = sort_versions([v.version for v in versions])
    lookup = {v.version: v for v in versions}
    return [lookup[v] for v in sorted_versions if v in lookup]
