from __future__ import annotations

import logging
from dataclasses import dataclass, field

from docforge.core.models import Chunk
from docforge.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


@dataclass
class DiffReport:
    """Chunk-level diff for a single page."""

    chunks_to_add: list[Chunk] = field(default_factory=list)
    chunks_updated: list[Chunk] = field(default_factory=list)
    chunks_to_remove: list[str] = field(default_factory=list)
    unchanged_chunk_ids: list[str] = field(default_factory=list)

    @property
    def total_changed(self) -> int:
        return (
            len(self.chunks_to_add)
            + len(self.chunks_updated)
            + len(self.chunks_to_remove)
        )


class ChunkDiffer:
    """Compares newly chunked page content against stored chunk hashes.

    Only chunks whose content_hash changed are flagged for re-embedding.
    This minimises the number of embedding API calls during incremental updates.
    """

    @staticmethod
    async def diff_page(
        page_url: str,
        new_chunks: list[Chunk],
        metadata_store: MetadataStore,
    ) -> DiffReport:
        """Compare new chunks against stored chunk state for a page.

        Args:
            page_url: The URL of the page being compared.
            new_chunks: The freshly chunked content for this page.
            metadata_store: Store with chunk state entries from the previous index.

        Returns:
            A DiffReport detailing which chunks changed and how.
        """
        stored_entries = metadata_store.list_chunk_states(page_url)
        stored_by_chunk_id: dict[str, str] = {
            e["chunk_id"]: e["content_hash"] for e in stored_entries
        }

        report = DiffReport()
        new_chunk_ids: set[str] = set()

        for chunk in new_chunks:
            cid = chunk.metadata.chunk_id
            new_chunk_ids.add(cid)
            new_hash = chunk.metadata.content_hash

            if cid not in stored_by_chunk_id:
                report.chunks_to_add.append(chunk)
            elif stored_by_chunk_id[cid] != new_hash:
                report.chunks_updated.append(chunk)
                stored_by_chunk_id.pop(cid, None)
            else:
                report.unchanged_chunk_ids.append(cid)
                stored_by_chunk_id.pop(cid, None)

        report.chunks_to_remove = [
            cid for cid in stored_by_chunk_id
            if cid not in new_chunk_ids
        ]

        return report
