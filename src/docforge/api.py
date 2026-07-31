"""DocForge — Automatically discover, crawl, version, chunk, and index software documentation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from docforge._version import __version__
from docforge.core.config import DocForgeConfig, load_config
from docforge.core.events import EventBus, PipelineEvent
from docforge.core.models import SearchResult
from docforge.core.pipeline import Pipeline, PipelineResult, PipelineVersionResult
from docforge.storage.engine import StorageEngine
from docforge.storage.metadata_store import MetadataStore
from docforge.versioning.manager import VersionManager


class DocForge:
    """Main entry point for DocForge Python API.

    Usage:
        forge = DocForge()
        await forge.index("postgresql")
        results = await forge.search("how to create index", software="postgresql")
    """

    def __init__(self, config: DocForgeConfig | None = None) -> None:
        """Initialize DocForge.

        Args:
            config: Optional configuration. If not provided, loads from default sources.
        """
        self._config = config or load_config()
        self._pipeline = Pipeline(self._config)
        self._events = EventBus()
        self._event_handlers: dict[str, list[Callable]] = {}

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def index(
        self,
        software: str,
        version: str | None = None,
        mode: str = "full",
        **kwargs: Any,
    ) -> PipelineResult:
        """Run the indexing pipeline.

        Args:
            software: Software name to index (e.g., "postgresql").
            version: Specific version to index (default: latest).
            mode: Pipeline mode - "full", "incremental", or "reembed".
            **kwargs: Additional arguments for specific modes.

        Returns:
            PipelineResult with statistics and status.
        """
        return await self._pipeline.run(software=software, version=version, mode=mode, **kwargs)

    async def search(
        self,
        query: str,
        software: str,
        version: str | None = None,
        k: int = 10,
    ) -> list[SearchResult]:
        """Search indexed documentation using semantic search.

        Args:
            query: Search query string.
            software: Software name to search in.
            version: Specific version (default: "latest").
            k: Number of results to return.

        Returns:
            List of SearchResult objects.
        """
        from docforge.embeddings.engine import EmbeddingEngine
        from docforge.embeddings.providers.sentence_transformers import SentenceTransformersProvider

        provider = SentenceTransformersProvider(model_name=self._config.embeddings.model)
        embedding_engine = EmbeddingEngine(
            provider=provider, batch_size=self._config.embeddings.batch_size
        )

        ver = version
        if ver is None or ver == "latest":
            from docforge.discovery.registry import load_registry

            registry = load_registry()
            entry = registry.lookup(software)
            if entry:
                ver = entry.latest_version
            else:
                raise ValueError(f"Software '{software}' not found in registry")

        storage = StorageEngine(self._config, software=software, version=ver)
        await storage.initialize(dimension=provider.dimension, model_name=provider.model_name)

        query_vector = (await embedding_engine.embed_batch([query]))[0]
        results = await storage.search(query_vector, k=k)

        await storage.close()
        await embedding_engine.close()
        return results

    async def update(
        self,
        software: str | None = None,
        version: str | None = None,
    ) -> PipelineResult:
        """Incrementally update indexed documentation.

        Args:
            software: Software name to update (default: all indexed).
            version: Specific version to update.

        Returns:
            PipelineResult with update statistics.
        """
        if software:
            return await self._pipeline.run(software=software, version=version, mode="incremental")

        meta_store = MetadataStore(self._config.storage.path / "metadata.db")
        indexed = meta_store.list_software()
        meta_store.close()

        if not indexed:
            return PipelineResult(software="all", status="completed")

        overall = PipelineResult(software="all", status="completed")
        for sw in indexed:
            result = await self._pipeline.run(
                software=sw["software"], version=version, mode="incremental"
            )
            overall.versions.extend(result.versions)
            if result.status == "failed":
                overall.status = "partial"
        return overall

    async def reembed(
        self,
        software: str,
        new_model: str,
        version: str | None = None,
        old_model: str | None = None,
    ) -> PipelineResult:
        """Re-embed existing chunks with a different model.

        Args:
            software: Software name.
            new_model: New embedding model name.
            version: Specific version (default: latest).
            old_model: Old model name (auto-detected if not provided).

        Returns:
            PipelineResult with re-embedding statistics.
        """
        kwargs = {"new_model": new_model}
        if old_model:
            kwargs["old_model"] = old_model
        return await self._pipeline.run(
            software=software, version=version, mode="reembed", **kwargs
        )

    async def list_indexed(self) -> list[dict[str, Any]]:
        """List all indexed software.

        Returns:
            List of software info dicts with versions, page/chunk counts.
        """
        meta_store = MetadataStore(self._config.storage.path / "metadata.db")
        indexed = meta_store.list_software()
        meta_store.close()
        return indexed

    async def stats(self, software: str, version: str | None = None) -> dict[str, Any]:
        """Get statistics for indexed software.

        Args:
            software: Software name.
            version: Specific version (optional).

        Returns:
            Statistics dictionary.
        """
        meta_store = MetadataStore(self._config.storage.path / "metadata.db")
        stats = meta_store.get_software_stats(software)
        if stats["version_count"] == 0:
            meta_store.close()
            raise ValueError(f"Software '{software}' not indexed")

        versions = meta_store.list_versions(software)
        meta_store.close()

        return {
            "software": software,
            "version_count": stats["version_count"],
            "page_count": stats["page_count"],
            "chunk_count": stats["chunk_count"],
            "versions": versions,
        }

    async def delete(self, software: str, version: str | None = None) -> None:
        """Delete indexed software or version.

        Args:
            software: Software name.
            version: Specific version to delete (default: all versions).
        """
        meta_store = MetadataStore(self._config.storage.path / "metadata.db")

        if version:
            if not meta_store.get_version(software, version):
                meta_store.close()
                raise ValueError(f"Version '{version}' not found for '{software}'")

            storage = StorageEngine(self._config, software=software, version=version)
            await storage.initialize()
            await storage.delete(filters={"software": software, "version": version})
            await storage.close()
            meta_store.delete_version(software, version)
        else:
            if not meta_store.get_software(software):
                meta_store.close()
                raise ValueError(f"Software '{software}' not indexed")

            versions = meta_store.list_versions(software)
            for v in versions:
                storage = StorageEngine(self._config, software=software, version=v["version"])
                await storage.initialize()
                await storage.delete(filters={"software": software, "version": v["version"]})
                await storage.close()
            meta_store.delete_software(software)

        meta_store.close()

    async def get_versions(self, software: str) -> list[str]:
        """Get all indexed versions for a software.

        Args:
            software: Software name.

        Returns:
            List of version strings sorted oldest to newest.
        """
        meta_store = MetadataStore(self._config.storage.path / "metadata.db")
        versions = meta_store.list_versions(software)
        meta_store.close()
        return [v["version"] for v in versions]

    async def get_latest_version(self, software: str) -> str | None:
        """Get the latest indexed version.

        Args:
            software: Software name.

        Returns:
            Latest version string or None if not indexed.
        """
        versions = await self.get_versions(software)
        return versions[-1] if versions else None

    async def close(self) -> None:
        """Close all resources."""
        await self._pipeline.close()

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler.

        Args:
            event_type: Event type to listen for (e.g., "crawl.page.fetched").
            handler: Callable that receives a PipelineEvent.
        """
        self._events.on(event_type, handler)
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable | None = None) -> None:
        """Remove an event handler.

        Args:
            event_type: Event type.
            handler: Specific handler to remove (or all if None).
        """
        self._events.off(event_type, handler)
        if handler is None:
            self._event_handlers.pop(event_type, None)
        else:
            handlers = self._event_handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def _run_sync(self, coro):
        """Run async method synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call sync method from running event loop. Use async method instead."
            )
        return asyncio.run(coro)

    # Sync versions of all async methods
    def index_sync(
        self,
        software: str,
        version: str | None = None,
        mode: str = "full",
        **kwargs: Any,
    ) -> PipelineResult:
        """Synchronous version of index()."""
        return self._run_sync(self.index(software, version, mode, **kwargs))

    def search_sync(
        self,
        query: str,
        software: str,
        version: str | None = None,
        k: int = 10,
    ) -> list[SearchResult]:
        """Synchronous version of search()."""
        return self._run_sync(self.search(query, software, version, k))

    def update_sync(
        self,
        software: str | None = None,
        version: str | None = None,
    ) -> PipelineResult:
        """Synchronous version of update()."""
        return self._run_sync(self.update(software, version))

    def reembed_sync(
        self,
        software: str,
        new_model: str,
        version: str | None = None,
        old_model: str | None = None,
    ) -> PipelineResult:
        """Synchronous version of reembed()."""
        return self._run_sync(self.reembed(software, new_model, version, old_model))

    def list_indexed_sync(self) -> list[dict[str, Any]]:
        """Synchronous version of list_indexed()."""
        return self._run_sync(self.list_indexed())

    def stats_sync(self, software: str, version: str | None = None) -> dict[str, Any]:
        """Synchronous version of stats()."""
        return self._run_sync(self.stats(software, version))

    def delete_sync(self, software: str, version: str | None = None) -> None:
        """Synchronous version of delete()."""
        return self._run_sync(self.delete(software, version))

    def get_versions_sync(self, software: str) -> list[str]:
        """Synchronous version of get_versions()."""
        return self._run_sync(self.get_versions(software))

    def get_latest_version_sync(self, software: str) -> str | None:
        """Synchronous version of get_latest_version()."""
        return self._run_sync(self.get_latest_version(software))

    def close_sync(self) -> None:
        """Synchronous version of close()."""
        return self._run_sync(self.close())

    def __enter__(self) -> DocForge:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_sync()

    async def __aenter__(self) -> DocForge:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


__all__ = ["DocForge", "PipelineEvent", "PipelineVersionResult", "VersionManager", "__version__"]
