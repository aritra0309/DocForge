from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from docforge._version import __version__
from docforge.chunker.engine import ChunkingEngine
from docforge.classifier.engine import ClassificationEngine
from docforge.core import events
from docforge.core.config import DocForgeConfig, load_config
from docforge.core.interfaces import EmbeddingProvider
from docforge.core.models import Chunk, DiscoveryResult
from docforge.crawler.engine import CrawlEngine
from docforge.discovery.engine import DiscoveryEngine
from docforge.discovery.registry import load_registry
from docforge.embeddings.cache import EmbeddingCache
from docforge.embeddings.engine import EmbeddingEngine
from docforge.embeddings.providers.openai import OpenAIEmbeddingProvider
from docforge.embeddings.providers.sentence_transformers import (
    SentenceTransformersProvider,
)
from docforge.embeddings.providers.voyage import VoyageEmbeddingProvider
from docforge.extractor.engine import ExtractionEngine
from docforge.metadata.generator import MetadataGenerator
from docforge.storage.engine import StorageEngine
from docforge.updates.detector import UpdateDetector
from docforge.updates.differ import ChunkDiffer

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageStats:
    pages_processed: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_produced: int = 0
    duration_ms: float = 0.0


@dataclass
class PipelineVersionResult:
    software: str
    version: str
    discovery: PipelineStageStats = field(default_factory=PipelineStageStats)
    crawl: PipelineStageStats = field(default_factory=PipelineStageStats)
    extraction: PipelineStageStats = field(default_factory=PipelineStageStats)
    classification: PipelineStageStats = field(default_factory=PipelineStageStats)
    chunking: PipelineStageStats = field(default_factory=PipelineStageStats)
    metadata: PipelineStageStats = field(default_factory=PipelineStageStats)
    embedding: PipelineStageStats = field(default_factory=PipelineStageStats)
    storage: PipelineStageStats = field(default_factory=PipelineStageStats)
    total_duration_ms: float = 0.0
    status: str = "completed"
    error: str | None = None


@dataclass
class PipelineResult:
    software: str
    versions: list[PipelineVersionResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    status: str = "completed"
    error: str | None = None


def _create_embedding_provider(config: DocForgeConfig) -> EmbeddingProvider:
    provider_name = config.embeddings.provider
    model_name = config.embeddings.model
    if provider_name == "sentence-transformers":
        return SentenceTransformersProvider(model_name=model_name)
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(model_name=model_name)
    if provider_name == "voyage":
        return VoyageEmbeddingProvider(model_name=model_name)
    msg = f"Unknown embedding provider: {provider_name!r}"
    raise ValueError(msg)


class Pipeline:
    def __init__(self, config: DocForgeConfig | None = None) -> None:
        self.config = config or load_config()
        self.events = events.EventBus()
        self._discovery: DiscoveryEngine | None = None
        self._crawler: CrawlEngine | None = None
        self._extractor: ExtractionEngine | None = None
        self._classifier: ClassificationEngine | None = None
        self._chunker: ChunkingEngine | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._embedding_engine: EmbeddingEngine | None = None
        self._embedding_cache: EmbeddingCache | None = None
        self._registry = load_registry()
        self._closed = False

    @property
    def discovery(self) -> DiscoveryEngine:
        if self._discovery is None:
            self._discovery = DiscoveryEngine(registry=self._registry)
        return self._discovery

    @property
    def crawler(self) -> CrawlEngine:
        if self._crawler is None:
            self._crawler = CrawlEngine(config=self.config)
        return self._crawler

    @property
    def extractor(self) -> ExtractionEngine:
        if self._extractor is None:
            self._extractor = ExtractionEngine()
        return self._extractor

    @property
    def classifier(self) -> ClassificationEngine:
        if self._classifier is None:
            self._classifier = ClassificationEngine()
        return self._classifier

    @property
    def chunker(self) -> ChunkingEngine:
        if self._chunker is None:
            self._chunker = ChunkingEngine(
                target_chunk_size=self.config.chunker.target_chunk_size,
                max_chunk_size=self.config.chunker.max_chunk_size,
                overlap_tokens=self.config.chunker.overlap_tokens,
            )
        return self._chunker

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = _create_embedding_provider(self.config)
        return self._embedding_provider

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            cache: EmbeddingCache | None = None
            if self.config.embeddings.cache_embeddings:
                cache_path = self.config.general.data_dir / "embedding_cache.db"
                cache = EmbeddingCache(cache_path)
                self._embedding_cache = cache
            self._embedding_engine = EmbeddingEngine(
                provider=self.embedding_provider,
                cache=cache,
                batch_size=self.config.embeddings.batch_size,
            )
        return self._embedding_engine

    async def run(
        self,
        software: str,
        version: str | None = None,
        mode: str = "full",
        **kwargs: Any,
    ) -> PipelineResult:
        if self._closed:
            msg = "Pipeline is closed"
            raise RuntimeError(msg)
        t0 = time.monotonic()
        await self.events.emit(
            events.PIPELINE_STARTED, software=software, version=version, mode=mode
        )
        error: str | None = None
        result: PipelineResult | None = None
        try:
            result = await self._run_with_mode(software, version, mode)
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.exception("Pipeline failed")
            error = f"{type(exc).__name__}: {exc}"
            result = PipelineResult(software=software, status="failed", error=error)
            await self.events.emit(events.PIPELINE_ERROR, software=software, error=error)
        assert result is not None
        result.total_duration_ms = (time.monotonic() - t0) * 1000
        if error:
            result.status = "failed"
            result.error = error
        await self.events.emit(
            events.PIPELINE_COMPLETED,
            software=software,
            status=result.status,
            duration_ms=result.total_duration_ms,
            version_count=len(result.versions),
        )
        return result

    async def _run_with_mode(
        self,
        software: str,
        version: str | None,
        mode: str,
    ) -> PipelineResult:
        if mode == "full":
            return await self._run_full(software, version)
        if mode == "incremental":
            return await self._run_incremental(software, version)
        if mode == "reembed":
            msg = "Re-embed mode requires Task 17 — not yet implemented"
            raise NotImplementedError(msg)
        msg = f"Unknown pipeline mode: {mode!r}"
        raise ValueError(msg)

    async def _run_full(
        self,
        software: str,
        version: str | None = None,
    ) -> PipelineResult:
        await self.events.emit(events.DISCOVERY_STARTED, software=software)
        discovery_result = await self.discovery.discover(software)
        await self.events.emit(
            events.DISCOVERY_COMPLETED,
            software=discovery_result.software,
            display_name=discovery_result.display_name,
            versions=discovery_result.versions,
        )
        versions_to_index: list[str] = []
        if version:
            if version not in discovery_result.versions:
                if version == "latest":
                    versions_to_index = [discovery_result.latest_version]
                else:
                    versions_to_index = [version]
            else:
                versions_to_index = [version]
        else:
            versions_to_index = [discovery_result.latest_version]

        version_results: list[PipelineVersionResult] = []
        for ver in versions_to_index:
            vr = await self._run_version(discovery_result, ver)
            version_results.append(vr)

        any_failed = any(vr.status == "failed" for vr in version_results)
        all_failed = all(vr.status == "failed" for vr in version_results)
        overall = "failed" if all_failed else ("partial" if any_failed else "completed")
        return PipelineResult(
            software=discovery_result.software,
            versions=version_results,
            status=overall,
        )

    async def _run_incremental(
        self,
        software: str,
        version: str | None = None,
    ) -> PipelineResult:
        await self.events.emit(events.DISCOVERY_STARTED, software=software)
        discovery_result = await self.discovery.discover(software)
        await self.events.emit(
            events.DISCOVERY_COMPLETED,
            software=discovery_result.software,
            display_name=discovery_result.display_name,
            versions=discovery_result.versions,
        )
        versions_to_index: list[str] = []
        if version:
            if version not in discovery_result.versions:
                if version == "latest":
                    versions_to_index = [discovery_result.latest_version]
                else:
                    versions_to_index = [version]
            else:
                versions_to_index = [version]
        else:
            versions_to_index = [discovery_result.latest_version]

        version_results: list[PipelineVersionResult] = []
        for ver in versions_to_index:
            vr = await self._run_version_incremental(discovery_result, ver)
            version_results.append(vr)

        any_failed = any(vr.status == "failed" for vr in version_results)
        all_failed = all(vr.status == "failed" for vr in version_results)
        overall = "failed" if all_failed else ("partial" if any_failed else "completed")
        return PipelineResult(
            software=discovery_result.software,
            versions=version_results,
            status=overall,
        )

    async def _run_version(  # ruff: ignore[PLR0914]
        self,
        discovery_result: DiscoveryResult,
        version: str,
    ) -> PipelineVersionResult:
        t0 = time.monotonic()
        software = discovery_result.software
        vr = PipelineVersionResult(software=software, version=version)

        engines = self._init_version_engines(discovery_result, version)
        extractor, classifier, chunker, metadata_gen, storage, provider = engines

        try:
            await storage.initialize(dimension=provider.dimension, model_name=provider.model_name)
        except Exception as exc:
            vr.status = "failed"
            vr.error = f"Storage initialization failed: {exc}"
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        metadata_store = storage.metadata_store
        run_id = metadata_store.create_run(software=software, version=version, mode="full")
        metadata_store.upsert_software(
            software=software, display_name=discovery_result.display_name,
        )

        fetch_results, err = await self._crawl_version(discovery_result, version, vr)
        if err:
            metadata_store.complete_run(run_id, status="failed", error_log=err)
            self.crawler.close()
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        all_chunks = await self._process_pages(
            fetch_results, extractor, classifier, chunker, metadata_gen,
            metadata_store, software, version, vr,
        )
        vr.chunking.chunks_produced = len(all_chunks)

        if not all_chunks:
            metadata_store.complete_run(
                run_id, status="completed",
                page_count=len(fetch_results), chunk_count=0,
                embedding_model=provider.model_name,
            )
            metadata_store.upsert_version(
                software=software, version=version,
                page_count=len(fetch_results), chunk_count=0,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimension,
            )
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        embedded, embed_err = await self._embed_chunks(
            all_chunks, provider, software, version, vr,
        )
        if embed_err:
            metadata_store.complete_run(run_id, status="failed", error_log=embed_err)
            self.crawler.close()
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        store_err = await self._store_chunks(
            embedded, storage, software, version, vr,
        )
        if store_err:
            metadata_store.complete_run(run_id, status="failed", error_log=store_err)
            self.crawler.close()
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        await self._finalize_version(
            metadata_store, run_id, fetch_results, embedded,
            provider, software, version, vr,
        )
        self.crawler.close()
        await storage.close()
        vr.total_duration_ms = (time.monotonic() - t0) * 1000
        return vr

    async def _run_version_incremental(  # ruff: ignore[PLR0914]
        self,
        discovery_result: DiscoveryResult,
        version: str,
    ) -> PipelineVersionResult:
        t0 = time.monotonic()
        software = discovery_result.software
        vr = PipelineVersionResult(software=software, version=version)

        engines = self._init_version_engines(discovery_result, version)
        extractor, classifier, chunker, metadata_gen, storage, provider = engines

        try:
            await storage.initialize(dimension=provider.dimension, model_name=provider.model_name)
        except Exception as exc:
            vr.status = "failed"
            vr.error = f"Storage initialization failed: {exc}"
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        metadata_store = storage.metadata_store
        run_id = metadata_store.create_run(
            software=software, version=version, mode="incremental",
        )
        metadata_store.upsert_software(
            software=software, display_name=discovery_result.display_name,
        )

        detector = UpdateDetector(config=self.config)
        report = await detector.detect(discovery_result, software, version, metadata_store)
        await self.events.emit(
            events.UPDATE_COMPLETED,
            software=software, version=version,
            new=len(report.new_urls), changed=len(report.changed_urls),
            removed=len(report.removed_urls), unchanged=len(report.unchanged_urls),
        )

        await self._delete_removed_pages(report, storage, metadata_store, vr)

        new_page_chunks = await self._process_pages(
            report.new_fetch_results, extractor, classifier, chunker,
            metadata_gen, metadata_store, software, version, vr,
        )
        if new_page_chunks:
            metadata_store.upsert_chunk_states(new_page_chunks)

        changed_page_chunks = await self._process_changed_pages(
            report.changed_fetch_results, extractor, classifier, chunker,
            metadata_gen, metadata_store, software, version, vr,
        )

        all_chunks_to_store = new_page_chunks + changed_page_chunks

        if not all_chunks_to_store:
            metadata_store.complete_run(
                run_id, status="completed",
                page_count=0, chunk_count=0,
                embedding_model=provider.model_name,
            )
            metadata_store.upsert_version(
                software=software, version=version,
                page_count=0, chunk_count=0,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimension,
            )
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        embedded, embed_err = await self._embed_chunks(
            all_chunks_to_store, provider, software, version, vr,
        )
        if embed_err:
            metadata_store.complete_run(run_id, status="failed", error_log=embed_err)
            self.crawler.close()
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        store_err = await self._store_chunks(
            embedded, storage, software, version, vr,
        )
        if store_err:
            metadata_store.complete_run(run_id, status="failed", error_log=store_err)
            self.crawler.close()
            await storage.close()
            vr.total_duration_ms = (time.monotonic() - t0) * 1000
            return vr

        await self._finalize_version(
            metadata_store, run_id, report.new_fetch_results + report.changed_fetch_results,
            embedded, provider, software, version, vr,
        )
        self.crawler.close()
        await storage.close()
        vr.total_duration_ms = (time.monotonic() - t0) * 1000
        return vr

    async def _delete_removed_pages(
        self,
        report: Any,
        storage: StorageEngine,
        metadata_store: Any,
        vr: PipelineVersionResult,
    ) -> None:
        for url in report.removed_urls:
            try:
                await storage.delete(filters={"url": url})
                metadata_store.delete_page_state(url)
                metadata_store.delete_chunk_state_by_page(url)
                await self.events.emit(events.UPDATE_PAGE_REMOVED, url=url)
                vr.extraction.pages_processed += 1
            except Exception as exc:
                vr.extraction.pages_failed += 1
                logger.warning("Failed to remove page %s: %s", url, exc)

    async def _process_changed_pages(  # ruff: ignore[PLR0917]
        self,
        fetch_results: list[Any],
        extractor: ExtractionEngine,
        classifier: ClassificationEngine,
        chunker: ChunkingEngine,
        metadata_gen: MetadataGenerator,
        metadata_store: Any,
        software: str,
        version: str,
        vr: PipelineVersionResult,
    ) -> list[Any]:
        all_chunks: list[Any] = []
        differ = ChunkDiffer()
        for fetch_result in fetch_results:
            try:
                await self._process_single_changed_page(
                    fetch_result, extractor, classifier, chunker,
                    metadata_gen, metadata_store, software, version, vr, differ, all_chunks,
                )
            except Exception as exc:
                vr.extraction.pages_failed += 1
                logger.warning("Failed to process changed page %s: %s", fetch_result.url, exc)
        return all_chunks

    async def _process_single_changed_page(  # ruff: ignore[PLR0917]
        self,
        fetch_result: Any,
        extractor: ExtractionEngine,
        classifier: ClassificationEngine,
        chunker: ChunkingEngine,
        metadata_gen: MetadataGenerator,
        metadata_store: Any,
        software: str,
        version: str,
        vr: PipelineVersionResult,
        differ: ChunkDiffer,
        all_chunks: list[Any],
    ) -> None:
        enriched = await self._process_single_page(
            fetch_result, extractor, classifier, chunker,
            metadata_gen, metadata_store, software, version, vr,
        )
        diff = await differ.diff_page(
            fetch_result.url, enriched, metadata_store,
        )
        await self.events.emit(
            events.UPDATE_CHUNK_DIFFED,
            url=fetch_result.url,
            to_add=len(diff.chunks_to_add),
            to_update=len(diff.chunks_updated),
            to_remove=len(diff.chunks_to_remove),
            unchanged=len(diff.unchanged_chunk_ids),
        )
        if diff.chunks_to_remove:
            try:
                for cid in diff.chunks_to_remove:
                    metadata_store.delete_chunk_state(cid)
            except Exception as exc:
                msg = "Failed to delete stale chunks for %s: %s"
                logger.warning(msg, fetch_result.url, exc)
        metadata_store.upsert_chunk_states(enriched)
        all_chunks.extend(diff.chunks_to_add + diff.chunks_updated)
        await self.events.emit(events.UPDATE_PAGE_REINDEXED, url=fetch_result.url)

    def _init_version_engines(
        self,
        discovery_result: DiscoveryResult,
        version: str,
    ) -> tuple[
        ExtractionEngine, ClassificationEngine, ChunkingEngine,
        MetadataGenerator, StorageEngine, EmbeddingProvider,
    ]:
        software = discovery_result.software
        content_selectors = discovery_result.content_selectors or {}
        registry_entry = self._registry.lookup(software) if self._registry else None
        page_type_hints: dict[str, list[str]] = {}
        if registry_entry:
            page_type_hints = registry_entry.page_type_hints

        extractor = ExtractionEngine(content_selectors=content_selectors)
        classifier = ClassificationEngine(page_type_hints=page_type_hints)
        chunker = ChunkingEngine(
            target_chunk_size=self.config.chunker.target_chunk_size,
            max_chunk_size=self.config.chunker.max_chunk_size,
            overlap_tokens=self.config.chunker.overlap_tokens,
        )
        provider = self.embedding_provider
        metadata_gen = MetadataGenerator(
            software=software,
            version=version,
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
            docforge_version=__version__,
        )
        storage = StorageEngine(self.config, software=software, version=version)
        return extractor, classifier, chunker, metadata_gen, storage, provider

    async def _crawl_version(
        self,
        discovery_result: DiscoveryResult,
        version: str,
        vr: PipelineVersionResult,
    ) -> tuple[list[Any] | None, str | None]:
        software = discovery_result.software
        base_url = discovery_result.base_url.rstrip("/")
        seed_url = f"{base_url}/{version}/"
        crawl_t0 = time.monotonic()
        await self.events.emit(events.CRAWL_STARTED, software=software, version=version)
        try:
            fetch_results = await self.crawler.crawl(
                seed_urls=[seed_url],
                discovery_result=discovery_result,
                max_pages=self.config.crawler.max_pages_per_version,
            )
        except Exception as exc:
            err = f"Crawl failed: {exc}"
            vr.status = "failed"
            vr.error = err
            return None, err
        vr.crawl.pages_processed = len(fetch_results)
        vr.crawl.duration_ms = (time.monotonic() - crawl_t0) * 1000
        await self.events.emit(
            events.CRAWL_COMPLETED, software=software, version=version,
            pages=len(fetch_results),
        )
        return fetch_results, None

    async def _process_pages(  # ruff: ignore[PLR0917]
        self,
        fetch_results: list[Any],
        extractor: ExtractionEngine,
        classifier: ClassificationEngine,
        chunker: ChunkingEngine,
        metadata_gen: MetadataGenerator,
        metadata_store: Any,
        software: str,
        version: str,
        vr: PipelineVersionResult,
    ) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        extract_t0 = time.monotonic()
        for fetch_result in fetch_results:
            try:
                enriched = await self._process_single_page(
                    fetch_result, extractor, classifier, chunker,
                    metadata_gen, metadata_store, software, version, vr,
                )
                all_chunks.extend(enriched)
            except Exception as exc:
                vr.extraction.pages_failed += 1
                logger.warning("Failed to process page %s: %s", fetch_result.url, exc)
        vr.extraction.duration_ms = (time.monotonic() - extract_t0) * 1000
        return all_chunks

    async def _process_single_page(  # ruff: ignore[PLR0917]
        self,
        fetch_result: Any,
        extractor: ExtractionEngine,
        classifier: ClassificationEngine,
        chunker: ChunkingEngine,
        metadata_gen: MetadataGenerator,
        metadata_store: Any,
        software: str,
        version: str,
        vr: PipelineVersionResult,
    ) -> list[Chunk]:
        await self.events.emit(events.EXTRACTION_STARTED, url=fetch_result.url)
        page = await extractor.extract(fetch_result)
        await self.events.emit(events.EXTRACTION_COMPLETED, url=fetch_result.url)
        vr.extraction.pages_processed += 1

        classified = classifier.classify(page)
        await self.events.emit(
            events.CLASSIFICATION_COMPLETED, url=fetch_result.url,
            page_type=classified.page_type.value,
        )
        vr.classification.pages_processed += 1

        chunks = chunker.chunk(classified)
        await self.events.emit(
            events.CHUNKING_COMPLETED, url=fetch_result.url, chunk_count=len(chunks),
        )
        vr.chunking.pages_processed += 1
        vr.chunking.chunks_produced += len(chunks)

        enriched = metadata_gen.generate(chunks, classified)
        await self.events.emit(
            events.METADATA_GENERATED, url=fetch_result.url, chunk_count=len(enriched),
        )
        vr.metadata.pages_processed += 1
        vr.metadata.chunks_produced += len(enriched)

        metadata_store.upsert_page_state(
            url=fetch_result.url,
            software=software,
            version=version,
            content_hash=enriched[0].metadata.content_hash if enriched else "",
            etag=fetch_result.etag or "",
            last_modified=fetch_result.last_modified or "",
        )
        if enriched:
            metadata_store.upsert_chunk_states(enriched)
        return enriched

    async def _embed_chunks(
        self,
        all_chunks: list[Chunk],
        provider: EmbeddingProvider,
        software: str,
        version: str,
        vr: PipelineVersionResult,
    ) -> tuple[list[Any] | None, str | None]:
        embed_t0 = time.monotonic()
        await self.events.emit(
            events.EMBEDDING_STARTED, software=software, version=version, chunks=len(all_chunks),
        )
        try:
            embedded = await self.embedding_engine.embed(all_chunks)
        except Exception as exc:
            err = f"Embedding failed: {exc}"
            vr.status = "failed"
            vr.error = err
            return None, err
        vr.embedding.chunks_produced = len(embedded)
        vr.embedding.duration_ms = (time.monotonic() - embed_t0) * 1000
        await self.events.emit(
            events.EMBEDDING_COMPLETED,
            software=software, version=version, chunks=len(embedded),
        )
        return embedded, None

    async def _store_chunks(
        self,
        embedded: list[Any],
        storage: StorageEngine,
        software: str,
        version: str,
        vr: PipelineVersionResult,
    ) -> str | None:
        store_t0 = time.monotonic()
        try:
            await storage.upsert(embedded)
        except Exception as exc:
            err = f"Storage upsert failed: {exc}"
            vr.status = "failed"
            vr.error = err
            return err
        vr.storage.chunks_produced = len(embedded)
        vr.storage.duration_ms = (time.monotonic() - store_t0) * 1000
        await self.events.emit(
            events.STORAGE_UPSERTED, software=software, version=version, chunks=len(embedded),
        )
        return None

    async def _finalize_version(  # ruff: ignore[PLR0917]
        self,
        metadata_store: Any,
        run_id: int,
        fetch_results: list[Any],
        embedded: list[Any],
        provider: EmbeddingProvider,
        software: str,
        version: str,
        vr: PipelineVersionResult,
    ) -> None:
        metadata_store.complete_run(
            run_id, status="completed",
            page_count=len(fetch_results), chunk_count=len(embedded),
            embedding_model=provider.model_name,
        )
        metadata_store.upsert_version(
            software=software, version=version,
            page_count=len(fetch_results), chunk_count=len(embedded),
            embedding_model=provider.model_name, embedding_dimension=provider.dimension,
        )
        await self.events.emit(
            events.VERSION_COMPLETED, software=software, version=version,
            pages=len(fetch_results), chunks=len(embedded),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.crawler.close()
        if self._embedding_cache is not None:
            self._embedding_cache.close()
        if self._embedding_engine is not None:
            await self._embedding_engine.close()
        self._discovery = None
        self._crawler = None
        self._extractor = None
        self._classifier = None
        self._chunker = None
        self._embedding_provider = None
        self._embedding_engine = None
        self._embedding_cache = None
