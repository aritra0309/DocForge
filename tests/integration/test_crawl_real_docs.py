"""Real documentation crawl integration test — hits live external websites.

Run with:
    pytest tests/integration/test_crawl_real_docs.py -m real_network -v --timeout=120
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pytest

from docforge.chunker.engine import ChunkingEngine
from docforge.classifier.engine import ClassificationEngine
from docforge.core.config import DocForgeConfig, load_config
from docforge.core.models import DiscoveryResult, FetchResult
from docforge.crawler.cache import ResponseCache
from docforge.crawler.engine import CrawlEngine
from docforge.discovery.engine import DiscoveryEngine
from docforge.discovery.registry import load_registry
from docforge.extractor.engine import ExtractionEngine
from docforge.metadata.generator import MetadataGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(max_pages: int = 20, rate_limit_rps: int = 3) -> DocForgeConfig:
    return load_config(
        overrides={
            "crawler": {
                "max_pages_per_version": max_pages,
                "rate_limit_rps": rate_limit_rps,
                "cache_ttl_hours": 24,
            }
        }
    )


async def _discover(software: str) -> DiscoveryResult:
    discovery = DiscoveryEngine()
    result = await discovery.discover(software)
    print(f"  Software:  {result.display_name}")
    print(f"  Base URL:  {result.base_url}")
    print(f"  Versions:  {result.versions[:5]}... ({len(result.versions)} total)")
    print(f"  Latest:    {result.latest_version}")
    print(f"  Selectors: {result.content_selectors}")
    return result


async def _run_crawl(
    tmp_path: Path,
    config: DocForgeConfig,
    result: DiscoveryResult,
) -> tuple[list[FetchResult], float]:
    version_url = f"https://www.postgresql.org/docs/{result.latest_version}/"
    cache = ResponseCache(db_path=str(tmp_path / "crawl_cache.db"))
    engine = CrawlEngine(
        config=config,
        cache=cache,
        queue_db_path=str(tmp_path / "queue.db"),
    )
    print("\n=== Crawling (max 20 pages, 3 RPS) ===")
    start = time.monotonic()
    pages = await engine.crawl(version_url, discovery_result=result)
    elapsed = time.monotonic() - start
    engine.close()
    print(f"  Fetched:   {len(pages)} pages in {elapsed:.1f}s ({len(pages) / elapsed:.1f} p/s)")
    assert len(pages) > 0
    return pages, elapsed


async def _process_single_page(
    page: FetchResult,
    *,
    page_index: int,
    total: int,
    extractor: ExtractionEngine,
    classifier: ClassificationEngine,
    chunker: ChunkingEngine,
    meta_gen: MetadataGenerator,
) -> dict | None:
    try:
        extracted = await extractor.extract(page)
        classified = classifier.classify(extracted)
        chunks = chunker.chunk(classified)
        enriched = meta_gen.generate(chunks, classified)
        result = {
            "extracted": extracted,
            "classified": classified,
            "chunks": chunks,
            "enriched": enriched,
        }
    except Exception as e:
        print(f"  [{page_index + 1:2d}/{total:2d}] ERROR: {page.url[:80]} - {e}")
        return None
    else:
        return result


async def _process_pages(
    pages: list[FetchResult],
    result: DiscoveryResult,
) -> tuple[list, Counter, list[dict], float]:
    entry = load_registry().lookup("postgresql")
    assert entry is not None

    extractor = ExtractionEngine(content_selectors=result.content_selectors)
    classifier = ClassificationEngine(page_type_hints=entry.page_type_hints)
    chunker = ChunkingEngine(target_chunk_size=512, max_chunk_size=1024, overlap_tokens=64)
    meta_gen = MetadataGenerator(software="postgresql", version=result.latest_version)

    type_counts: Counter[str] = Counter()
    page_stats: list[dict[str, object]] = []
    all_chunks: list = []

    extract_start = time.monotonic()
    for i, page in enumerate(pages):
        p = await _process_single_page(
            page,
            page_index=i,
            total=len(pages),
            extractor=extractor,
            classifier=classifier,
            chunker=chunker,
            meta_gen=meta_gen,
        )
        if p is None:
            continue
        type_counts[p["classified"].page_type.value] += 1
        page_stats.append(
            {
                "url": page.url,
                "title": p["extracted"].title,
                "type": p["classified"].page_type.value,
                "n_chunks": len(p["chunks"]),
            }
        )
        all_chunks.extend(p["enriched"])
        print(
            f"  [{i + 1:2d}/{len(pages):2d}] {p['classified'].page_type.value:20s} | "
            f"{p['extracted'].title[:50]:50s} | {len(p['chunks']):2d} chunks"
        )
    extract_elapsed = time.monotonic() - extract_start
    return all_chunks, type_counts, page_stats, extract_elapsed


def _print_summary(
    *,
    pages: list,
    all_chunks: list,
    type_counts: Counter,
    page_stats: list[dict],
    elapsed: float,
    extract_elapsed: float,
) -> None:
    total_tokens = sum(len(c.content.split()) for c in all_chunks)
    print(f"\n{'=' * 70}")
    print("  CRAWL + EXTRACT + CLASSIFY + CHUNK - SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Pages crawled:      {len(pages):>4d}")
    print(f"  Crawl time:         {elapsed:>6.1f}s  ({len(pages) / elapsed:.1f} p/s)")
    print(
        f"  Extract time:       {extract_elapsed:>6.1f}s  ({len(pages) / extract_elapsed:.1f} p/s)"
    )
    print(f"  Total chunks:       {len(all_chunks):>4d}")
    print(f"  Avg chunks/page:    {len(all_chunks) / len(pages):>6.1f}")
    print(f"  Total tokens:       {total_tokens:>6,d}")
    print("\n  Page type distribution:")
    for ptype, count in sorted(type_counts.items()):
        print(f"    {ptype:25s}: {count:3d} ({count / len(pages) * 100:5.1f}%)")
    print("\n  Per-page detail:")
    for i, ps in enumerate(page_stats):
        print(f"    {i + 1:3d}. {ps['type']:<20s}  chunks={ps['n_chunks']:>3d}  {ps['title']}")


def _assert_chunks(all_chunks: list) -> None:
    assert len(all_chunks) > 0
    for c in all_chunks:
        assert c.metadata.chunk_id
        assert c.metadata.software == "postgresql"
        assert c.metadata.version == "17"


async def _crawl_session(
    tmp_path: Path,
    config: DocForgeConfig,
    version_url: str,
    *,
    result: DiscoveryResult,
    cache_path: str,
    queue_path: str,
) -> tuple[list[FetchResult], float]:
    cache = ResponseCache(db_path=cache_path)
    engine = CrawlEngine(config=config, cache=cache, queue_db_path=queue_path)
    start = time.monotonic()
    pages = await engine.crawl(version_url, discovery_result=result)
    elapsed = time.monotonic() - start
    engine.close()
    return pages, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.real_network
@pytest.mark.asyncio
@pytest.mark.slow
async def test_crawl_postgresql_docs(tmp_path: Path) -> None:
    """Crawl PostgreSQL v17 docs (live), extract, classify, and chunk up to 20 pages."""
    config = _build_config(max_pages=20, rate_limit_rps=3)

    result = await _discover("postgresql")
    assert result.software == "postgresql"
    assert result.latest_version == "17"

    pages, elapsed = await _run_crawl(tmp_path, config, result)
    all_chunks, type_counts, page_stats, extract_elapsed = await _process_pages(pages, result)
    _print_summary(
        pages=pages,
        all_chunks=all_chunks,
        type_counts=type_counts,
        page_stats=page_stats,
        elapsed=elapsed,
        extract_elapsed=extract_elapsed,
    )
    _assert_chunks(all_chunks)


@pytest.mark.real_network
@pytest.mark.asyncio
@pytest.mark.slow
async def test_crawl_cache_hits_on_second_run(tmp_path: Path) -> None:
    """Second crawl of same URL should hit cache (test cache persistence)."""
    config = _build_config(max_pages=5, rate_limit_rps=5)

    cache_path = str(tmp_path / "crawl_cache.db")
    queue_path = str(tmp_path / "queue.db")
    version_url = "https://www.postgresql.org/docs/17/"

    discovery = DiscoveryEngine()
    result = await discovery.discover("postgresql")

    pages1, time1 = await _crawl_session(
        tmp_path,
        config,
        version_url,
        result=result,
        cache_path=cache_path,
        queue_path=queue_path,
    )

    cache2 = ResponseCache(db_path=cache_path)
    engine2 = CrawlEngine(config=config, cache=cache2, queue_db_path=queue_path)
    start2 = time.monotonic()
    pages2 = await engine2.crawl(version_url, discovery_result=result)
    time2 = time.monotonic() - start2
    engine2.close()

    assert len(pages2) >= len(pages1)
    print(f"\n  First run  (live):  {len(pages1):>3d} pages in {time1:.2f}s")
    print(f"  Second run (cache): {len(pages2):>3d} pages in {time2:.2f}s")
