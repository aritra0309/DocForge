# DocForge — Improvement Tasks for AI Agent

Source: static/code review comparing DocForge against patterns found in `activity-frames`. Each task below is scoped, verifiable, and ordered by priority. Give this file directly to a coding agent — each task has a clear "done when" condition.

---

## 1. [HIGH] Fix FAISS backend: post-filtering instead of pre-filtering breaks metadata search

**File:** `src/docforge/storage/backends/faiss.py` (method `search()`, helper `_matches_filters`)

**Problem:**
ChromaDB and Qdrant push metadata filters into the native query (`where=`, `query_filter=`), so their ANN search operates only within the filtered subset. FAISS instead runs an unfiltered brute-force top-`k` search first, then filters those `k` results in Python *after* retrieval. If a filter like `{"software": "postgresql"}` isn't well represented in the raw top-`k` by similarity, the call can return far fewer than `k` results — or zero — even though many matching chunks exist deeper in the index (rank 50, 200, etc.). This fails silently: no exception, no warning, just quietly worse results specific to FAISS.

**Fix (pick one):**
- **(a) Over-fetch and filter (recommended, minimal change):** In `FAISSBackend.search()`, when a filter is present, search for a larger candidate pool (e.g. `k * multiplier`, start with `multiplier=5`) — or loop, expanding the candidate pool geometrically — until either `k` filtered results are found or the full index has been searched. Return whatever is found; log a debug message if the index was exhausted before reaching `k`.
- **(b) Make the interface honest:** Add a `filters_are_native: bool` property (or similar) to the `VectorStore` ABC in `core/interfaces.py` so callers can detect approximate-filter backends and adjust behavior (e.g. request a larger `k` upstream).

**Recommended approach:** Do both — (a) fixes the actual bug, (b) prevents the same class of bug from recurring silently in a future backend (e.g. LanceDB, Weaviate).

**Done when:**
- A test with a sparse filter (e.g. seed 100 vectors, 5 matching a filter, request `k=10`) returns up to 10 matches on FAISS, not fewer than what ChromaDB/Qdrant return for the same data and filter.
- `VectorStore` ABC documents (via docstring or property) which backends are native-filtering vs. approximate.

---

## 2. [HIGH] Add capability-probing pattern across all 5 vector-store backends

**Files:** `src/docforge/core/interfaces.py`, `src/docforge/storage/backends/{chromadb,faiss,qdrant,lancedb,weaviate}.py`, `src/docforge/storage/engine.py`

**Problem:** DocForge's `VectorStore` ABC requires every backend to support the same feature surface (e.g. metadata filtering) with no way to probe what a given backend actually supports natively. This is the same failure shape as issue #1 — one implementation silently doesn't hold up the contract others do.

**Fix:**
- Add a `supports(feature: str) -> bool` method (or a set of explicit boolean properties, e.g. `filters_are_native`, `supports_hybrid_search`) to the `VectorStore` ABC.
- Implement it explicitly per backend rather than defaulting to `True` — force each backend file to declare its actual capabilities.
- In `storage/engine.py`, check `supports(...)` before relying on a feature, and either degrade gracefully (see #1) or raise a clear, typed error instead of failing silently.

**Done when:**
- Every backend under `storage/backends/` explicitly declares its capability flags (no inherited default of "supports everything").
- `storage/engine.py` has at least one code path that branches on a capability flag instead of assuming uniform behavior.

---

## 3. [MEDIUM] Verify classifier fallback handles unrecognized page types safely

**Files:** `src/docforge/classifier/taxonomy.py`, `src/docforge/classifier/rules.py`, `src/docforge/discovery/heuristics.py`

**Problem (to verify, not yet confirmed against real file contents):** Discovery and classification turn arbitrary crawled input (page structure/content) into typed categories. If there's no generic fallback bucket for unrecognized types, an unfamiliar page could crash the pipeline or silently drop instead of being classified as `"unknown"` and passed through.

**Fix:**
- Confirm `classifier/taxonomy.py` has an explicit fallback category (e.g. `PageType.UNKNOWN`) that `rules.py` returns when no rule matches, rather than raising or returning `None` unhandled.
- Add a unit test that feeds a deliberately nonsensical/unrecognized page into the classifier and asserts it returns the fallback type without raising.

**Done when:** A test exists proving unrecognized input classifies to a defined "unknown" bucket rather than raising or crashing the pipeline.

---

## 4. [MEDIUM] Add golden-output determinism tests for chunking strategies

**Files:** `src/docforge/chunker/strategies/{api_ref,base,code,heading,table,tutorial}.py`, new tests under `tests/unit/chunker/`

**Problem:** Chunk-boundary decisions are exactly the kind of logic that silently drifts as code evolves — a change to one strategy can shift chunk boundaries in ways that degrade retrieval quality months later without any test failing.

**Fix:**
- For each of the 5+ chunking strategies, add one "golden" test: a fixed input document → a committed expected output (chunk boundaries/content), checked into the test fixtures.
- Any future change to a strategy that alters chunk boundaries will fail the test and require an explicit, reviewed update to the golden fixture — turning silent drift into a visible, reviewed decision.

**Done when:** Each chunking strategy file has at least one corresponding golden-output test with a committed fixture.

---

## 5. [MEDIUM] Sync README "Status" section with actual code state

**File:** `README.md`

**Problem:** The README's Status section lists discovery, crawling, extraction, and indexing as "Next up," but the actual repo has fully built modules for all of them (`crawler/` — 5 files, `discovery/` — 6 files, `extractor/` — 6 files, `classifier/` — 3 files, `chunker/` — 6 files + 5 strategies, `embeddings/` — 3 files + 3 providers, `storage/` — 3 files + 3 backends, `metadata/` — 3 files), plus a built wheel in `dist/` (`docforge-0.1.0.dev0-py3-none-any.whl`) and matching unit/integration test suites for every module. Someone could `pip install` that wheel today and get a CLI that doesn't match what the README describes.

**Fix:**
- Update the README Status section to reflect the real state of each module (built/tested/not-yet-started).
- Add a lightweight CI check ("doc-freshness test") that greps the README's Status checklist against the actual module list in `src/docforge/` and fails if a claimed-incomplete module has non-trivial file content.

**Done when:**
- README accurately reflects current module status.
- A CI test fails if the README claims a module is unbuilt while real source files exist for it.

---

## 6. [LOW] Prevent version-string duplication drift

**Files:** `src/docforge/_version.py`, `pyproject.toml`, any CLI `--version` flag, docs

**Problem:** `pyproject.toml` correctly uses `[tool.hatch.version] source = "file"` pointing to `_version.py` as the single source of truth — this is good and should be preserved. But a hardcoded version string anywhere else (CLI `--version` output, docs, a server-info block) would silently drift from `_version.py` over time, the same bug class found in a prior project's `activity-frames` review.

**Fix:**
- Grep the codebase for `__version__` and literal version strings like `"0.1.0"` outside `_version.py` and its declared consumers.
- Ensure any place that needs to display the version imports it from `_version.py` rather than hardcoding it.

**Done when:** A repo-wide search confirms `_version.py` is the only literal version string; all other usages import it.

---

## 7. [LOW] Confirm read-only safety on metadata store reads

**File:** `src/docforge/storage/metadata_store.py`

**Problem (to verify):** Search-time metadata reads should never be capable of writes. Worth confirming there's a defensive guard (e.g. a read-only connection mode or explicit query type check) analogous to a `PRAGMA query_only`-style protection, so a bug in a read path can't accidentally mutate metadata.

**Fix:** Open read-only metadata connections in a read-only mode where the underlying DB supports it, or add an explicit assertion/guard against write operations in read-path code.

**Done when:** Metadata store read paths are demonstrably incapable of performing writes (test: attempt a write through the read-path connection/session and confirm it's rejected).

---

## Suggested implementation order for the agent
1. Task 1 (FAISS filtering bug) — real, user-facing correctness bug.
2. Task 2 (capability probing) — structural fix that also resolves the root cause of Task 1's bug class.
3. Task 5 (README sync) — cheap, prevents confusion for anyone using the repo right now.
4. Task 3 (classifier fallback) — verify and harden.
5. Task 4 (golden chunking tests) — regression protection.
6. Task 6 (version drift) — cheap grep-and-fix.
7. Task 7 (metadata read-only safety) — verify and harden.
