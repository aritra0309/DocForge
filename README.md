# DocForge

> **Automatically discovers, crawls, versions, chunks, and indexes official software documentation into a RAG-ready knowledge base.**

[![CI](https://github.com/aritra0309/DocForge/actions/workflows/ci.yml/badge.svg)](https://github.com/aritra0309/DocForge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/github/license/aritra0309/DocForge.svg)](https://github.com/aritra0309/DocForge/blob/main/LICENSE)

## Status

Phases 1–4 of the [implementation plan](plan.md) are complete — Tasks 1–17 are built and tested:

| Phase | Tasks | Status |
|-------|-------|--------|
| 1 — Foundation | 1–5 | ✅ Project scaffold, data models, interfaces, config, software registry |
| 2 — Pipeline Core | 6–11 | ✅ Discovery, crawling, extraction, classification, chunking, metadata generation |
| 3 — Intelligence | 12–14 | ✅ Embeddings (3 providers), storage (3 backends), end-to-end pipeline orchestrator |
| 4 — Incremental & Providers | 15–17 | ✅ Incremental updates, version management, 5 embedding providers + 5 vector backends |

**Embedding providers:** Sentence Transformers, OpenAI, Voyage, BGE (with presets), Jina  
**Vector backends:** ChromaDB, FAISS, Qdrant, LanceDB, Weaviate  
**Pipeline modes:** `full`, `incremental`, `reembed`

Next up: Phase 5 — CLI & Python API, full test suite, documentation, and v0.1.0 release.

## Quickstart

```bash
# Install
pip install docforge

# Index PostgreSQL documentation
docforge index postgresql

# Search
docforge search "how to create an index in postgresql"
```

## Python API

```python
import asyncio
from docforge import DocForge

async def main():
    forge = DocForge()
    await forge.index("postgresql")
    results = await forge.search(
        "how to create an index",
        software="postgresql",
        version="latest",
        k=10
    )
    for r in results:
        print(f"{r.metadata.title}: {r.content[:200]}...")

asyncio.run(main())
```

## Features

- **Zero-config discovery** — Give it a name (`postgresql`, `fastapi`, `redis`), it finds the docs
- **Multi-version support** — Index v16, v17, v18 side-by-side; query with `version="latest"`
- **Type-aware chunking** — API reference, tutorials, examples each get their own strategy
- **Pluggable embeddings** — Sentence Transformers (local), OpenAI, Voyage, Jina, BGE
- **Pluggable vector stores** — ChromaDB (default), FAISS, Qdrant, LanceDB, Weaviate
- **Incremental updates** — `docforge update` only re-indexes changed pages
- **Rich CLI & Python API** — Progress bars, formatted output, event hooks

## Installation

```bash
# Core dependencies only
pip install docforge

# With development dependencies
pip install "docforge[dev]"

# With optional providers
pip install "docforge[openai,qdrant,faiss]"
```

## Configuration

Create a `docforge.toml` in your project or `~/.config/docforge/config.toml`:

```toml
[general]
data_dir = "~/.docforge"
log_level = "INFO"
parallelism = 8

[crawler]
max_pages_per_version = 5000
rate_limit_rps = 5
timeout_seconds = 30

[chunker]
target_chunk_size = 512
max_chunk_size = 1024
overlap_tokens = 64

[embeddings]
provider = "sentence-transformers"
model = "BAAI/bge-base-en-v1.5"
batch_size = 64

[storage]
backend = "chromadb"
path = "~/.docforge/vectordb"
```

Environment variables override config file (`DOCFORGE_*` prefix):
```bash
export DOCFORGE_EMBEDDINGS__PROVIDER=openai
export DOCFORGE_EMBEDDINGS__MODEL=text-embedding-3-small
export DOCFORGE_OPENAI_API_KEY=sk-...
```

## Supported Software (Registry)

| Software | Versions |
|----------|----------|
| PostgreSQL | 13–17 |
| MySQL | 5.7, 8.0–8.4 |
| MongoDB | 4.4–7.0 |
| FastAPI | latest |
| React | latest |
| Kubernetes | 1.27–1.31 |
| Redis | 6.2, 7.0–7.4, latest |

Add more via `registry/software/*.yaml`.

## CLI Commands

| Command | Description |
|---------|-------------|
| `docforge index <software>` | Full index pipeline |
| `docforge search <query>` | Semantic search |
| `docforge update [software]` | Incremental update |
| `docforge reembed <software>` | Re-embed with new model |
| `docforge list` | List indexed software |
| `docforge stats [software]` | Show statistics |
| `docforge delete <software>` | Delete indexed data |
| `docforge config` | Show current configuration |

## Architecture

```
discover → crawl → extract → classify → chunk → embed → store
              ↑                                              │
              └────────────── incremental ──────────────────┘
```

Each stage is a pluggable interface — swap in your own crawler, extractor, chunker, embedding provider, or vector store.

## Development

```bash
# Clone and setup
git clone https://github.com/aritra0309/DocForge
cd docforge
make dev-install

# Run checks
make lint
make typecheck
make test

# Build
make build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Development setup
- Branch naming & PR process
- Adding new software to the registry
- Writing custom plugins

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.