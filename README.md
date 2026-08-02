# DocForge

> Build a versioned, RAG-ready knowledge base from official software documentation.

[![CI](https://github.com/aritra0309/DocForge/actions/workflows/ci.yml/badge.svg)](https://github.com/aritra0309/DocForge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/aritra0309/DocForge.svg)](LICENSE)

DocForge discovers documentation, crawls pages, extracts clean Markdown, classifies and chunks content, creates embeddings, and stores vectors for semantic search. It supports versioned indexes, incremental updates, and replaceable pipeline providers.


## Features

- Zero-config discovery for PostgreSQL, MySQL, MongoDB, FastAPI, React, Kubernetes, and Redis
- Version-aware indexing and filtered semantic search
- Type-aware chunking for guides, tutorials, API references, code, and tables
- Incremental updates and re-embedding without a fresh crawl
- Embedding providers: Sentence Transformers, OpenAI, Voyage, BGE, and Jina
- Vector backends: ChromaDB, FAISS, Qdrant, LanceDB, and Weaviate
- CLI, async Python API, custom plugin contracts, and MkDocs documentation

## Quickstart

Requires Python 3.11+ and internet access. First index downloads default local embedding model and fetches documentation pages.

```bash
git clone https://github.com/aritra0309/DocForge.git
cd DocForge

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

docforge index postgresql --version 17
docforge search "how to create an index" --software postgresql --version 17 --top-k 5
```

DocForge stores local data in `~/.docforge` by default.

## Python API

```python
import asyncio

from docforge import DocForge


async def main() -> None:
    async with DocForge() as forge:
        await forge.index("postgresql", version="17")
        results = await forge.search(
            "how to create an index",
            software="postgresql",
            version="17",
            k=5,
        )

    for result in results:
        print(f"{result.metadata.title}: {result.metadata.url}")


asyncio.run(main())
```

## Configuration

Create `docforge.toml` in project directory. Environment variables with `DOCFORGE_` prefix override file values.

```toml
[crawler]
max_pages_per_version = 500
rate_limit_rps = 2

[storage]
backend = "chromadb"
path = "~/.docforge/vectordb"
```

```bash
export DOCFORGE_CRAWLER__MAX_PAGES_PER_VERSION=100
export DOCFORGE_STORAGE__PATH="$PWD/.docforge/vectordb"
docforge config --source
```

## Common commands

```bash
docforge index redis
docforge update postgresql
docforge reembed postgresql --model BAAI/bge-small-en-v1.5
docforge list
docforge stats postgresql
docforge delete postgresql --version 17
```

Run `docforge --help` or `docforge COMMAND --help` for all options.

## Documentation

📖 **Live Documentation:** [DocForge Documentation](https://aritra0309.github.io/DocForge/)

## Development

```bash
pip install -e ".[dev,docs]"

make lint
make typecheck
make test
make docs-build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull-request process and registry contributions.

## License

Apache-2.0. See [LICENSE](LICENSE).
