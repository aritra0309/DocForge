# DocForge

DocForge discovers, crawls, versions, chunks, and indexes official software documentation into a RAG-ready knowledge base.

## Install

```bash
pip install docforge
```

For local development:

```bash
git clone https://github.com/aritra0309/DocForge.git
cd DocForge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## 30-second quickstart

```bash
docforge index postgresql
docforge search "how to create an index" --software postgresql --top-k 5
```

DocForge stores data under `~/.docforge` by default. First local embedding run downloads configured sentence-transformers model.

## Next

- [First index and search](getting-started.md)
- [CLI commands](cli-reference.md)
- [Python API](api-reference.md)
- [Custom plugins](plugins.md)
