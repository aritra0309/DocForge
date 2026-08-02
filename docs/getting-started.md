# Getting started

## 1. Install

```bash
pip install docforge
```

Use optional extras for remote services, for example `pip install "docforge[openai,qdrant]"`.

## 2. Index documentation

```bash
docforge index postgresql --version 17
```

Omit `--version` to index registry latest version. Indexing fetches site pages, converts them to Markdown, produces chunks, embeds chunks, and writes vectors locally.

## 3. Search it

```bash
docforge search "create a B-tree index" --software postgresql --version 17 --top-k 5
```

## 4. Inspect indexed data

```bash
docforge list
docforge stats postgresql
```

## Configuration

Create `docforge.toml` in project directory:

```toml
[general]
data_dir = "~/.docforge"

[crawler]
max_pages_per_version = 500
rate_limit_rps = 2

[storage]
backend = "chromadb"
path = "~/.docforge/vectordb"
```

Environment variables override files:

```bash
export DOCFORGE_CRAWLER__MAX_PAGES_PER_VERSION=100
export DOCFORGE_STORAGE__PATH="$PWD/.docforge/vectordb"
```

Run `docforge config --source` to inspect active values.
