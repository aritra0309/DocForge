# Architecture

```mermaid
flowchart LR
    A[Discovery] --> B[Crawler]
    B --> C[Extractor]
    C --> D[Classifier]
    D --> E[Chunker]
    E --> F[Metadata]
    F --> G[Embeddings]
    G --> H[Vector store]
    H --> I[Semantic search]
```

Every stage has an interface in `docforge.core.interfaces`. Providers can replace discovery, fetching, content extraction, chunking, embeddings, or vector storage without changing pipeline callers.

```mermaid
flowchart TB
    Registry[Software registry] --> Discovery
    Config[Config: TOML, environment, defaults] --> Pipeline
    Pipeline --> MetadataStore[SQLite metadata store]
    Pipeline --> VectorStore[Selected vector backend]
    MetadataStore --> Updates[Incremental updates]
    VectorStore --> Search
```

`full` indexing writes pages and vectors. `incremental` compares saved page state and processes only changes. `reembed` keeps crawled chunks and replaces embeddings with a new model.
