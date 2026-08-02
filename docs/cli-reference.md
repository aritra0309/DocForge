# CLI reference

Run `docforge --help` or `docforge COMMAND --help` for installed version details.

## Index

```bash
docforge index postgresql
docforge index postgresql --version 17
docforge index postgresql --mode incremental
```

## Search

`--software` is required so DocForge can select matching vector collection.

```bash
docforge search "connection pooling" --software postgresql --top-k 5
docforge search "hooks" --software react --version latest
```

## Maintain an index

```bash
docforge update postgresql
docforge update
docforge reembed postgresql --model BAAI/bge-small-en-v1.5
docforge list
docforge stats postgresql
docforge delete postgresql --version 17
docforge delete postgresql --force
```

## Configuration and version

```bash
docforge config
docforge config --source
docforge --config ./docforge.toml index redis
docforge --version
```
