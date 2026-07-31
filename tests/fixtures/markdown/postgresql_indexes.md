# Indexes

Indexes are a common way to enhance database performance. An index allows the database server to find and retrieve specific rows much faster than it could do without an index.

## Index Types

PostgreSQL provides several index types:

- **B-tree** — the default index type, can handle equality and range queries.
- **Hash** — can only handle simple equality comparisons.
- **GiST** — a framework that allows implementation of many different indexing strategies.
- **SP-GiST** — supports partitioned search trees.
- **GIN** — suitable for composite values (arrays, full-text search).
- **BRIN** — block range indexes, work well for naturally ordered large tables.

## Creating an Index

```sql
CREATE INDEX test1_id_index ON test1 (id);
```

## Multicolumn Indexes

An index can be defined on more than one column of a table.

```sql
CREATE INDEX test2_mm_idx ON test2 (major, minor);
```

## Unique Indexes

```sql
CREATE UNIQUE INDEX title_idx ON films (title);
```

## Partial Indexes

A partial index is an index built over a subset of a table.

```sql
CREATE INDEX orders_unbilled_index ON orders (order_nr)
    WHERE billed IS NOT true;
```
