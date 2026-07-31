# CREATE INDEX

CREATE INDEX constructs an index on the specified column(s) of a table.

> **Note:** Indexes can significantly improve query performance.

## Syntax

```sql
CREATE INDEX index_name ON table_name (column_name);
```

## Parameters

| Parameter | Description |
| --- | --- |
| index\_name | Name of the index to create |
| table\_name | Name of the table |

See also [DROP INDEX](https://www.postgresql.org/docs/17/sql/drop-index.html).
