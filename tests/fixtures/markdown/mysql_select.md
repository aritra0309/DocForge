# SELECT Statement

SELECT is used to retrieve rows selected from one or more tables, and can include UNION statements and subqueries.

## Syntax

```sql
SELECT
    [ALL | DISTINCT | DISTINCTROW ]
    [HIGH_PRIORITY]
    [STRAIGHT_JOIN]
    select_expr [, select_expr] ...
    [FROM table_references
      [PARTITION partition_list]]
    [WHERE where_condition]
    [GROUP BY {col_name | expr | position}]
    [HAVING where_condition]
    [ORDER BY {col_name | expr | position}
      [ASC | DESC]]
    [LIMIT {[offset,] row_count | row_count OFFSET offset}]
```

## Examples

```sql
SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id;
```

```sql
SELECT a, b, a+b FROM t1;
```

## SELECT ... INTO

The SELECT ... INTO form of SELECT enables a query result to be stored in variables or written to a file.

> **Note:** SELECT INTO requires privileges appropriate to the target.

See also [INSERT](https://dev.mysql.com/doc/refman/8.4/en/insert.html), [UPDATE](https://dev.mysql.com/doc/refman/8.4/en/update.html).
