# Data Types

MySQL supports SQL data types in several categories: numeric types, date and time types, string (character and byte) types, spatial types, and the JSON data type.

## Numeric Data Types

| Type | Storage (Bytes) | Minimum Value (Signed) | Maximum Value (Signed) |
| --- | --- | --- | --- |
| TINYINT | 1 | -128 | 127 |
| SMALLINT | 2 | -32768 | 32767 |
| MEDIUMINT | 3 | -8388608 | 8388607 |
| INT | 4 | -2147483648 | 2147483647 |
| BIGINT | 8 | -2^63 | 2^63-1 |

## String Data Types

The string data types are CHAR, VARCHAR, BINARY, VARBINARY, BLOB, TEXT, ENUM, and SET.

```sql
CREATE TABLE example (
    name VARCHAR(255),
    description TEXT,
    status ENUM('active', 'inactive', 'pending')
);
```

## Date and Time Types

The date and time data types for representing temporal values are DATE, TIME, DATETIME, TIMESTAMP, and YEAR.

```sql
CREATE TABLE events (
    event_date DATE,
    event_time TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
