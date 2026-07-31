# Tutorial

Welcome to PostgreSQL. The following few chapters are intended to give a simple introduction to PostgreSQL, relational database concepts, and the SQL language to those who are new to any one of these aspects.

## 1. Getting Started

Before you can use PostgreSQL you need to install it. It is possible that PostgreSQL is already installed at your site, either because it was included in your operating system distribution or because the system administrator already installed it.

## 2. The SQL Language

SQL is the standard language for interacting with relational databases.

### 2.1. Introduction

This chapter provides an overview of how to use SQL to perform simple operations.

```sql
CREATE TABLE weather (
    city            varchar(80),
    temp_lo         int,           -- low temperature
    temp_hi         int,           -- high temperature
    prcp            real,          -- precipitation
    date            date
);
```

### 2.2. Populating a Table With Rows

```sql
INSERT INTO weather VALUES ('San Francisco', 46, 50, 0.25, '1994-11-27');
INSERT INTO weather (city, temp_lo, temp_hi, prcp, date)
    VALUES ('San Francisco', 43, 57, 0.0, '1994-11-29');
```

### 2.3. Querying a Table

```sql
SELECT * FROM weather;
```

```sql
SELECT city, temp_lo, temp_hi, prcp, date FROM weather;
```

Continue to [Chapter 3. Advanced Features](https://www.postgresql.org/docs/17/advanced.html).
