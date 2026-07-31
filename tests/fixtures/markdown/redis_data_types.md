# Data Types

Redis is not a plain key-value store; it is actually a *data structures server*, supporting different kinds of values.

## Strings

The most basic Redis data structure is a string. Use SET and GET to store and retrieve the value of a string:

```redis
127.0.0.1:6379> SET mykey somevalue
OK
127.0.0.1:6379> GET mykey
"somevalue"
```

## Lists

Redis Lists are lists of strings, sorted by insertion order.

```redis
LPUSH mylist a
LPUSH mylist b
LRANGE mylist 0 -1
```

## Sets

Redis Sets are unordered collections of Strings.

```redis
SADD myset 1 2 3
SMEMBERS myset
```

## Hashes

Redis Hashes are maps between string fields and string values.

```redis
HSET user:1000 username antirez birthyear 1977 verified 1
HGET user:1000 username
```

## Sorted Sets

Redis Sorted Sets are similar to Sets, but every member is associated with a score (floating-point number).

```redis
ZADD myzset 1 "one"
ZADD myzset 2 "two"
ZRANGE myzset 0 -1 WITHSCORES
```
