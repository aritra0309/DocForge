# Strings

Redis strings are the most basic Redis data structure. They store a sequence of bytes.

> **Note:** String values can be at most 512 MB in length.

## Commands

### SET

Set the string value of a key.

```redis
SET key value [NX | XX] [GET] [EX seconds | PX milliseconds]
```

### GET

Get the string value of a key.

```redis
GET key
```

### INCR

Increment the integer value of a key by one.

```redis
INCR key
```

## Performance

Most string commands are O(1), which means they're very efficient.

| Command | Complexity |
| --- | --- |
| SET | O(1) |
| GET | O(1) |
| STRLEN | O(1) |

See also [Data Types Overview](https://redis.io/docs/latest/develop/data-types/strings/data-types.html).
