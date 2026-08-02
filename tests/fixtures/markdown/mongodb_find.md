# Find Documents

This page describes how to select or query for documents in a collection.

## db.collection.find()

The `find()` method returns a cursor to the documents that match the query criteria.

### Syntax

```javascript
db.collection.find(query, projection, options)
```

### Parameters

**query**
: Optional. Specifies selection filter using query operators.

**projection**
: Optional. Specifies the fields to return in the documents that match the query filter.

**options**
: Optional. Additional options for the query.

## Examples

### Select all documents

```javascript
db.inventory.find({})
```

### Specify equality condition

```javascript
db.inventory.find({ status: "D" })
```

### Specify conditions using query operators

```javascript
db.inventory.find({ status: { $in: [ "A", "D" ] } })
```

> **Important:** Starting in MongoDB 4.4, you can specify a projection expression as a field argument to the find() method.

For more information, see [Query Documents](https://www.mongodb.com/docs/manual/tutorial/query-documents/query-documents.html).
