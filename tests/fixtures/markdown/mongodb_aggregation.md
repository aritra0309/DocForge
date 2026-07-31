# Aggregation Pipeline

An aggregation pipeline consists of one or more stages that process documents.

Each stage performs an operation on the input documents and passes the results to the next stage.

## Pipeline Stages

### $match

Filters the documents to pass only the documents that match the specified condition(s).

```javascript
{ $match: { status: "A" } }
```

### $group

Groups input documents by the specified \_id expression and for each distinct grouping, outputs a document.

```javascript
{ $group : { _id : "$author", books: { $push: "$title" } } }
```

### $sort

```javascript
{ $sort: { count: -1, title: 1 } }
```

## Example Pipeline

```javascript
db.orders.aggregate([
   { $match: { status: "A" } },
   { $group: { _id: "$cust_id", total: { $sum: "$amount" } } }
])
```

> **Note:** Starting in MongoDB 4.2, you can use aggregation expressions in more contexts.
