# Request Body

When you need to send data from a client (let's say, a browser) to your API, you send it as a **request body**.

A **request body** is data sent by the client to your API. A **response body** is the data your API sends to the client.

## Import Pydantic's BaseModel

First, you need to import `BaseModel` from `pydantic`:

```python
from pydantic import BaseModel
```

## Create your data model

```python
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
```

## Declare it as a parameter

```python
@app.post("/items/")
async def create_item(item: Item):
    return item
```

> **Tip:** If you declare the parameter with type Item, FastAPI will automatically read the body as JSON.

## Results

With just that Python type declaration, FastAPI will:

- Read the body of the request as JSON.
- Convert the corresponding types.
- Validate the data.
- Give you the received data in the parameter `item`.
