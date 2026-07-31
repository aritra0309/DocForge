# Routing

FastAPI uses Python type hints to define path operations (routes).

> **Tip:** Each path operation decorator accepts path parameters defined with curly braces.

## Path Operations

A *path operation* is a function decorated with one of FastAPI's HTTP method decorators.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## Path Parameters

Path parameters are declared in the path string using curly braces and matched to function parameters.

```python
@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

## Query Parameters

Function parameters not part of the path are treated as query parameters.

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 100):
    return {"skip": skip, "limit": limit}
```

## Request Body

Use Pydantic models to declare a request body.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return item
```

See also [Query Parameters](https://fastapi.tiangolo.com/tutorial/path-params/query-params.html) and [Request Body](https://fastapi.tiangolo.com/tutorial/path-params/body.html).
