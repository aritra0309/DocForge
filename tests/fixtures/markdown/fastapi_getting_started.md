# First Steps

The simplest FastAPI file could look like this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Copy that to a file `main.py`.

## Run it

Run the live server with:

```bash
uvicorn main:app --reload
```

> **Note:** The --reload flag makes the server restart after code changes. Only use it for development.

## Check it

Open your browser at <http://127.0.0.1:8000>.

You will see the JSON response:

```json
{"message": "Hello World"}
```

## Interactive API docs

Now go to <http://127.0.0.1:8000/docs>.

You will see the automatic interactive API documentation (provided by Swagger UI).
