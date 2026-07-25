"""SQLite-backed embedding cache.

Keyed by ``(model_name, content_hash)`` so that re-embedding the same
content with the same model returns the cached vector without an API call.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class EmbeddingCache:
    """Persistent, thread-safe SQLite cache for embedding vectors.

    Each entry is identified by ``(model_name, content_hash)``.
    Vectors are stored as pickle blobs for compactness and speed.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embedding_cache ("
                "model_name TEXT NOT NULL, "
                "content_hash TEXT NOT NULL, "
                "vector BLOB NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
                "PRIMARY KEY (model_name, content_hash)"
                ")"
            )
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def get(self, model_name: str, content_hash: str) -> list[float] | None:
        """Look up a cached embedding vector.

        Returns the vector if found, or ``None`` on cache miss.
        """
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT vector FROM embedding_cache "
                "WHERE model_name = ? AND content_hash = ?",
                (model_name, content_hash),
            ).fetchone()
            if row is None:
                return None
            result: list[float] = pickle.loads(row[0])
            return result

    def put(self, model_name: str, content_hash: str, vector: list[float]) -> None:
        """Store an embedding vector in the cache.

        Uses INSERT OR REPLACE so repeated calls for the same key update
        the existing entry.
        """
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache "
                "(model_name, content_hash, vector) VALUES (?, ?, ?)",
                (model_name, content_hash, pickle.dumps(vector)),
            )
            conn.commit()

    def put_batch(
        self, model_name: str, keys: list[str], vectors: list[list[float]]
    ) -> None:
        """Store multiple embedding vectors in a single transaction."""
        with self._lock:
            conn = self._get_conn()
            rows = [
                (model_name, key, pickle.dumps(vec))
                for key, vec in zip(keys, vectors, strict=False)
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache "
                "(model_name, content_hash, vector) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()

    def clear(self, model_name: str | None = None) -> None:
        """Clear cache entries, optionally filtered by model name."""
        with self._lock:
            conn = self._get_conn()
            if model_name:
                conn.execute(
                    "DELETE FROM embedding_cache WHERE model_name = ?",
                    (model_name,),
                )
            else:
                conn.execute("DELETE FROM embedding_cache")
            conn.commit()

    def count(self, model_name: str | None = None) -> int:
        """Return the number of cached entries, optionally filtered."""
        with self._lock:
            conn = self._get_conn()
            if model_name:
                row = conn.execute(
                    "SELECT COUNT(*) FROM embedding_cache WHERE model_name = ?",
                    (model_name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM embedding_cache"
                ).fetchone()
            result: int = row[0] if row else 0
            return result

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = Lock()
        self._init_db()


__all__ = ["EmbeddingCache"]
