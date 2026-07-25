"""SQLite-backed state tracker for pipeline runs, indexed software, and page state."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


class MetadataStore:
    """Tracks what has been indexed and with what configuration.

    Tables:
        - ``indexed_software`` — software entries with their config snapshots
        - ``indexed_versions`` — per-version stats (page count, chunk count, model, timestamps)
        - ``page_state`` — per-page crawl state (URL, content hash, ETag, last crawled)
        - ``pipeline_runs`` — run history with status, mode, and error log
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
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS indexed_software (
                    software TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    config_snapshot TEXT NOT NULL DEFAULT '{}',
                    last_indexed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS indexed_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    software TEXT NOT NULL,
                    version TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    embedding_dimension INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT NOT NULL,
                    UNIQUE(software, version)
                );

                CREATE TABLE IF NOT EXISTS page_state (
                    url TEXT PRIMARY KEY,
                    software TEXT NOT NULL,
                    version TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT '',
                    etag TEXT NOT NULL DEFAULT '',
                    last_modified TEXT NOT NULL DEFAULT '',
                    last_crawled_at TEXT
                );

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    software TEXT NOT NULL,
                    version TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'full',
                    status TEXT NOT NULL DEFAULT 'running',
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    error_log TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_versions_software
                    ON indexed_versions(software);
                CREATE INDEX IF NOT EXISTS idx_page_state_software_version
                    ON page_state(software, version);
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_software
                    ON pipeline_runs(software);
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ------------------------------------------------------------------
    # indexed_software
    # ------------------------------------------------------------------

    def upsert_software(
        self, software: str, display_name: str, config_snapshot: str | None = None
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO indexed_software "
                "(software, display_name, config_snapshot, last_indexed_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    software,
                    display_name,
                    config_snapshot or "{}",
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()

    def get_software(self, software: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM indexed_software WHERE software = ?", (software,)
            ).fetchone()
            return dict(row) if row else None

    def list_software(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT * FROM indexed_software ORDER BY software").fetchall()
            return [dict(r) for r in rows]

    def delete_software(self, software: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM indexed_software WHERE software = ?", (software,))
            conn.execute("DELETE FROM indexed_versions WHERE software = ?", (software,))
            conn.execute("DELETE FROM page_state WHERE software = ?", (software,))
            conn.commit()

    # ------------------------------------------------------------------
    # indexed_versions
    # ------------------------------------------------------------------

    def upsert_version(
        self,
        software: str,
        version: str,
        page_count: int = 0,
        chunk_count: int = 0,
        embedding_model: str = "",
        embedding_dimension: int = 0,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO indexed_versions "
                "(software, version, page_count, chunk_count, "
                "embedding_model, embedding_dimension, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    software,
                    version,
                    page_count,
                    chunk_count,
                    embedding_model,
                    embedding_dimension,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()

    def get_version(self, software: str, version: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM indexed_versions WHERE software = ? AND version = ?",
                (software, version),
            ).fetchone()
            return dict(row) if row else None

    def list_versions(self, software: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM indexed_versions WHERE software = ? ORDER BY version",
                (software,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_version(self, software: str, version: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM indexed_versions WHERE software = ? AND version = ?",
                (software, version),
            )
            conn.execute(
                "DELETE FROM page_state WHERE software = ? AND version = ?",
                (software, version),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # page_state
    # ------------------------------------------------------------------

    def upsert_page_state(
        self,
        url: str,
        software: str,
        version: str,
        content_hash: str = "",
        etag: str = "",
        last_modified: str = "",
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO page_state "
                "(url, software, version, content_hash, etag, last_modified, last_crawled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    software,
                    version,
                    content_hash,
                    etag,
                    last_modified,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()

    def get_page_state(self, url: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM page_state WHERE url = ?", (url,)).fetchone()
            return dict(row) if row else None

    def list_page_states(self, software: str, version: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM page_state WHERE software = ? AND version = ?",
                (software, version),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_page_state(self, url: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM page_state WHERE url = ?", (url,))
            conn.commit()

    # ------------------------------------------------------------------
    # pipeline_runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        software: str,
        version: str,
        mode: str = "full",
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "INSERT INTO pipeline_runs "
                "(software, version, mode, status, started_at) "
                "VALUES (?, ?, ?, 'running', ?)",
                (software, version, mode, datetime.now(UTC).isoformat()),
            )
            conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def complete_run(
        self,
        run_id: int,
        status: str = "completed",
        page_count: int = 0,
        chunk_count: int = 0,
        embedding_model: str = "",
        error_log: str = "",
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE pipeline_runs SET "
                "status = ?, page_count = ?, chunk_count = ?, "
                "embedding_model = ?, error_log = ?, completed_at = ? "
                "WHERE id = ?",
                (
                    status,
                    page_count,
                    chunk_count,
                    embedding_model,
                    error_log,
                    datetime.now(UTC).isoformat(),
                    run_id,
                ),
            )
            conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self, software: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            if software:
                rows = conn.execute(
                    "SELECT * FROM pipeline_runs WHERE software = ? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (software, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # stats helpers
    # ------------------------------------------------------------------

    def get_software_stats(self, software: str) -> dict[str, Any]:
        with self._lock:
            conn = self._get_conn()
            versions = conn.execute(
                "SELECT COUNT(*) as cnt FROM indexed_versions WHERE software = ?",
                (software,),
            ).fetchone()
            pages = conn.execute(
                "SELECT COUNT(*) as cnt FROM page_state WHERE software = ?",
                (software,),
            ).fetchone()
            chunks = conn.execute(
                "SELECT SUM(chunk_count) as total FROM indexed_versions WHERE software = ?",
                (software,),
            ).fetchone()
            return {
                "software": software,
                "version_count": versions["cnt"] if versions else 0,
                "page_count": pages["cnt"] if pages else 0,
                "chunk_count": chunks["total"] or 0,
            }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
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


__all__ = ["MetadataStore"]
