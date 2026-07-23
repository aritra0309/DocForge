"""SQLite-backed HTML response cache with Gzip compression and TTL management."""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from docforge.core.models import FetchResult


class ResponseCache:
    """SQLite-backed cache for storing and retrieving HTTP FetchResults."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    url TEXT PRIMARY KEY,
                    status_code INTEGER NOT NULL,
                    headers TEXT NOT NULL,
                    html BLOB NOT NULL,
                    etag TEXT,
                    last_modified TEXT,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)

    def get(self, url: str) -> FetchResult | None:
        """Get cached FetchResult for url if present and not expired."""
        row = self._fetch_row(url)
        if row is None:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(UTC) >= expires_at:
            return None

        return self._row_to_fetch_result(row)

    def get_stale(self, url: str) -> FetchResult | None:
        """Get cached FetchResult even if expired (used for conditional request headers)."""
        row = self._fetch_row(url)
        if row is None:
            return None
        return self._row_to_fetch_result(row)

    def _fetch_row(self, url: str) -> dict[str, Any] | None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT url, status_code, headers, html, etag, last_modified, fetched_at, expires_at
            FROM response_cache
            WHERE url = ?
            """,
            (url,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "url": row[0],
            "status_code": row[1],
            "headers": row[2],
            "html": row[3],
            "etag": row[4],
            "last_modified": row[5],
            "fetched_at": row[6],
            "expires_at": row[7],
        }

    def _row_to_fetch_result(self, row: dict[str, Any]) -> FetchResult:
        html_bytes = row["html"]
        try:
            html_text = gzip.decompress(html_bytes).decode("utf-8")
        except Exception:
            html_text = html_bytes.decode("utf-8", errors="replace")

        headers = json.loads(row["headers"])
        fetched_at = datetime.fromisoformat(row["fetched_at"])

        return FetchResult(
            url=row["url"],
            status_code=row["status_code"],
            html=html_text,
            headers=headers,
            etag=row["etag"],
            last_modified=row["last_modified"],
            fetched_at=fetched_at,
        )

    def put(self, url: str, result: FetchResult, ttl_hours: int = 168) -> None:
        """Store FetchResult in cache with gzip-compressed HTML body and TTL."""
        compressed_html = gzip.compress(result.html.encode("utf-8"))
        headers_json = json.dumps(result.headers)
        fetched_at_str = result.fetched_at.isoformat()
        expires_at = result.fetched_at + timedelta(hours=ttl_hours)
        expires_at_str = expires_at.isoformat()

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO response_cache
                (url, status_code, headers, html, etag, last_modified, fetched_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url,
                    result.status_code,
                    headers_json,
                    compressed_html,
                    result.etag,
                    result.last_modified,
                    fetched_at_str,
                    expires_at_str,
                ),
            )

    def delete(self, url: str) -> None:
        """Remove a URL from the cache."""
        with self._conn:
            self._conn.execute("DELETE FROM response_cache WHERE url = ?", (url,))

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._conn:
            self._conn.execute("DELETE FROM response_cache")

    def close(self) -> None:
        """Close the SQLite database connection."""
        self._conn.close()
