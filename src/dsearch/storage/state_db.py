"""SQLite store for a node's applied seq and its latest snapshot (D-010, D-011).

Tables:
  meta(key, value)                 -- 'applied_seq', 'snapshot_seq'
  snapshot_docs(doc_id, version, text)  -- text NULL = tombstone

The `*_sync` methods block; async callers use the wrappers, which run them in a
worker thread. A threading lock serialises use of the single connection.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

from dsearch.index.inverted_index import DocState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot_docs (
    doc_id  TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    text    TEXT
);
"""


class StateDB:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    @classmethod
    def open_sync(cls, path: Path) -> "StateDB":
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript(_SCHEMA)
        return cls(conn)

    @classmethod
    async def open(cls, path: Path) -> "StateDB":
        return await asyncio.to_thread(cls.open_sync, path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- applied seq ----------------------------------------------------

    def applied_seq_sync(self) -> int:
        return self._get_meta("applied_seq")

    def set_applied_seq_sync(self, seq: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('applied_seq', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (seq,),
            )

    async def set_applied_seq(self, seq: int) -> None:
        await asyncio.to_thread(self.set_applied_seq_sync, seq)

    # ---- snapshots ------------------------------------------------------

    def snapshot_seq_sync(self) -> int:
        return self._get_meta("snapshot_seq")

    def write_snapshot_sync(self, seq: int, docs: dict[str, DocState]) -> None:
        """Replace the stored snapshot with `docs` as of `seq`, in one transaction."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM snapshot_docs")
                conn.executemany(
                    "INSERT INTO snapshot_docs(doc_id, version, text) VALUES (?, ?, ?)",
                    ((doc_id, d.version, d.text) for doc_id, d in docs.items()),
                )
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('snapshot_seq', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (seq,),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    async def write_snapshot(self, seq: int, docs: dict[str, DocState]) -> None:
        await asyncio.to_thread(self.write_snapshot_sync, seq, docs)

    def load_snapshot_sync(self) -> tuple[int, dict[str, DocState]]:
        """Return (snapshot_seq, docs); (0, {}) if no snapshot was ever taken."""
        with self._lock:
            rows = self._conn.execute("SELECT doc_id, version, text FROM snapshot_docs").fetchall()
            seq = self._get_meta_locked("snapshot_seq")
        return seq, {d: DocState(v, t) for d, v, t in rows}

    def _get_meta(self, key: str) -> int:
        with self._lock:
            return self._get_meta_locked(key)

    def _get_meta_locked(self, key: str) -> int:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return 0 if row is None else int(row[0])
