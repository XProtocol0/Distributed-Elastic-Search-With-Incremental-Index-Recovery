"""One node's durable copy of a shard: update log + SQLite state + in-memory index.

Used by both primaries and replicas (D-010). Every write follows the
durability order of Invariant 4:

    1. append the entry to the update log and fsync
    2. apply it to the in-memory index
    3. persist applied_seq

so the persisted applied_seq can never be ahead of what is in the index.
On open, the index is rebuilt from the latest snapshot plus a replay of the
local log after it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from dsearch.index.inverted_index import InvertedIndex
from dsearch.index.operation import ApplyResult
from dsearch.proto import search_pb2
from dsearch.storage.log_entry import entry_to_operation
from dsearch.storage.state_db import StateDB
from dsearch.storage.update_log import UpdateLog

log = logging.getLogger(__name__)

LOG_FILE = "update.log"
DB_FILE = "state.db"


class CorruptStateError(Exception):
    """Persisted state breaks snapshot_seq <= applied_seq <= last logged seq."""


class StoreFailedError(Exception):
    """A write failed part-way; the node must restart to recover from disk."""


@dataclass(frozen=True)
class OpenReport:
    """What `ShardStore.open` found and replayed, for logs and tests."""

    snapshot_seq: int
    persisted_applied_seq: int
    last_logged_seq: int
    replayed: int


class ShardStore:
    def __init__(
        self,
        update_log: UpdateLog,
        db: StateDB,
        index: InvertedIndex,
        snapshot_seq: int,
        snapshot_every: int,
        report: OpenReport,
    ) -> None:
        self.update_log = update_log
        self.db = db
        self.index = index
        self.applied_seq = update_log.last_seq
        self.snapshot_seq = snapshot_seq
        self.snapshot_every = snapshot_every
        self.open_report = report
        self._write_lock = asyncio.Lock()  # one write at a time, in seq order (D-010)
        self._failed = False

    @classmethod
    async def open(cls, data_dir: Path, snapshot_every: int = 0) -> "ShardStore":
        """Open the node's store, rebuilding the index from snapshot + local log replay."""
        update_log = await UpdateLog.open(data_dir / LOG_FILE)
        db = await StateDB.open(data_dir / DB_FILE)
        snapshot_seq, docs = await asyncio.to_thread(db.load_snapshot_sync)
        persisted = await asyncio.to_thread(db.applied_seq_sync)
        last = update_log.last_seq

        if not snapshot_seq <= persisted <= last:
            update_log.close()
            db.close()
            raise CorruptStateError(
                f"{data_dir}: expected snapshot_seq <= applied_seq <= last logged seq, got "
                f"snapshot_seq={snapshot_seq} applied_seq={persisted} last_logged_seq={last}"
            )

        index = InvertedIndex.from_state(docs)
        replayed = 0
        async for entry in update_log.read_after(snapshot_seq):
            result = index.apply(entry_to_operation(entry))
            log.debug("replay seq=%d doc=%s v=%d -> %s", entry.seq, entry.doc_id, entry.version, result.value)
            replayed += 1
        await db.set_applied_seq(last)

        report = OpenReport(snapshot_seq, persisted, last, replayed)
        log.info(
            "opened %s: snapshot_seq=%d persisted applied_seq=%d replayed seq=%d..%d (%d entries)",
            data_dir, snapshot_seq, persisted, snapshot_seq + 1, last, replayed,
        )
        return cls(update_log, db, index, snapshot_seq, snapshot_every, report)

    async def write(self, entry: search_pb2.LogEntry) -> ApplyResult:
        """Durably log `entry`, apply it and record it as applied.

        `entry.seq` must be exactly `applied_seq + 1`. Any failure part-way
        leaves the store failed (crash-stop); reopening recovers from disk.
        """
        op = entry_to_operation(entry)  # reject malformed entries before logging
        async with self._write_lock:
            if self._failed:
                raise StoreFailedError("store failed earlier; restart the node")
            if entry.seq != self.applied_seq + 1:
                raise ValueError(f"write seq={entry.seq}, expected seq={self.applied_seq + 1}")
            try:
                await self.update_log.append(entry)  # 1. log + fsync
                result = self.index.apply(op)  # 2. apply (sync, atomic for search)
                await self.db.set_applied_seq(entry.seq)  # 3. persist position
            except BaseException:
                self._failed = True
                log.exception("write seq=%d failed; store is now failed", entry.seq)
                raise
            self.applied_seq = entry.seq
            log.debug(
                "applied seq=%d %s doc=%s v=%d -> %s",
                entry.seq, op.op_type.value, op.doc_id, op.version, result.value,
            )
            if self.snapshot_every and self.applied_seq - self.snapshot_seq >= self.snapshot_every:
                try:
                    await self._snapshot_locked()
                except Exception:
                    # The write is already durable; a missed snapshot only means a longer replay.
                    log.exception("snapshot at seq=%d failed", self.applied_seq)
            return result

    async def snapshot(self) -> None:
        """Write a full snapshot of the index as of the current applied_seq."""
        async with self._write_lock:
            await self._snapshot_locked()

    async def _snapshot_locked(self) -> None:
        seq = self.applied_seq
        docs = self.index.state()  # copied on the event loop: consistent with seq
        await self.db.write_snapshot(seq, docs)
        self.snapshot_seq = seq
        log.info("snapshot at seq=%d (%d docs)", seq, len(docs))

    async def close(self) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self.update_log.close)
            await asyncio.to_thread(self.db.close)
