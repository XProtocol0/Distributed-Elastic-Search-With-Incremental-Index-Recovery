"""One node's durable copy of a shard: update log + SQLite state + in-memory index.

Used by both primaries and replicas (D-010). Every write follows the
durability order of Invariant 4:

    1. append the entry to the update log and fsync
    2. apply it to the in-memory index
    3. persist applied_seq

so the persisted applied_seq can never be ahead of what is in the index.
On open, the index is rebuilt from the latest snapshot plus a replay of the
local log after it.

With `log_retention` > 0, each snapshot is followed by dropping log entries
more than `log_retention` behind it (D-030). A replica whose needed range is
gone is rebuilt with `install_snapshot` (D-031).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from dsearch.index.inverted_index import DocState, InvertedIndex
from dsearch.index.operation import ApplyResult, Operation
from dsearch.proto import search_pb2
from dsearch.storage.log_entry import entry_to_operation, make_entry
from dsearch.storage.state_db import StateDB
from dsearch.storage.update_log import UpdateLog

log = logging.getLogger(__name__)

LOG_FILE = "update.log"
DB_FILE = "state.db"


class CorruptStateError(Exception):
    """Persisted state breaks snapshot_seq <= applied_seq <= last logged seq."""


class SeqMismatchError(ValueError):
    """A write's seq is not applied_seq + 1."""


class StoreFailedError(Exception):
    """A write failed part-way; the node must restart to recover from disk."""


@dataclass(frozen=True)
class OpenReport:
    """What `ShardStore.open` found and replayed, for logs and tests."""

    snapshot_seq: int
    persisted_applied_seq: int
    last_logged_seq: int
    replayed: int
    seconds: float = 0.0


class ShardStore:
    def __init__(
        self,
        update_log: UpdateLog,
        db: StateDB,
        index: InvertedIndex,
        snapshot_seq: int,
        snapshot_every: int,
        report: OpenReport,
        log_retention: int = 0,
    ) -> None:
        self.update_log = update_log
        self.db = db
        self.index = index
        self.applied_seq = update_log.last_seq
        self.snapshot_seq = snapshot_seq
        self.snapshot_every = snapshot_every
        self.log_retention = log_retention  # 0 = never truncate (D-030)
        self.open_report = report
        self._write_lock = asyncio.Lock()  # one write at a time, in seq order (D-010)
        self._applied_changed = asyncio.Condition()  # notified after each write (D-014)
        self._failed = False

    @classmethod
    async def open(cls, data_dir: Path, snapshot_every: int = 0, log_retention: int = 0) -> "ShardStore":
        """Open the node's store, rebuilding the index from snapshot + local log replay."""
        started = time.monotonic()
        db = await StateDB.open(data_dir / DB_FILE)
        snapshot_seq, docs = await asyncio.to_thread(db.load_snapshot_sync)
        persisted = await asyncio.to_thread(db.applied_seq_sync)
        update_log = await UpdateLog.open(data_dir / LOG_FILE, base_seq=snapshot_seq)
        if update_log.last_seq < snapshot_seq:
            # Everything in it is covered by the snapshot: a crash between the two
            # steps of install_snapshot (D-031). Start the log after the snapshot.
            log.warning(
                "%s: log ends at seq=%d, before snapshot_seq=%d; resetting it",
                data_dir, update_log.last_seq, snapshot_seq,
            )
            await update_log.reset(snapshot_seq)
        last = update_log.last_seq

        if update_log.first_seq > snapshot_seq + 1 or not snapshot_seq <= persisted <= last:
            update_log.close()
            db.close()
            raise CorruptStateError(
                f"{data_dir}: expected first logged seq - 1 <= snapshot_seq <= applied_seq <= last logged "
                f"seq, got first_logged_seq={update_log.first_seq} snapshot_seq={snapshot_seq} "
                f"applied_seq={persisted} last_logged_seq={last}"
            )

        index = InvertedIndex.from_state(docs)
        replayed = 0
        async for entry in update_log.read_after(snapshot_seq):
            result = index.apply(entry_to_operation(entry))
            log.debug("replay seq=%d doc=%s v=%d -> %s", entry.seq, entry.doc_id, entry.version, result.value)
            replayed += 1
        await db.set_applied_seq(last)

        report = OpenReport(snapshot_seq, persisted, last, replayed, time.monotonic() - started)
        log.info(
            "opened %s in %.3fs: snapshot_seq=%d persisted applied_seq=%d replayed seq=%d..%d (%d entries)",
            data_dir, report.seconds, snapshot_seq, persisted, snapshot_seq + 1, last, replayed,
        )
        return cls(update_log, db, index, snapshot_seq, snapshot_every, report, log_retention)

    async def write(self, entry: search_pb2.LogEntry) -> ApplyResult:
        """Durably log `entry`, apply it and record it as applied (a batch of one)."""
        return (await self.write_batch([entry]))[0]

    async def write_batch(self, entries: list[search_pb2.LogEntry]) -> list[ApplyResult]:
        """Replica path: log the consecutive `entries` with one fsync, apply them,
        then persist the last seq as applied (group commit, D-034).

        The first seq must be exactly `applied_seq + 1` and the rest dense;
        otherwise nothing is logged. Any failure part-way leaves the store
        failed (crash-stop); reopening recovers from disk.
        """
        ops = [entry_to_operation(e) for e in entries]  # reject malformed entries before logging
        # Shielded (D-021): a cancelled caller must not cut the durability steps in half.
        return await asyncio.shield(self._write_batch(entries, ops))

    async def append_new(self, op: Operation) -> tuple[int, ApplyResult]:
        """Assign the next seq to `op` and write it (primary path)."""
        return await asyncio.shield(self._append_new(op))  # see D-021

    async def _write_batch(self, entries: list[search_pb2.LogEntry], ops: list[Operation]) -> list[ApplyResult]:
        async with self._write_lock:
            for i, entry in enumerate(entries):
                if entry.seq != self.applied_seq + 1 + i:
                    raise SeqMismatchError(f"write seq={entry.seq}, expected seq={self.applied_seq + 1 + i}")
            return await self._write_locked(entries, ops)

    async def _append_new(self, op: Operation) -> tuple[int, ApplyResult]:
        async with self._write_lock:
            entry = make_entry(self.applied_seq + 1, op)
            return entry.seq, (await self._write_locked([entry], [op]))[0]

    async def _write_locked(self, entries: list[search_pb2.LogEntry], ops: list[Operation]) -> list[ApplyResult]:
        if not entries:
            return []
        if self._failed:
            raise StoreFailedError("store failed earlier; restart the node")
        last = entries[-1].seq
        try:
            await self.update_log.append_batch(entries)  # 1. log + fsync (once per batch)
            results = [self.index.apply(op) for op in ops]  # 2. apply (sync, atomic for search)
            await self.db.set_applied_seq(last)  # 3. persist position
        except BaseException:
            self._failed = True
            log.exception("write seq=%d..%d failed; store is now failed", entries[0].seq, last)
            raise
        self.applied_seq = last
        for entry, op, result in zip(entries, ops, results):
            log.debug(
                "applied seq=%d %s doc=%s v=%d -> %s",
                entry.seq, op.op_type.value, op.doc_id, op.version, result.value,
            )
        async with self._applied_changed:
            self._applied_changed.notify_all()
        if self.snapshot_every and self.applied_seq - self.snapshot_seq >= self.snapshot_every:
            try:
                await self._snapshot_locked()
            except Exception:
                # The write is already durable; a missed snapshot only means a longer replay.
                log.exception("snapshot at seq=%d failed", self.applied_seq)
        return results

    # ---- reading applied entries (replication source, D-014) -----------

    async def read_applied_after(self, after_seq: int, max_entries: int) -> list[search_pb2.LogEntry]:
        """Up to `max_entries` logged entries with after_seq < seq <= applied_seq."""
        limit = min(max_entries, self.applied_seq - after_seq)
        if limit <= 0:
            return []
        return await asyncio.to_thread(self.update_log.read_after_sync, after_seq, limit)

    async def wait_for_applied_beyond(self, seq: int) -> None:
        """Return once applied_seq > seq."""
        async with self._applied_changed:
            await self._applied_changed.wait_for(lambda: self.applied_seq > seq)

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
        if self.log_retention:
            dropped = await self.update_log.truncate_through(seq - self.log_retention)
            if dropped:
                log.info(
                    "truncated log: dropped %d entries, log now starts at seq=%d (retention=%d)",
                    dropped, self.update_log.first_seq, self.log_retention,
                )

    # ---- full rebuild (D-031) ------------------------------------------

    async def consistent_state(self) -> tuple[int, dict[str, DocState]]:
        """(applied_seq, copy of every doc) at one seq, for a replica's full rebuild.
        Taken under the write lock: a write applies to the index before its
        applied_seq is updated, so the two are only consistent between writes."""
        async with self._write_lock:
            return self.applied_seq, self.index.state()

    async def install_snapshot(self, seq: int, docs: dict[str, DocState]) -> None:
        """Replace this copy's whole state with `docs` as of `seq` (replica full rebuild).

        Order: (1) SQLite snapshot + applied_seq in one transaction, (2) empty the
        local log so it continues after `seq`, (3) swap the in-memory index. A crash
        after (1) leaves a log older than the snapshot, which `open` resets.
        """
        await asyncio.shield(self._install_snapshot(seq, docs))  # see D-021

    async def _install_snapshot(self, seq: int, docs: dict[str, DocState]) -> None:
        async with self._write_lock:
            if self._failed:
                raise StoreFailedError("store failed earlier; restart the node")
            try:
                await self.db.write_snapshot(seq, docs, set_applied=True)
                await self.update_log.reset(seq)
                self.index = InvertedIndex.from_state(docs)
            except BaseException:
                self._failed = True
                log.exception("install of snapshot at seq=%d failed; store is now failed", seq)
                raise
            self.applied_seq = self.snapshot_seq = seq
            log.warning("installed snapshot at seq=%d (%d docs); local log reset", seq, len(docs))
        async with self._applied_changed:
            self._applied_changed.notify_all()

    async def close(self) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self.update_log.close)
            await asyncio.to_thread(self.db.close)
