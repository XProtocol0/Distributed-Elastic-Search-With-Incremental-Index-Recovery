"""Shard primary: assigns seqs to writes and serves replication streams.

A write is acknowledged once the primary has logged, fsynced and applied it
(D-013). Replicas pull entries with `Replicate(from_seq)` (D-012). The stream
reads from the primary's log, so catch-up and live replication are the same
loop (D-014).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from dsearch.index.operation import ApplyResult, Operation
from dsearch.proto import search_pb2
from dsearch.wire import state_to_proto
from dsearch.shard.store import ShardStore
from dsearch.storage.update_log import LogTruncatedError

log = logging.getLogger(__name__)


class ReplicaAheadError(Exception):
    """A replica asked for entries after a seq the primary has not reached (D-015)."""


class Primary:
    def __init__(self, store: ShardStore, batch_size: int = 500, batch_bytes: int = 1 << 20) -> None:
        self.store = store
        self.batch_size = batch_size  # entries per log read and per message
        self.batch_bytes = batch_bytes  # message size cap, well under gRPC's 4 MB limit

    async def write(self, op: Operation) -> tuple[int, ApplyResult]:
        """Assign the next seq to `op`, make it durable and apply it."""
        seq, result = await self.store.append_new(op)
        log.info(
            "primary seq=%d %s doc=%s v=%d -> %s",
            seq, op.op_type.value, op.doc_id, op.version, result.value,
        )
        return seq, result

    async def replicate(
        self, replica_id: str, from_seq: int
    ) -> AsyncIterator[search_pb2.ReplicateMessage]:
        """Stream entries with seq > from_seq: catch-up, a caught_up marker, then live."""
        target = self.store.applied_seq
        if from_seq > target:
            raise ReplicaAheadError(
                f"replica={replica_id} from_seq={from_seq} is ahead of primary applied_seq={target}"
            )
        if from_seq + 1 < self.store.update_log.first_seq:
            # Reads would raise this too; checking first gives a clear log line (D-030).
            raise LogTruncatedError(
                f"replica={replica_id} needs entries after seq={from_seq}, "
                f"log starts at seq={self.store.update_log.first_seq}"
            )
        log.info(
            "replica=%s connected from_seq=%d: catch-up seq=%d..%d (%d entries)",
            replica_id, from_seq, from_seq + 1, target, target - from_seq,
        )
        next_seq = from_seq
        caught_up = False
        while True:
            if not caught_up and next_seq >= target:
                caught_up = True
                log.info("replica=%s caught up to seq=%d, streaming live", replica_id, target)
                yield search_pb2.ReplicateMessage(caught_up_seq=target)

            # During catch-up, stop each batch at `target` so the marker comes right
            # after it and the replica's recovery record counts only missed entries (D-029).
            limit = self.batch_size if caught_up else min(self.batch_size, target - next_seq)
            batch = await self.store.read_applied_after(next_seq, limit)
            if not batch:
                await self.store.wait_for_applied_beyond(next_seq)
                continue
            for message in self._messages(batch):
                log.debug(
                    "replica=%s send seq=%d..%d", replica_id,
                    message.entries.entries[0].seq, message.entries.entries[-1].seq,
                )
                yield message
            next_seq = batch[-1].seq

    def _messages(self, batch: list[search_pb2.LogEntry]) -> list[search_pb2.ReplicateMessage]:
        """Split consecutive entries into messages of at most `batch_bytes` (D-034)."""
        messages = []
        current = search_pb2.ReplicateMessage()
        size = 0
        for entry in batch:
            if current.entries.entries and size + entry.ByteSize() > self.batch_bytes:
                messages.append(current)
                current = search_pb2.ReplicateMessage()
                size = 0
            current.entries.entries.append(entry)
            size += entry.ByteSize()
        messages.append(current)
        return messages

    async def snapshot_chunks(
        self, replica_id: str, chunk_bytes: int = 1 << 20
    ) -> AsyncIterator[search_pb2.SnapshotChunk]:
        """The whole shard state at one seq, for a replica's full rebuild (D-031),
        in chunks of about `chunk_bytes` (well under gRPC's 4 MB message limit)."""
        seq, docs = await self.store.consistent_state()
        log.warning("replica=%s full rebuild: sending snapshot at seq=%d (%d docs)", replica_id, seq, len(docs))
        chunk = search_pb2.SnapshotChunk(seq=seq)
        size = 0
        for record in state_to_proto(docs):
            if chunk.docs and size + record.ByteSize() > chunk_bytes:
                yield chunk
                chunk = search_pb2.SnapshotChunk(seq=seq)
                size = 0
            chunk.docs.append(record)
            size += record.ByteSize()
        yield chunk  # always at least one, so an empty shard still sends its seq
