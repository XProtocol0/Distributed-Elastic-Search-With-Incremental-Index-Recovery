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
from dsearch.shard.store import ShardStore

log = logging.getLogger(__name__)


class ReplicaAheadError(Exception):
    """A replica asked for entries after a seq the primary has not reached (D-015)."""


class Primary:
    def __init__(self, store: ShardStore, batch_size: int = 500) -> None:
        self.store = store
        self.batch_size = batch_size

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
            for entry in batch:
                log.debug("replica=%s send seq=%d", replica_id, entry.seq)
                yield search_pb2.ReplicateMessage(entry=entry)
            next_seq = batch[-1].seq
