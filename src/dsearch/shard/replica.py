"""Shard replica: pulls log entries from its primary and applies them in seq order.

The replica calls `Replicate(from_seq = applied_seq)` on the primary (D-012).
Everything before the `caught_up_seq` marker is incremental recovery; after
it, entries arrive live. If the stream breaks for any reason, the replica
reconnects from its persisted applied_seq, so only missed entries are
replayed. Any unexpected seq drops the stream (D-015).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import grpc

from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.shard.store import SeqMismatchError, ShardStore, StoreFailedError

log = logging.getLogger(__name__)


class SeqGapError(Exception):
    """The primary sent a seq other than applied_seq + 1."""


@dataclass(frozen=True)
class RecoveryRecord:
    """One catch-up session: from connect until the caught_up marker."""

    from_seq: int
    caught_up_seq: int
    entries_replayed: int
    seconds: float


class Replica:
    def __init__(
        self,
        store: ShardStore,
        primary_address: str,
        replica_id: str,
        retry_delay: float = 0.5,
    ) -> None:
        self.store = store
        self.primary_address = primary_address
        self.replica_id = replica_id
        self.retry_delay = retry_delay
        self.recoveries: list[RecoveryRecord] = []
        self.full_rebuilds = 0  # stays 0 until the Phase 7 full-rebuild path exists
        self._task: asyncio.Task[None] | None = None
        self._recovered = asyncio.Condition()  # notified when a catch-up session completes

    # ---- control (also the in-process fault-injection hooks, D-016) -----

    def connect(self) -> None:
        """Start (or resume) pulling from the primary in the background."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"replicate-{self.replica_id}")

    async def disconnect(self) -> None:
        """Drop the replication stream and stop reconnecting until `connect()`."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            log.info("replica=%s disconnected at applied_seq=%d", self.replica_id, self.store.applied_seq)

    @property
    def is_connected(self) -> bool:
        return self._task is not None and not self._task.done()

    async def wait_for_recovery(self, count: int) -> RecoveryRecord:
        """Wait until `count` catch-up sessions have completed; return the latest one."""
        async with self._recovered:
            await self._recovered.wait_for(lambda: len(self.recoveries) >= count)
        return self.recoveries[-1]

    # ---- replication loop -----------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                await self._replicate_once()
                log.warning("replica=%s: primary ended the stream", self.replica_id)
            except grpc.aio.AioRpcError as exc:
                log.warning(
                    "replica=%s: stream from %s failed at applied_seq=%d: %s %s",
                    self.replica_id, self.primary_address, self.store.applied_seq,
                    exc.code().name, exc.details(),
                )
            except (SeqGapError, SeqMismatchError) as exc:
                # e.g. a write cancelled by disconnect() finished after we reconnected (D-021)
                log.warning("replica=%s: %s; reconnecting", self.replica_id, exc)
            except StoreFailedError:
                log.error("replica=%s: local store failed; stopping replication", self.replica_id)
                return
            await asyncio.sleep(self.retry_delay)

    async def _replicate_once(self) -> None:
        """Run one Replicate stream until it ends or fails."""
        from_seq = self.store.applied_seq
        started = time.monotonic()
        replayed = 0
        log.info("replica=%s requesting entries after seq=%d from %s", self.replica_id, from_seq, self.primary_address)
        async with grpc.aio.insecure_channel(self.primary_address) as channel:
            stream = search_pb2_grpc.ShardStub(channel).Replicate(
                search_pb2.ReplicateRequest(replica_id=self.replica_id, from_seq=from_seq)
            )
            async for message in stream:
                if message.HasField("entry"):
                    entry = message.entry
                    expected = self.store.applied_seq + 1
                    if entry.seq != expected:
                        raise SeqGapError(f"received seq={entry.seq}, expected seq={expected}")
                    result = await self.store.write(entry)
                    replayed += 1
                    log.debug("replica=%s applied seq=%d -> %s", self.replica_id, entry.seq, result.value)
                else:
                    record = RecoveryRecord(
                        from_seq, message.caught_up_seq, replayed, time.monotonic() - started
                    )
                    self.recoveries.append(record)
                    async with self._recovered:
                        self._recovered.notify_all()
                    log.info(
                        "replica=%s caught up: replayed seq=%d..%d (%d entries) in %.3fs",
                        self.replica_id, from_seq + 1, record.caught_up_seq, replayed, record.seconds,
                    )
