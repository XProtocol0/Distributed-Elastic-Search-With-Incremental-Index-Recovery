"""Shard replica: pulls log entries from its primary and applies them in seq order.

The replica calls `Replicate(from_seq = applied_seq)` on the primary.
Everything before the `caught_up_seq` marker is incremental recovery; after
it, entries arrive live. If the stream breaks for any reason, the replica
reconnects from its persisted applied_seq, so only missed entries are
replayed. Any unexpected seq drops the stream. The channel uses
keepalive, so a silent partition also breaks the stream.

If the primary answers OUT_OF_RANGE (its log no longer holds the entries
after our applied_seq), the next session is a full rebuild: fetch the
primary's state at some seq S, install it, then replicate from S as usual.
Every full rebuild is logged and counted (Invariant 8).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import grpc

from dsearch.index.inverted_index import DocState
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.shard.store import SeqMismatchError, ShardStore, StoreFailedError
from dsearch.wire import state_from_proto

log = logging.getLogger(__name__)


class SeqGapError(Exception):
    """The primary sent a seq other than applied_seq + 1."""


@dataclass(frozen=True)
class RecoveryRecord:
    """One catch-up session: from connect until the caught_up marker."""

    from_seq: int
    caught_up_seq: int
    entries_replayed: int  # log entries applied (after the snapshot, for a full rebuild)
    seconds: float
    full_rebuild: bool = False
    snapshot_docs: int = 0


class Replica:
    def __init__(
        self,
        store: ShardStore,
        primary_address: str,
        replica_id: str,
        retry_delay: float = 0.5,
        channel_options: list[tuple[str, int]] | None = None,
        force_full_rebuild: bool = False,
    ) -> None:
        self.store = store
        self.primary_address = primary_address
        self.replica_id = replica_id
        self.retry_delay = retry_delay
        self.channel_options = channel_options or []
        self.live = False  # past the caught_up marker on the current stream
        self.recoveries: list[RecoveryRecord] = []
        self.full_rebuilds = 0
        # Why the next session must start with a full rebuild, or None.
        self._rebuild_reason: str | None = "forced (evaluation baseline)" if force_full_rebuild else None
        self._task: asyncio.Task[None] | None = None
        self._recovered = asyncio.Condition()  # notified when a catch-up session completes

    # ---- control (also the in-process fault-injection hooks) ------------

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
                if exc.code() is grpc.StatusCode.OUT_OF_RANGE:
                    self._rebuild_reason = f"primary log truncated past applied_seq={self.store.applied_seq}"
                log.warning(
                    "replica=%s: stream from %s failed at applied_seq=%d: %s %s",
                    self.replica_id, self.primary_address, self.store.applied_seq,
                    exc.code().name, exc.details(),
                )
            except (SeqGapError, SeqMismatchError) as exc:
                # e.g. a write cancelled by disconnect() finished after we reconnected
                log.warning("replica=%s: %s; reconnecting", self.replica_id, exc)
            except StoreFailedError:
                log.error("replica=%s: local store failed; stopping replication", self.replica_id)
                return
            finally:
                self.live = False
            await asyncio.sleep(self.retry_delay)

    async def _replicate_once(self) -> None:
        """Run one Replicate stream until it ends or fails."""
        from_seq = self.store.applied_seq
        started = time.monotonic()
        replayed = 0
        snapshot_docs = 0
        full_rebuild = self._rebuild_reason is not None
        async with grpc.aio.insecure_channel(self.primary_address, options=self.channel_options) as channel:
            stub = search_pb2_grpc.ShardStub(channel)
            if full_rebuild:
                snapshot_docs = await self._full_rebuild(stub)
            log.info(
                "replica=%s requesting entries after seq=%d from %s",
                self.replica_id, self.store.applied_seq, self.primary_address,
            )
            stream = stub.Replicate(
                search_pb2.ReplicateRequest(replica_id=self.replica_id, from_seq=self.store.applied_seq)
            )
            async for message in stream:
                if message.HasField("entries"):
                    entries = list(message.entries.entries)
                    expected = self.store.applied_seq + 1
                    if [e.seq for e in entries] != list(range(expected, expected + len(entries))):
                        raise SeqGapError(
                            f"received seq={entries[0].seq}..{entries[-1].seq}, expected from seq={expected}"
                        )
                    await self.store.write_batch(entries)  # one group commit per message
                    replayed += len(entries)
                    log.debug("replica=%s applied seq=%d..%d", self.replica_id, entries[0].seq, entries[-1].seq)
                else:
                    record = RecoveryRecord(
                        from_seq, message.caught_up_seq, replayed, time.monotonic() - started,
                        full_rebuild, snapshot_docs,
                    )
                    self.recoveries.append(record)
                    self.live = True
                    async with self._recovered:
                        self._recovered.notify_all()
                    log.info(
                        "replica=%s caught up%s: replayed seq=%d..%d (%d entries) in %.3fs",
                        self.replica_id, " after full rebuild" if full_rebuild else "",
                        record.caught_up_seq - replayed + 1, record.caught_up_seq, replayed, record.seconds,
                    )

    async def _full_rebuild(self, stub: search_pb2_grpc.ShardStub) -> int:
        """Replace local state with the primary's snapshot; returns the doc count."""
        log.warning(
            "replica=%s FULL REBUILD at applied_seq=%d: %s",
            self.replica_id, self.store.applied_seq, self._rebuild_reason,
        )
        seq = None
        docs: dict[str, DocState] = {}
        async for chunk in stub.FetchSnapshot(search_pb2.FetchSnapshotRequest(replica_id=self.replica_id)):
            seq = chunk.seq
            docs.update(state_from_proto(chunk.docs))
        if seq is None:
            raise SeqGapError("primary sent an empty snapshot stream")
        await self.store.install_snapshot(seq, docs)
        self.full_rebuilds += 1
        self._rebuild_reason = None
        log.warning(
            "replica=%s full rebuild #%d installed %d docs at seq=%d",
            self.replica_id, self.full_rebuilds, len(docs), seq,
        )
        return len(docs)
