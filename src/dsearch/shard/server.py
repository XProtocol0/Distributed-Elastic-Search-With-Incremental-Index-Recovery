"""gRPC `Shard` service for primaries and replicas."""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc

from dsearch.index.operation import ApplyResult
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.shard.primary import Primary, ReplicaAheadError
from dsearch.shard.replica import Replica
from dsearch.shard.store import ShardStore, StoreFailedError
from dsearch.wire import (
    hits_to_proto,
    state_to_proto,
    stats_from_proto,
    stats_to_proto,
    write_request_to_operation,
)


class ShardServicer(search_pb2_grpc.ShardServicer):
    """Serves one shard copy. Exactly one of `primary` / `replica` is set."""

    def __init__(
        self, node_id: str, store: ShardStore, primary: Primary | None, replica: Replica | None = None
    ) -> None:
        self._node_id = node_id
        self._store = store
        self._primary = primary
        self._replica = replica

    async def Write(
        self, request: search_pb2.WriteRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.WriteReply:
        primary = await self._require_primary(context)
        try:
            op = write_request_to_operation(request)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        try:
            seq, result = await primary.write(op)
        except StoreFailedError as exc:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        return search_pb2.WriteReply(seq=seq, applied=result is ApplyResult.APPLIED)

    async def Replicate(
        self, request: search_pb2.ReplicateRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[search_pb2.ReplicateMessage]:
        primary = await self._require_primary(context)
        try:
            async for message in primary.replicate(request.replica_id, request.from_seq):
                yield message
        except ReplicaAheadError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    async def GetStats(
        self, request: search_pb2.StatsRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.CorpusStats:
        return stats_to_proto(self._store.index.local_stats(request.query))

    async def SearchShard(
        self, request: search_pb2.ShardSearchRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.ShardSearchReply:
        stats = stats_from_proto(request.global_stats)
        hits = self._store.index.search(request.query, stats, request.k)
        return search_pb2.ShardSearchReply(
            hits=hits_to_proto(hits), applied_seq=self._store.applied_seq
        )

    async def DumpState(
        self, request: search_pb2.DumpStateRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.DumpStateReply:
        return search_pb2.DumpStateReply(
            applied_seq=self._store.applied_seq, docs=state_to_proto(self._store.index.state())
        )

    async def GetStatus(
        self, request: search_pb2.StatusRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.StatusReply:
        """Replication and recovery status of this copy (D-025)."""
        store = self._store
        reply = search_pb2.StatusReply(
            node_id=self._node_id,
            role="primary" if self._primary is not None else "replica",
            applied_seq=store.applied_seq,
            last_logged_seq=store.update_log.last_seq,
            log_bytes=store.update_log.size_bytes,
            snapshot_seq=store.snapshot_seq,
            local_replayed=store.open_report.replayed,
        )
        if self._replica is not None:
            reply.live = self._replica.live
            reply.full_rebuilds = self._replica.full_rebuilds
            for r in self._replica.recoveries:
                reply.recoveries.add(
                    from_seq=r.from_seq,
                    caught_up_seq=r.caught_up_seq,
                    entries_replayed=r.entries_replayed,
                    seconds=r.seconds,
                )
        return reply

    async def _require_primary(self, context: grpc.aio.ServicerContext) -> Primary:
        if self._primary is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "this node is not the primary")
        return self._primary
