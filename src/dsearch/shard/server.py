"""gRPC `Shard` service for primaries and replicas."""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc

from dsearch.index.operation import ApplyResult
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.shard.primary import Primary, ReplicaAheadError
from dsearch.shard.store import ShardStore, StoreFailedError
from dsearch.wire import (
    hits_to_proto,
    state_to_proto,
    stats_from_proto,
    stats_to_proto,
    write_request_to_operation,
)


class ShardServicer(search_pb2_grpc.ShardServicer):
    """Serves one shard copy. `primary` is None on replicas."""

    def __init__(self, store: ShardStore, primary: Primary | None) -> None:
        self._store = store
        self._primary = primary

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

    async def _require_primary(self, context: grpc.aio.ServicerContext) -> Primary:
        if self._primary is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "this node is not the primary")
        return self._primary
