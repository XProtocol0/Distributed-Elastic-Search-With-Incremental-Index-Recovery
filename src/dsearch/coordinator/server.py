"""gRPC `Coordinator` service and the coordinator node."""

from __future__ import annotations

import logging

import grpc

from dsearch.config import Config, ConfigError, Role
from dsearch.coordinator.router import shard_for
from dsearch.coordinator.search import ScatterGather, ShardUnavailableError
from dsearch.node_server import channel_options, create_server
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.wire import hits_to_proto

log = logging.getLogger(__name__)


class CoordinatorServicer(search_pb2_grpc.CoordinatorServicer):
    def __init__(self, config: Config, scatter_gather: ScatterGather) -> None:
        self._num_shards = config.num_shards
        self._primaries = {s.shard_id: s.primary for s in config.shards}
        self._scatter_gather = scatter_gather
        self._timeout = config.rpc_timeout

    async def Write(
        self, request: search_pb2.WriteRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.CoordinatorWriteReply:
        """Forward the write to its shard's primary; primary errors pass through."""
        shard_id = shard_for(request.doc_id, self._num_shards)
        primary = self._primaries[shard_id]
        try:
            reply = await self._scatter_gather.stub(primary).Write(request, timeout=self._timeout)
        except grpc.aio.AioRpcError as exc:
            log.warning("write doc=%s -> shard=%d %s failed: %s", request.doc_id, shard_id, primary, exc.code().name)
            await context.abort(exc.code(), f"shard {shard_id} primary {primary}: {exc.details()}")
        log.debug("write doc=%s -> shard=%d seq=%d applied=%s", request.doc_id, shard_id, reply.seq, reply.applied)
        return search_pb2.CoordinatorWriteReply(shard_id=shard_id, seq=reply.seq, applied=reply.applied)

    async def Search(
        self, request: search_pb2.SearchRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.SearchReply:
        try:
            result = await self._scatter_gather.search(request.query, request.k)
        except ShardUnavailableError as exc:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        return search_pb2.SearchReply(hits=hits_to_proto(result.hits), copies_read=result.copies_read)


class CoordinatorNode:
    def __init__(self, server: grpc.aio.Server, port: int, scatter_gather: ScatterGather) -> None:
        self.server = server
        self.port = port
        self.scatter_gather = scatter_gather

    @classmethod
    async def start(cls, config: Config) -> "CoordinatorNode":
        if config.role is not Role.COORDINATOR:
            raise ConfigError(f"coordinator needs role coordinator, got {config.role.value}")
        if len(config.shards) != config.num_shards:
            raise ConfigError(f"coordinator needs [[shards]] topology for all {config.num_shards} shards")
        scatter_gather = ScatterGather(
            config.shards, config.search_preference, config.rpc_timeout, channel_options(config)
        )
        server, port = create_server(config)
        search_pb2_grpc.add_CoordinatorServicer_to_server(
            CoordinatorServicer(config, scatter_gather), server
        )
        await server.start()
        log.info(
            "coordinator %s serving on port %d: %d shards, search_preference=%s",
            config.node_id, port, config.num_shards, config.search_preference.value,
        )
        return cls(server, port, scatter_gather)

    async def stop(self) -> None:
        await self.server.stop(grace=None)
        await self.scatter_gather.close()
