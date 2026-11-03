"""Helpers for in-process shard clusters (real gRPC on localhost, no Docker)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import grpc

from dsearch.config import Config, Role, ShardTopology
from dsearch.index.inverted_index import DocState
from dsearch.index.operation import Operation
from dsearch.local_cluster import free_port
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.shard.node import ShardNode
from dsearch.shard.replica import RecoveryRecord
from dsearch.wire import operation_to_write_request, state_from_proto


def shard_config(data_dir: Path, role: Role, node_id: str, port: int, primary_port: int) -> Config:
    return Config(
        node_id=node_id,
        role=role,
        shard_id=0,
        num_shards=1,
        replicas_per_shard=1,
        port=port,
        data_dir=data_dir,
        snapshot_every=0,
        shards=(ShardTopology(0, f"127.0.0.1:{primary_port}", ()),),
    )


async def start_primary(data_dir: Path, port: int) -> ShardNode:
    return await ShardNode.start(shard_config(data_dir, Role.PRIMARY, "s0-primary", port, port))


async def start_replica(data_dir: Path, primary_port: int, node_id: str = "s0-replica1") -> ShardNode:
    config = shard_config(data_dir, Role.REPLICA, node_id, 0, primary_port)
    return await ShardNode.start(config, replica_retry_delay=0.05)


class ShardClient:
    """Thin async client for one shard node."""

    def __init__(self, port: int) -> None:
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.stub = search_pb2_grpc.ShardStub(self.channel)

    async def write(self, op: Operation) -> search_pb2.WriteReply:
        return await self.stub.Write(operation_to_write_request(op))

    async def dump(self) -> tuple[int, dict[str, DocState]]:
        reply = await self.stub.DumpState(search_pb2.DumpStateRequest())
        return reply.applied_seq, state_from_proto(reply.docs)

    async def close(self) -> None:
        await self.channel.close()


async def wait_for_seq(node: ShardNode, seq: int, timeout: float = 10.0) -> None:
    """Wait until `node` has applied `seq`."""
    await asyncio.wait_for(node.store.wait_for_applied_beyond(seq - 1), timeout)


async def wait_for_recovery(node: ShardNode, count: int, timeout: float = 10.0) -> RecoveryRecord:
    """Wait until replica `node` has completed `count` catch-up sessions."""
    assert node.replica is not None
    return await asyncio.wait_for(node.replica.wait_for_recovery(count), timeout)
