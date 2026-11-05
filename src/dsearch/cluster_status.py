"""Polling a running cluster over gRPC: liveness and replication status.

Used by the cluster scripts and by tests that run nodes as separate
processes or containers, where node objects can't be inspected directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import grpc

from dsearch.config import ShardTopology
from dsearch.proto import search_pb2, search_pb2_grpc


async def ping(address: str, timeout: float = 2.0) -> search_pb2.PingReply:
    async with grpc.aio.insecure_channel(address) as channel:
        return await search_pb2_grpc.NodeStub(channel).Ping(search_pb2.PingRequest(), timeout=timeout)


async def get_status(address: str, timeout: float = 2.0) -> search_pb2.StatusReply:
    async with grpc.aio.insecure_channel(address) as channel:
        return await search_pb2_grpc.ShardStub(channel).GetStatus(search_pb2.StatusRequest(), timeout=timeout)


async def wait_until_up(addresses: Sequence[str], timeout: float, poll: float = 0.2) -> None:
    """Wait until every address answers Ping."""
    deadline = time.monotonic() + timeout
    pending = list(addresses)
    while pending:
        still_down = []
        for address in pending:
            try:
                await ping(address)
            except grpc.aio.AioRpcError:
                still_down.append(address)
        pending = still_down
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"nodes not up after {timeout}s: {', '.join(pending)}")
            await asyncio.sleep(poll)


async def wait_for_replicas(
    shards: Sequence[ShardTopology], timeout: float, poll: float = 0.2
) -> dict[str, search_pb2.StatusReply]:
    """Wait until every replica is streaming live and has applied everything
    its primary has. Call it once writes have stopped. Returns the final
    status of every copy, by address."""
    deadline = time.monotonic() + timeout
    while True:
        statuses: dict[str, search_pb2.StatusReply] = {}
        lagging: list[str] = []
        for shard in shards:
            try:
                primary = await get_status(shard.primary)
            except grpc.aio.AioRpcError as exc:
                lagging.append(f"{shard.primary} ({exc.code().name})")
                continue
            statuses[shard.primary] = primary
            for address in shard.replicas:
                try:
                    replica = await get_status(address)
                except grpc.aio.AioRpcError as exc:
                    lagging.append(f"{address} ({exc.code().name})")
                    continue
                statuses[address] = replica
                if not replica.live or replica.applied_seq != primary.applied_seq:
                    lagging.append(f"{address} (seq {replica.applied_seq}/{primary.applied_seq})")
        if not lagging:
            return statuses
        if time.monotonic() >= deadline:
            raise TimeoutError(f"replicas not caught up after {timeout}s: {', '.join(lagging)}")
        await asyncio.sleep(poll)
