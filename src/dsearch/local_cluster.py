"""Run a whole cluster inside one process: one gRPC server per node on localhost.

Used by integration tests and evaluation scripts. The Docker Compose cluster
runs the same node classes as separate processes.
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import replace
from pathlib import Path

from dsearch.config import Config, Role, SearchPreference, ShardTopology
from dsearch.coordinator.server import CoordinatorNode
from dsearch.shard.node import ShardNode


def free_port() -> int:
    """A currently unused localhost port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def node_id(shard_id: int, copy: int) -> str:
    """copy 0 is the primary, 1..R are replicas."""
    return f"shard{shard_id}-primary" if copy == 0 else f"shard{shard_id}-replica{copy}"


class LocalCluster:
    def __init__(self, base: Config, ports: list[list[int]]) -> None:
        self.base = base
        self.ports = ports  # ports[shard_id][copy]
        self.shards: list[list[ShardNode | None]] = [[None] * len(p) for p in ports]
        self.coordinator: CoordinatorNode | None = None

    @classmethod
    async def start(
        cls,
        data_dir: Path,
        num_shards: int,
        replicas_per_shard: int,
        search_preference: SearchPreference = SearchPreference.PRIMARY,
        snapshot_every: int = 0,
    ) -> "LocalCluster":
        ports = [[free_port() for _ in range(replicas_per_shard + 1)] for _ in range(num_shards)]
        topology = tuple(
            ShardTopology(
                shard_id=i,
                primary=f"127.0.0.1:{p[0]}",
                replicas=tuple(f"127.0.0.1:{port}" for port in p[1:]),
            )
            for i, p in enumerate(ports)
        )
        base = Config(
            num_shards=num_shards,
            replicas_per_shard=replicas_per_shard,
            data_dir=data_dir,
            snapshot_every=snapshot_every,
            search_preference=search_preference,
            shards=topology,
        )
        cluster = cls(base, ports)
        for shard_id in range(num_shards):
            for copy in range(replicas_per_shard + 1):
                await cluster.start_node(shard_id, copy)
        coordinator_config = replace(base, node_id="coordinator", role=Role.COORDINATOR, port=free_port())
        cluster.coordinator = await CoordinatorNode.start(coordinator_config)
        return cluster

    def node_config(self, shard_id: int, copy: int) -> Config:
        role = Role.PRIMARY if copy == 0 else Role.REPLICA
        return replace(
            self.base,
            node_id=node_id(shard_id, copy),
            role=role,
            shard_id=shard_id,
            port=self.ports[shard_id][copy],
        )

    async def start_node(self, shard_id: int, copy: int) -> ShardNode:
        """Start (or restart, on its existing data dir) one shard node."""
        node = await ShardNode.start(self.node_config(shard_id, copy), replica_retry_delay=0.05)
        self.shards[shard_id][copy] = node
        return node

    async def stop_node(self, shard_id: int, copy: int) -> None:
        node = self.shards[shard_id][copy]
        if node is not None:
            await node.stop()
            self.shards[shard_id][copy] = None

    def node(self, shard_id: int, copy: int) -> ShardNode:
        node = self.shards[shard_id][copy]
        if node is None:
            raise RuntimeError(f"{node_id(shard_id, copy)} is not running")
        return node

    @property
    def coordinator_address(self) -> str:
        assert self.coordinator is not None
        return f"127.0.0.1:{self.coordinator.port}"

    async def wait_replicated(self, timeout: float = 10.0) -> None:
        """Wait until every running, connected replica has applied everything its
        primary has. Replicas deliberately disconnected (D-016) are skipped."""
        waits = []
        for copies in self.shards:
            primary = copies[0]
            if primary is None:
                continue
            target = primary.store.applied_seq
            for replica in copies[1:]:
                if replica is not None and replica.replica is not None and replica.replica.is_connected:
                    waits.append(replica.store.wait_for_applied_beyond(target - 1))
        await asyncio.wait_for(asyncio.gather(*waits), timeout)

    async def stop(self) -> None:
        if self.coordinator is not None:
            await self.coordinator.stop()
        for shard_id, copies in enumerate(self.shards):
            for copy in range(len(copies) - 1, -1, -1):  # replicas first
                await self.stop_node(shard_id, copy)
