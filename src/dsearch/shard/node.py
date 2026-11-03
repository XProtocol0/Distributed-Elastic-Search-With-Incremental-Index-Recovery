"""A running shard node (primary or replica): store + gRPC server + role logic."""

from __future__ import annotations

import logging

import grpc

from dsearch.config import Config, ConfigError, Role
from dsearch.node_server import channel_options, create_server
from dsearch.proto import search_pb2_grpc
from dsearch.shard.primary import Primary
from dsearch.shard.replica import Replica
from dsearch.shard.server import ShardServicer
from dsearch.shard.store import ShardStore

log = logging.getLogger(__name__)


class ShardNode:
    def __init__(
        self,
        config: Config,
        store: ShardStore,
        server: grpc.aio.Server,
        port: int,
        primary: Primary | None,
        replica: Replica | None,
    ) -> None:
        self.config = config
        self.store = store
        self.server = server
        self.port = port
        self.primary = primary
        self.replica = replica

    @classmethod
    async def start(cls, config: Config, replica_retry_delay: float = 0.5) -> "ShardNode":
        """Open the node's store, start serving, and (replicas) start pulling."""
        if config.role not in (Role.PRIMARY, Role.REPLICA):
            raise ConfigError(f"a shard node needs role primary or replica, got {config.role.value}")
        store = await ShardStore.open(config.node_data_dir, config.snapshot_every, config.log_retention)
        primary = Primary(store) if config.role is Role.PRIMARY else None
        replica = None
        if config.role is Role.REPLICA:
            replica = Replica(
                store, primary_address(config), config.node_id, replica_retry_delay, channel_options(config),
                force_full_rebuild=config.force_full_rebuild,
            )

        server, port = create_server(config)
        search_pb2_grpc.add_ShardServicer_to_server(
            ShardServicer(config.node_id, store, primary, replica), server
        )
        await server.start()
        if replica is not None:
            replica.connect()
        log.info(
            "shard=%s %s %s serving on port %d at applied_seq=%d",
            config.shard_id, config.role.value, config.node_id, port, store.applied_seq,
        )
        return cls(config, store, server, port, primary, replica)

    async def stop(self) -> None:
        """Stop replication and serving, then close the store."""
        if self.replica is not None:
            await self.replica.disconnect()
        await self.server.stop(grace=None)
        await self.store.close()


def primary_address(config: Config) -> str:
    for shard in config.shards:
        if shard.shard_id == config.shard_id:
            return shard.primary
    raise ConfigError(f"no [[shards]] entry for shard {config.shard_id}; replica can't find its primary")
