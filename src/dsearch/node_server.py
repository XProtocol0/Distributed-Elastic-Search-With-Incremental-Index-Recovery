"""gRPC server shared by every node role.

Each role's server (coordinator, primary, replica) starts from `create_server`
and adds its own servicers; the `Node` liveness service is always present.
"""

from __future__ import annotations

import logging

import grpc

from dsearch.config import Config
from dsearch.proto import search_pb2, search_pb2_grpc

log = logging.getLogger(__name__)


class NodeServicer(search_pb2_grpc.NodeServicer):
    """Answers liveness checks with this node's id and role."""

    def __init__(self, config: Config) -> None:
        self._config = config

    async def Ping(
        self, request: search_pb2.PingRequest, context: grpc.aio.ServicerContext
    ) -> search_pb2.PingReply:
        return search_pb2.PingReply(node_id=self._config.node_id, role=self._config.role.value)


def create_server(config: Config) -> tuple[grpc.aio.Server, int]:
    """Create an asyncio gRPC server bound to the configured address.

    Returns the server (not yet started) and the bound port, which differs
    from `config.port` only when port 0 asks the OS for a free port.
    """
    server = grpc.aio.server()
    search_pb2_grpc.add_NodeServicer_to_server(NodeServicer(config), server)
    port = server.add_insecure_port(config.listen_address)
    if port == 0:
        raise RuntimeError(f"could not bind {config.listen_address}")
    log.info("bound %s:%d role=%s", config.host, port, config.role.value)
    return server, port
