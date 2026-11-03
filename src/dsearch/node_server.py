"""gRPC server and channel setup shared by every node role.

Each role's server (coordinator, primary, replica) starts from `create_server`
and adds its own servicers; the `Node` liveness service is always present.
Servers and long-lived channels use HTTP/2 keepalive so a silent network
partition is noticed within seconds (D-024).
"""

from __future__ import annotations

import asyncio
import logging
import signal

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


def channel_options(config: Config) -> list[tuple[str, int]]:
    """Client-side keepalive: ping every keepalive_time, drop the connection
    (failing its calls) if a ping isn't acked within keepalive_timeout."""
    return [
        ("grpc.keepalive_time_ms", int(config.keepalive_time * 1000)),
        ("grpc.keepalive_timeout_ms", int(config.keepalive_timeout * 1000)),
        # gRPC >= 1.6x times out unacked pings with ping_timeout_ms (default 60 s),
        # not keepalive_timeout_ms alone; without it a partition takes a minute to notice.
        ("grpc.http2.ping_timeout_ms", int(config.keepalive_timeout * 1000)),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),  # keep pinging a stream that only receives
    ]


def server_options(config: Config) -> list[tuple[str, int]]:
    """Server-side keepalive (drops streams to vanished clients) and permission
    for clients to ping as often as `channel_options` does."""
    return [
        ("grpc.keepalive_time_ms", int(config.keepalive_time * 1000)),
        ("grpc.keepalive_timeout_ms", int(config.keepalive_timeout * 1000)),
        ("grpc.http2.ping_timeout_ms", int(config.keepalive_timeout * 1000)),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_ping_interval_without_data_ms", int(config.keepalive_time * 500)),
        ("grpc.http2.max_ping_strikes", 0),  # never GOAWAY a client for pinging
    ]


def create_server(config: Config) -> tuple[grpc.aio.Server, int]:
    """Create an asyncio gRPC server bound to the configured address.

    Returns the server (not yet started) and the bound port, which differs
    from `config.port` only when port 0 asks the OS for a free port.
    """
    server = grpc.aio.server(options=server_options(config))
    search_pb2_grpc.add_NodeServicer_to_server(NodeServicer(config), server)
    port = server.add_insecure_port(config.listen_address)
    if port == 0:
        raise RuntimeError(f"could not bind {config.listen_address}")
    log.info("bound %s:%d role=%s", config.host, port, config.role.value)
    return server, port


async def wait_for_shutdown_signal() -> None:
    """Return on SIGTERM or SIGINT. In a container the node is PID 1, which
    ignores SIGTERM unless it installs a handler, so `docker stop` needs this."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
