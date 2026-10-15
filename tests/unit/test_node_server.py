import grpc

from dsearch.config import Config, Role
from dsearch.node_server import create_server
from dsearch.proto import search_pb2, search_pb2_grpc


async def test_ping_round_trip() -> None:
    config = Config(node_id="shard0-primary", role=Role.PRIMARY, shard_id=0, port=0)
    server, port = create_server(config)
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            reply = await search_pb2_grpc.NodeStub(channel).Ping(search_pb2.PingRequest())
        assert reply.node_id == "shard0-primary"
        assert reply.role == "primary"
    finally:
        await server.stop(grace=None)
