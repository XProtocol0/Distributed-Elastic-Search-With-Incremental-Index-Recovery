"""Thin async client for the coordinator, used by scripts, tests and evaluation."""

from __future__ import annotations

import grpc

from dsearch.index.inverted_index import SearchHit
from dsearch.index.operation import Operation
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.wire import hits_from_proto, operation_to_write_request


class SearchClient:
    def __init__(self, address: str) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = search_pb2_grpc.CoordinatorStub(self._channel)

    async def write(self, op: Operation, timeout: float | None = None) -> search_pb2.CoordinatorWriteReply:
        """Send an insert/replace/delete. Safe to retry with the same version."""
        return await self._stub.Write(operation_to_write_request(op), timeout=timeout)

    async def search(self, query: str, k: int, timeout: float | None = None) -> list[SearchHit]:
        reply = await self._stub.Search(search_pb2.SearchRequest(query=query, k=k), timeout=timeout)
        return hits_from_proto(reply.hits)

    async def search_reply(self, query: str, k: int) -> search_pb2.SearchReply:
        """Raw reply, including which copy served each shard."""
        return await self._stub.Search(search_pb2.SearchRequest(query=query, k=k))

    async def close(self) -> None:
        await self._channel.close()
