"""A localhost TCP proxy that can simulate a network partition.

While partitioned, bytes on open connections are silently dropped in both
directions (no FIN, no reset: like `docker network disconnect` or a pulled
cable), and new connections are refused. `heal()` forwards again for new
connections; connections black-holed during the partition stay dead, as a
real TCP connection would once its peer has given up on it.
"""

from __future__ import annotations

import asyncio


class PartitionProxy:
    def __init__(self, target_port: int) -> None:
        self.target_port = target_port
        self.partitioned = False
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> int:
        """Start listening; returns the proxy's port."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    def partition(self) -> None:
        self.partitioned = True

    def heal(self) -> None:
        for writer in self._writers:
            writer.transport.abort()
        self._writers.clear()
        self.partitioned = False

    async def stop(self) -> None:
        self.heal()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        if self.partitioned:
            client_writer.transport.abort()
            return
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", self.target_port)
        except OSError:
            client_writer.transport.abort()
            return
        self._writers.update((client_writer, upstream_writer))
        await asyncio.gather(
            self._pipe(client_reader, upstream_writer), self._pipe(upstream_reader, client_writer)
        )

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(65536):
                if not self.partitioned:  # partitioned: read and drop
                    writer.write(data)
                    await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.transport.abort()
            self._writers.discard(writer)
