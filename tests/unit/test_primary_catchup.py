"""Replication stream framing: the caught_up marker comes right after the target
seq even while writes continue, and entries travel in size-bounded
batches."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from dsearch.index.operation import Operation, OpType
from dsearch.proto import search_pb2
from dsearch.shard.primary import Primary
from dsearch.shard.store import ShardStore


def seqs(message: search_pb2.ReplicateMessage) -> list[int]:
    return [e.seq for e in message.entries.entries]


async def test_marker_follows_target_even_when_writes_land_during_catch_up(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path / "primary")
    primary = Primary(store, batch_size=7)
    try:
        for i in range(50):
            await store.append_new(Operation(OpType.INSERT, f"d{i}", 1, "text"))

        stream = primary.replicate("r1", from_seq=0)
        first = await anext(stream)  # catch-up has started; target = 50
        assert seqs(first) == list(range(1, 8))
        before_marker = seqs(first)
        for i in range(20):  # live writes arrive mid catch-up
            await store.append_new(Operation(OpType.REPLACE, f"d{i}", 2, "new text"))

        while (message := await anext(stream)).HasField("entries"):
            assert len(message.entries.entries) <= 7
            before_marker += seqs(message)
        assert message.caught_up_seq == 50
        assert before_marker == list(range(1, 51))  # exactly the missed entries
        live: list[int] = []
        while len(live) < 20:
            live += seqs(await anext(stream))
        assert live == list(range(51, 71))
        await stream.aclose()
    finally:
        await store.close()


async def test_batches_are_bounded_by_bytes(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path / "primary")
    primary = Primary(store, batch_size=100, batch_bytes=10_000)
    try:
        for i in range(30):
            await store.append_new(Operation(OpType.INSERT, f"d{i}", 1, "x" * 2000))
        stream: AsyncIterator[search_pb2.ReplicateMessage] = primary.replicate("r1", from_seq=0)
        received: list[int] = []
        while (message := await anext(stream)).HasField("entries"):
            assert message.entries.ByteSize() < 10_000 + 2_100  # at most one entry over the cap
            received += seqs(message)
        assert received == list(range(1, 31)) and message.caught_up_seq == 30
        await stream.aclose()
    finally:
        await store.close()
