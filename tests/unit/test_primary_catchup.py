"""The caught_up marker comes right after the target seq, even while writes continue (D-029)."""

from __future__ import annotations

from pathlib import Path

from dsearch.index.operation import Operation, OpType
from dsearch.shard.primary import Primary
from dsearch.shard.store import ShardStore


async def test_marker_follows_target_even_when_writes_land_during_catch_up(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path / "primary")
    primary = Primary(store, batch_size=7)
    try:
        for i in range(50):
            await store.append_new(Operation(OpType.INSERT, f"d{i}", 1, "text"))

        stream = primary.replicate("r1", from_seq=0)
        seqs_before_marker = [(await anext(stream)).entry.seq]  # catch-up has started; target = 50
        for i in range(20):  # live writes arrive mid catch-up
            await store.append_new(Operation(OpType.REPLACE, f"d{i}", 2, "new text"))

        while (message := await anext(stream)).HasField("entry"):
            seqs_before_marker.append(message.entry.seq)
        assert message.caught_up_seq == 50
        assert seqs_before_marker == list(range(1, 51))  # exactly the missed entries
        live = [(await anext(stream)).entry.seq for _ in range(20)]
        assert live == list(range(51, 71))
        await stream.aclose()
    finally:
        await store.close()
