"""Invariant 5: replaying log entries again leaves the same state."""

from pathlib import Path

from dsearch.index.inverted_index import InvertedIndex
from dsearch.index.operation import Operation, OpType
from dsearch.shard.store import ShardStore
from dsearch.storage.log_entry import entry_to_operation, make_entry

OPS = [
    Operation(OpType.INSERT, "a", 1, "red apple"),
    Operation(OpType.INSERT, "b", 1, "green pear"),
    Operation(OpType.REPLACE, "a", 2, "yellow banana"),
    Operation(OpType.DELETE, "b", 3),
    Operation(OpType.INSERT, "b", 2, "stale pear"),  # blocked by the tombstone
    Operation(OpType.INSERT, "c", 1, "plum"),
]


async def write_all(path: Path, snapshot_every: int = 0) -> ShardStore:
    store = await ShardStore.open(path, snapshot_every)
    for seq, op in enumerate(OPS, start=1):
        await store.write(make_entry(seq, op))
    return store


async def test_replaying_every_entry_twice_is_a_noop(tmp_path: Path) -> None:
    store = await write_all(tmp_path)
    before = store.index.state()
    entries = [e async for e in store.update_log.read_after(0)]
    for e in entries + entries:
        store.index.apply(entry_to_operation(e))
    assert store.index.state() == before


async def test_replaying_full_log_over_snapshot_is_a_noop(tmp_path: Path) -> None:
    """Replay overlapping a snapshot (e.g. a stale snapshot_seq) changes nothing."""
    store = await write_all(tmp_path, snapshot_every=4)
    assert store.snapshot_seq == 4
    _, docs = store.db.load_snapshot_sync()
    index = InvertedIndex.from_state(docs)
    async for e in store.update_log.read_after(0):  # from 0, not from snapshot_seq
        index.apply(entry_to_operation(e))
    assert index.state() == store.index.state()


async def test_repeated_restarts_give_same_state(tmp_path: Path) -> None:
    store = await write_all(tmp_path, snapshot_every=4)
    expected = store.index.state()
    await store.close()
    for _ in range(3):
        store = await ShardStore.open(tmp_path, 4)
        assert store.index.state() == expected
        assert store.applied_seq == len(OPS)
        await store.close()
