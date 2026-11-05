"""Group-committed replica writes: one fsync and one applied_seq update
per batch, and the same durability order as single writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from dsearch.index.inverted_index import DocState
from dsearch.index.operation import Operation, OpType
from dsearch.proto import search_pb2
from dsearch.reference.single_node import SingleNodeReference
from dsearch.shard.store import LOG_FILE, SeqMismatchError, ShardStore
from dsearch.storage.log_entry import make_entry
from dsearch.storage.update_log import UpdateLog

OPS = [Operation(OpType.INSERT, f"d{i % 9}", i, f"text {i}") for i in range(1, 41)]
ENTRIES = [make_entry(seq, op) for seq, op in enumerate(OPS, start=1)]


def reference_of(ops: list[Operation]) -> dict[str, DocState]:
    reference = SingleNodeReference()
    for op in ops:
        reference.apply(op)
    return reference.state()


async def test_one_log_append_and_one_position_update_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await ShardStore.open(tmp_path)
    calls = {"append": 0, "persist": 0}
    append, persist = store.update_log.append_batch_sync, store.db.set_applied_seq_sync

    def counting_append(entries: list[search_pb2.LogEntry]) -> None:
        calls["append"] += 1
        append(entries)

    def counting_persist(seq: int) -> None:
        calls["persist"] += 1
        persist(seq)

    monkeypatch.setattr(store.update_log, "append_batch_sync", counting_append)
    monkeypatch.setattr(store.db, "set_applied_seq_sync", counting_persist)
    await store.write_batch(ENTRIES[:25])
    await store.write_batch(ENTRIES[25:])
    assert calls == {"append": 2, "persist": 2}
    assert store.applied_seq == 40 and store.db.applied_seq_sync() == 40
    assert store.index.state() == reference_of(OPS)
    await store.close()


@pytest.mark.parametrize("bad", [ENTRIES[1:5], [ENTRIES[0], ENTRIES[2]], [ENTRIES[0], ENTRIES[0]]])
async def test_non_dense_batch_is_rejected_before_logging(tmp_path: Path, bad: list) -> None:
    store = await ShardStore.open(tmp_path)
    with pytest.raises(SeqMismatchError):
        await store.write_batch(bad)
    assert store.update_log.last_seq == 0 and store.applied_seq == 0
    await store.write_batch(ENTRIES[:3])  # store is still healthy
    assert store.applied_seq == 3
    await store.close()


async def test_crash_part_way_through_a_batch_append_keeps_the_complete_prefix(tmp_path: Path) -> None:
    """Only part of a batch's records reached the file before the crash: no fsync
    returned, nothing was applied and applied_seq was never persisted."""
    store = await ShardStore.open(tmp_path)
    await store.write_batch(ENTRIES[:10])
    await store.close()
    scratch = UpdateLog.open_sync(tmp_path / "scratch.log", base_seq=10)
    scratch.append_batch_sync(ENTRIES[10:20])  # the bytes the next batch would have written
    scratch.close()
    batch_bytes = (tmp_path / "scratch.log").read_bytes()
    with open(tmp_path / LOG_FILE, "ab") as f:
        f.write(batch_bytes[: len(batch_bytes) * 3 // 10])  # about 3 records, the last one torn

    reopened = await ShardStore.open(tmp_path)
    kept = reopened.applied_seq
    assert 10 < kept < 20  # the complete records of the partial batch are kept and replayed
    assert reopened.index.state() == reference_of(OPS[:kept])
    assert reopened.db.applied_seq_sync() == kept
    await reopened.close()
