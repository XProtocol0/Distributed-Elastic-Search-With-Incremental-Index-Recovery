"""Log truncation after snapshots (D-030) and crash-safe snapshot install (D-031)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dsearch.index.inverted_index import DocState
from dsearch.index.operation import Operation, OpType
from dsearch.reference.single_node import SingleNodeReference
from dsearch.shard.store import CorruptStateError, ShardStore
from dsearch.storage.log_entry import make_entry
from dsearch.storage.update_log import UpdateLog

OPS = [Operation(OpType.INSERT, f"d{i % 7}", i, f"text {i}") for i in range(1, 31)]


def reference_of(ops: list[Operation]) -> dict[str, DocState]:
    reference = SingleNodeReference()
    for op in ops:
        reference.apply(op)
    return reference.state()


async def test_snapshot_truncates_behind_retention_and_reopens(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path, snapshot_every=10, log_retention=4)
    for op in OPS[:25]:
        await store.append_new(op)
    # Snapshots at 10 and 20; after the one at 20, entries <= 16 are dropped.
    assert (store.snapshot_seq, store.update_log.first_seq, store.update_log.last_seq) == (20, 17, 25)
    await store.close()

    reopened = await ShardStore.open(tmp_path, snapshot_every=10, log_retention=4)
    assert reopened.index.state() == reference_of(OPS[:25])
    assert reopened.open_report.replayed == 5  # seq 21..25 after the snapshot
    for op in OPS[25:]:
        await reopened.append_new(op)
    assert reopened.index.state() == reference_of(OPS)
    await reopened.close()


async def test_without_retention_nothing_is_truncated(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path, snapshot_every=5)
    for op in OPS:
        await store.append_new(op)
    assert store.update_log.first_seq == 1
    await store.close()


async def test_consistent_state_and_install_snapshot(tmp_path: Path) -> None:
    primary = await ShardStore.open(tmp_path / "p")
    for op in OPS:
        await primary.append_new(op)
    seq, docs = await primary.consistent_state()
    assert (seq, docs) == (30, reference_of(OPS))

    replica = await ShardStore.open(tmp_path / "r")
    for i, op in enumerate(OPS[:3], start=1):  # some stale local state
        await replica.write(make_entry(i, op))
    await replica.install_snapshot(seq, docs)
    assert (replica.applied_seq, replica.snapshot_seq, replica.update_log.first_seq) == (30, 30, 31)
    assert replica.index.state() == docs
    extra = Operation(OpType.DELETE, "d1", 99)
    await replica.write(make_entry(31, extra))
    await replica.close()

    reopened = await ShardStore.open(tmp_path / "r")
    assert reopened.index.state() == reference_of([*OPS, extra])
    assert (reopened.applied_seq, reopened.open_report.replayed) == (31, 1)
    await reopened.close()


async def test_crash_between_install_steps_reopens_from_the_snapshot(tmp_path: Path) -> None:
    """Step 1 (SQLite snapshot + applied_seq) committed, step 2 (log reset) never ran."""
    replica = await ShardStore.open(tmp_path)
    for i, op in enumerate(OPS[:3], start=1):
        await replica.write(make_entry(i, op))
    await replica.db.write_snapshot(30, reference_of(OPS), set_applied=True)
    await replica.close()  # the old log still ends at seq 3

    reopened = await ShardStore.open(tmp_path)
    assert reopened.index.state() == reference_of(OPS)
    assert (reopened.applied_seq, reopened.update_log.first_seq) == (30, 31)
    await reopened.close()


async def test_log_starting_after_the_snapshot_is_a_gap(tmp_path: Path) -> None:
    log = UpdateLog.open_sync(tmp_path / "update.log", base_seq=10)  # entries 1..10 never existed
    log.append_sync(make_entry(11, OPS[0]))
    log.close()
    with pytest.raises(CorruptStateError, match="first_logged_seq=11"):
        await ShardStore.open(tmp_path)
