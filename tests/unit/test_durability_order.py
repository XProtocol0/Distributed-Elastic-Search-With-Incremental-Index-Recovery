"""Invariant 4: log + fsync, then apply, then persist applied_seq; crash at any step recovers."""

from pathlib import Path

import pytest

from dsearch.index.inverted_index import DocState
from dsearch.index.operation import ApplyResult, Operation, OpType
from dsearch.proto import search_pb2
from dsearch.shard.store import LOG_FILE, CorruptStateError, ShardStore, StoreFailedError
from dsearch.storage.log_entry import make_entry


def ins(seq: int, doc_id: str, version: int, text: str) -> search_pb2.LogEntry:
    return make_entry(seq, Operation(OpType.INSERT, doc_id, version, text))


def delete(seq: int, doc_id: str, version: int) -> search_pb2.LogEntry:
    return make_entry(seq, Operation(OpType.DELETE, doc_id, version))


async def test_write_logs_applies_and_persists(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path)
    assert await store.write(ins(1, "a", 1, "hello")) is ApplyResult.APPLIED
    assert await store.write(ins(2, "a", 1, "dup")) is ApplyResult.STALE  # still logged
    assert store.applied_seq == 2
    assert store.update_log.last_seq == 2
    assert store.db.applied_seq_sync() == 2
    assert store.index.get("a") == DocState(1, "hello")
    await store.close()


async def test_reopen_rebuilds_same_state(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path)
    await store.write(ins(1, "a", 1, "x"))
    await store.write(ins(2, "b", 1, "y"))
    await store.write(delete(3, "a", 2))
    before = store.index.state()
    await store.close()

    reopened = await ShardStore.open(tmp_path)
    assert reopened.index.state() == before
    assert reopened.applied_seq == 3
    assert reopened.open_report.replayed == 3


async def test_rejects_out_of_order_seq_without_logging(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path)
    await store.write(ins(1, "a", 1, "x"))
    for bad in (1, 3):
        with pytest.raises(ValueError):
            await store.write(ins(bad, "b", 1, "y"))
    assert store.update_log.last_seq == 1


@pytest.mark.parametrize("failing_step", ["apply", "persist"])
async def test_crash_after_logging_recovers_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_step: str
) -> None:
    """A crash after the fsync but before apply/persist must not lose the write,
    and the persisted applied_seq must never get ahead of the index."""
    store = await ShardStore.open(tmp_path)
    await store.write(ins(1, "a", 1, "x"))

    def boom(*args, **kwargs):
        raise OSError("simulated crash")

    if failing_step == "apply":
        monkeypatch.setattr(store.index, "apply", boom)
    else:
        monkeypatch.setattr(store.db, "set_applied_seq_sync", boom)

    with pytest.raises(OSError):
        await store.write(ins(2, "b", 1, "y"))
    assert store.db.applied_seq_sync() == 1  # never ahead of the index
    with pytest.raises(StoreFailedError):  # crash-stop: no more writes
        await store.write(ins(2, "b", 1, "y"))
    await store.close()

    reopened = await ShardStore.open(tmp_path)
    assert reopened.applied_seq == 2
    assert reopened.db.applied_seq_sync() == 2
    assert reopened.index.get("b") == DocState(1, "y")


async def test_crash_mid_append_drops_unacknowledged_entry(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path)
    await store.write(ins(1, "a", 1, "x"))
    await store.write(ins(2, "b", 1, "y"))
    await store.close()
    partial = ins(3, "c", 1, "never acknowledged").SerializeToString()[:5]
    with open(tmp_path / LOG_FILE, "ab") as f:  # crash while writing seq 3
        f.write(b"\x00\x00\x00\x40" + partial)

    reopened = await ShardStore.open(tmp_path)
    assert reopened.applied_seq == 2
    assert reopened.index.get("c") is None
    assert await reopened.write(ins(3, "c", 1, "z")) is ApplyResult.APPLIED


async def test_damaged_acknowledged_entry_is_detected(tmp_path: Path) -> None:
    """Losing an entry that was already marked applied must not go unnoticed."""
    store = await ShardStore.open(tmp_path)
    await store.write(ins(1, "a", 1, "x"))
    await store.write(ins(2, "b", 1, "y"))
    await store.close()
    log_path = tmp_path / LOG_FILE
    with open(log_path, "r+b") as f:
        f.truncate(log_path.stat().st_size - 3)
    with pytest.raises(CorruptStateError, match="applied_seq=2 last_logged_seq=1"):
        await ShardStore.open(tmp_path)


async def test_applied_seq_ahead_of_log_refuses_to_open(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path)
    await store.write(ins(1, "a", 1, "x"))
    await store.close()
    (tmp_path / LOG_FILE).unlink()  # log lost, DB says applied_seq=1
    with pytest.raises(CorruptStateError, match="applied_seq=1"):
        await ShardStore.open(tmp_path)


async def test_snapshot_bounds_replay(tmp_path: Path) -> None:
    store = await ShardStore.open(tmp_path, snapshot_every=3)
    for seq in range(1, 8):
        await store.write(ins(seq, f"d{seq % 3}", seq, f"text {seq}"))
    assert store.snapshot_seq == 6
    before = store.index.state()
    await store.close()

    reopened = await ShardStore.open(tmp_path, snapshot_every=3)
    assert reopened.open_report.snapshot_seq == 6
    assert reopened.open_report.replayed == 1  # only seq 7
    assert reopened.index.state() == before


async def test_failed_snapshot_does_not_fail_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await ShardStore.open(tmp_path, snapshot_every=1)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store.db, "write_snapshot_sync", boom)
    assert await store.write(ins(1, "a", 1, "x")) is ApplyResult.APPLIED
    assert store.snapshot_seq == 0
    assert await store.write(ins(2, "b", 1, "y")) is ApplyResult.APPLIED
