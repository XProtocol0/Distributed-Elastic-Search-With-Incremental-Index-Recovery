import struct
import zlib
from pathlib import Path

import pytest

from dsearch.index.operation import Operation, OpType
from dsearch.proto import search_pb2
from dsearch.storage.log_entry import entry_to_operation, make_entry
from dsearch.storage.update_log import CorruptLogError, UpdateLog


def entry(seq: int) -> search_pb2.LogEntry:
    return make_entry(seq, Operation(OpType.INSERT, f"d{seq}", seq, f"text {seq}"))


def filled_log(path: Path, n: int) -> UpdateLog:
    log = UpdateLog.open_sync(path)
    for seq in range(1, n + 1):
        log.append_sync(entry(seq))
    return log


def test_empty_log(tmp_path: Path) -> None:
    log = UpdateLog.open_sync(tmp_path / "u.log")
    assert log.last_seq == 0 and log.size_bytes == 0
    assert log.read_after_sync(0, 10) == []


def test_append_and_read_ranges(tmp_path: Path) -> None:
    log = filled_log(tmp_path / "u.log", 10)
    assert log.last_seq == 10
    assert [e.seq for e in log.read_after_sync(0, 100)] == list(range(1, 11))
    assert [e.seq for e in log.read_after_sync(4, 3)] == [5, 6, 7]
    assert [e.seq for e in log.read_after_sync(8, 100)] == [9, 10]
    assert log.read_after_sync(10, 100) == []
    assert log.read_after_sync(0, 100)[2] == entry(3)


async def test_async_read_after_crosses_batches(tmp_path: Path) -> None:
    log = filled_log(tmp_path / "u.log", 25)
    seqs = [e.seq async for e in log.read_after(5, batch_size=4)]
    assert seqs == list(range(6, 26))


def test_rejects_gaps_and_duplicates(tmp_path: Path) -> None:
    log = filled_log(tmp_path / "u.log", 3)
    for bad in (3, 5):
        with pytest.raises(ValueError):
            log.append_sync(entry(bad))
    assert log.last_seq == 3


def test_reopen_keeps_entries_and_continues(tmp_path: Path) -> None:
    path = tmp_path / "u.log"
    filled_log(path, 5).close()
    log = UpdateLog.open_sync(path)
    assert log.last_seq == 5
    log.append_sync(entry(6))
    assert [e.seq for e in log.read_after_sync(0, 100)] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("cut", [1, 5, 9, -1])
def test_torn_tail_is_dropped(tmp_path: Path, cut: int) -> None:
    """A crash mid-append leaves a partial record; open drops only that record."""
    path = tmp_path / "u.log"
    log = filled_log(path, 3)
    size_after_two = log._offsets[2]
    log.close()
    with open(path, "r+b") as f:
        f.truncate(size_after_two + cut if cut > 0 else path.stat().st_size + cut)

    log = UpdateLog.open_sync(path)
    assert log.last_seq == 2
    assert path.stat().st_size == size_after_two
    log.append_sync(entry(3))
    assert UpdateLog.open_sync(path).last_seq == 3


def test_corrupt_last_record_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "u.log"
    filled_log(path, 3).close()
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF  # flip a payload byte of seq 3
    path.write_bytes(bytes(data))
    assert UpdateLog.open_sync(path).last_seq == 2


def test_wrong_seq_with_valid_checksum_refuses_to_open(tmp_path: Path) -> None:
    path = tmp_path / "u.log"
    filled_log(path, 2).close()
    payload = entry(7).SerializeToString()
    with open(path, "ab") as f:
        f.write(struct.pack(">II", len(payload), zlib.crc32(payload)) + payload)
    with pytest.raises(CorruptLogError, match="seq=7"):
        UpdateLog.open_sync(path)


def test_entry_conversion_round_trip() -> None:
    for op in [
        Operation(OpType.INSERT, "a", 1, "x"),
        Operation(OpType.REPLACE, "a", 2, ""),
        Operation(OpType.DELETE, "a", 3),
    ]:
        assert entry_to_operation(make_entry(9, op)) == op
    with pytest.raises(ValueError):
        entry_to_operation(search_pb2.LogEntry(seq=1, doc_id="a", version=1))
