from pathlib import Path

from dsearch.index.inverted_index import DocState
from dsearch.storage.state_db import StateDB


def test_applied_seq_defaults_to_zero_and_persists(tmp_path: Path) -> None:
    db = StateDB.open_sync(tmp_path / "s.db")
    assert db.applied_seq_sync() == 0
    db.set_applied_seq_sync(42)
    db.close()
    assert StateDB.open_sync(tmp_path / "s.db").applied_seq_sync() == 42


def test_snapshot_round_trip_includes_tombstones(tmp_path: Path) -> None:
    db = StateDB.open_sync(tmp_path / "s.db")
    assert db.load_snapshot_sync() == (0, {})
    docs = {"a": DocState(3, "hello"), "b": DocState(5, None)}
    db.write_snapshot_sync(10, docs)
    db.close()
    assert StateDB.open_sync(tmp_path / "s.db").load_snapshot_sync() == (10, docs)


def test_new_snapshot_replaces_old(tmp_path: Path) -> None:
    db = StateDB.open_sync(tmp_path / "s.db")
    db.write_snapshot_sync(1, {"a": DocState(1, "x"), "b": DocState(1, "y")})
    db.write_snapshot_sync(2, {"a": DocState(2, None)})
    assert db.load_snapshot_sync() == (2, {"a": DocState(2, None)})
