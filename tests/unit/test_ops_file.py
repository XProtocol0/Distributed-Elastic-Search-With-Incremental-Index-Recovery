from pathlib import Path

from dsearch.index.operation import Operation, OpType
from dsearch.ops_file import append_ops, read_ops

OPS = [
    Operation(OpType.INSERT, "a", 1, "Café naïve"),
    Operation(OpType.REPLACE, "a", 2, ""),
    Operation(OpType.DELETE, "a", 3),
]


def test_round_trip_in_two_appends(tmp_path: Path) -> None:
    path = tmp_path / "ops.jsonl"
    append_ops(path, OPS[:1])
    append_ops(path, OPS[1:])
    assert read_ops(path) == OPS


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "ops.jsonl"
    append_ops(path, OPS)
    path.write_text(path.read_text() + "\n\n")
    assert read_ops(path) == OPS
