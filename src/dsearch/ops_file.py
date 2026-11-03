"""Ops file: JSON lines recording every operation a workload sent (D-023).

The convergence checker replays it into the single-node reference. Order in
the file doesn't matter: the final state depends only on the highest version
per doc (D-003).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from dsearch.index.operation import Operation, OpType


def operation_to_json(op: Operation) -> str:
    record: dict[str, object] = {"op": op.op_type.value, "doc_id": op.doc_id, "version": op.version}
    if op.text is not None:
        record["text"] = op.text
    return json.dumps(record, ensure_ascii=False)


def operation_from_json(line: str) -> Operation:
    record = json.loads(line)
    return Operation(OpType(record["op"]), record["doc_id"], int(record["version"]), record.get("text"))


def append_ops(path: Path, ops: Iterable[Operation]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for op in ops:
            f.write(operation_to_json(op) + "\n")


def read_ops(path: Path) -> list[Operation]:
    with open(path, encoding="utf-8") as f:
        return [operation_from_json(line) for line in f if line.strip()]
