"""Conversion between protobuf `LogEntry` messages and index `Operation`s."""

from __future__ import annotations

from dsearch.index.operation import Operation, OpType
from dsearch.proto import search_pb2

_TO_PROTO = {
    OpType.INSERT: search_pb2.OP_TYPE_INSERT,
    OpType.REPLACE: search_pb2.OP_TYPE_REPLACE,
    OpType.DELETE: search_pb2.OP_TYPE_DELETE,
}
_FROM_PROTO = {v: k for k, v in _TO_PROTO.items()}


def make_entry(seq: int, op: Operation) -> search_pb2.LogEntry:
    entry = search_pb2.LogEntry(
        seq=seq, op_type=_TO_PROTO[op.op_type], doc_id=op.doc_id, version=op.version
    )
    if op.text is not None:
        entry.text = op.text
    return entry


def entry_to_operation(entry: search_pb2.LogEntry) -> Operation:
    """Raises ValueError for an entry that doesn't describe a valid operation."""
    if entry.op_type not in _FROM_PROTO:
        raise ValueError(f"seq={entry.seq}: unknown op_type {entry.op_type}")
    text = entry.text if entry.HasField("text") else None
    return Operation(_FROM_PROTO[entry.op_type], entry.doc_id, entry.version, text)
