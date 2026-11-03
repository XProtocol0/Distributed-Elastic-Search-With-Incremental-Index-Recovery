"""Conversions between RPC protobuf messages and in-process types."""

from __future__ import annotations

from dsearch.index.bm25 import CorpusStats
from dsearch.index.inverted_index import DocState, SearchHit
from dsearch.index.operation import Operation
from dsearch.proto import search_pb2
from dsearch.storage.log_entry import op_type_from_proto, op_type_to_proto


def operation_to_write_request(op: Operation) -> search_pb2.WriteRequest:
    request = search_pb2.WriteRequest(
        op_type=op_type_to_proto(op.op_type), doc_id=op.doc_id, version=op.version
    )
    if op.text is not None:
        request.text = op.text
    return request


def write_request_to_operation(request: search_pb2.WriteRequest) -> Operation:
    """Raises ValueError for an invalid write (bad type, empty id, version 0, ...)."""
    text = request.text if request.HasField("text") else None
    return Operation(op_type_from_proto(request.op_type), request.doc_id, request.version, text)


def stats_to_proto(stats: CorpusStats) -> search_pb2.CorpusStats:
    return search_pb2.CorpusStats(
        doc_count=stats.doc_count, total_length=stats.total_length, doc_freq=dict(stats.doc_freq)
    )


def stats_from_proto(message: search_pb2.CorpusStats) -> CorpusStats:
    return CorpusStats(message.doc_count, message.total_length, dict(message.doc_freq))


def hits_to_proto(hits: list[SearchHit]) -> list[search_pb2.SearchHit]:
    return [search_pb2.SearchHit(doc_id=h.doc_id, score=h.score) for h in hits]


def hits_from_proto(messages: list[search_pb2.SearchHit]) -> list[SearchHit]:
    return [SearchHit(m.doc_id, m.score) for m in messages]


def state_to_proto(state: dict[str, DocState]) -> list[search_pb2.DocRecord]:
    records = []
    for doc_id in sorted(state):
        record = search_pb2.DocRecord(doc_id=doc_id, version=state[doc_id].version)
        if state[doc_id].text is not None:
            record.text = state[doc_id].text
        records.append(record)
    return records


def state_from_proto(records: list[search_pb2.DocRecord]) -> dict[str, DocState]:
    return {
        r.doc_id: DocState(r.version, r.text if r.HasField("text") else None) for r in records
    }
