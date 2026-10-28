"""Invariants 1-3: version gating, tombstones and replace."""

import pytest

from dsearch.index.inverted_index import DocState, InvertedIndex
from dsearch.index.operation import ApplyResult, Operation, OpType


def ins(doc_id: str, version: int, text: str) -> Operation:
    return Operation(OpType.INSERT, doc_id, version, text)


def rep(doc_id: str, version: int, text: str) -> Operation:
    return Operation(OpType.REPLACE, doc_id, version, text)


def delete(doc_id: str, version: int) -> Operation:
    return Operation(OpType.DELETE, doc_id, version)


def test_newer_version_applies_older_is_stale() -> None:
    index = InvertedIndex()
    assert index.apply(ins("d", 1, "old")) is ApplyResult.APPLIED
    assert index.apply(rep("d", 3, "new")) is ApplyResult.APPLIED
    assert index.apply(rep("d", 2, "delayed")) is ApplyResult.STALE
    assert index.get("d") == DocState(3, "new")


def test_duplicate_is_noop() -> None:
    index = InvertedIndex()
    index.apply(ins("d", 1, "x"))
    assert index.apply(ins("d", 1, "x")) is ApplyResult.STALE
    assert index.live_count == 1


def test_reordered_writes_end_in_same_state() -> None:
    ops = [ins("d", 1, "a"), rep("d", 2, "b"), delete("d", 3), ins("d", 4, "c")]
    forward, backward = InvertedIndex(), InvertedIndex()
    for op in ops:
        forward.apply(op)
    for op in reversed(ops):
        backward.apply(op)
    assert forward.state() == backward.state() == {"d": DocState(4, "c")}


def test_delete_leaves_tombstone_and_removes_from_search() -> None:
    index = InvertedIndex()
    index.apply(ins("d", 1, "apple"))
    assert index.apply(delete("d", 2)) is ApplyResult.APPLIED
    state = index.get("d")
    assert state == DocState(2, None) and state.deleted
    assert index.live_count == 0
    assert index.postings_snapshot() == {}
    assert index.search("apple", index.local_stats("apple"), 10) == []


def test_tombstone_blocks_older_reinsert() -> None:
    index = InvertedIndex()
    index.apply(delete("d", 5))  # delete may arrive before any insert
    assert index.apply(ins("d", 4, "zombie")) is ApplyResult.STALE
    assert index.apply(ins("d", 5, "zombie")) is ApplyResult.STALE
    assert index.get("d") == DocState(5, None)


def test_newer_insert_after_tombstone_revives() -> None:
    index = InvertedIndex()
    index.apply(ins("d", 1, "a"))
    index.apply(delete("d", 2))
    assert index.apply(ins("d", 3, "b")) is ApplyResult.APPLIED
    assert index.get("d") == DocState(3, "b")
    assert index.live_count == 1


def test_replace_swaps_postings_and_stats() -> None:
    index = InvertedIndex()
    index.apply(ins("d", 1, "cat cat dog"))
    index.apply(rep("d", 2, "fish"))
    assert index.postings_snapshot() == {"fish": {"d": 1}}
    stats = index.local_stats("cat fish")
    assert (stats.doc_count, stats.total_length) == (1, 1)
    assert stats.doc_freq == {"cat": 0, "fish": 1}


def test_search_breaks_ties_by_doc_id() -> None:
    index = InvertedIndex()
    for doc_id in ["c", "a", "b"]:
        index.apply(ins(doc_id, 1, "same words"))
    hits = index.search("same", index.local_stats("same"), 10)
    assert [h.doc_id for h in hits] == ["a", "b", "c"]
    assert len({h.score for h in hits}) == 1


def test_search_orders_by_score_and_respects_k() -> None:
    index = InvertedIndex()
    index.apply(ins("low", 1, "apple pie with many other words in it"))
    index.apply(ins("high", 1, "apple apple"))
    index.apply(ins("none", 1, "banana"))
    hits = index.search("apple", index.local_stats("apple"), 1)
    assert [h.doc_id for h in hits] == ["high"]
    assert index.search("apple", index.local_stats("apple"), 0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(op_type=OpType.INSERT, doc_id="", version=1, text="x"),
        dict(op_type=OpType.INSERT, doc_id="d", version=0, text="x"),
        dict(op_type=OpType.REPLACE, doc_id="d", version=1, text=None),
        dict(op_type=OpType.DELETE, doc_id="d", version=1, text="x"),
    ],
)
def test_invalid_operations_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Operation(**kwargs)
