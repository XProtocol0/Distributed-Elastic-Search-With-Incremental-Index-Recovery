"""Hypothesis checks for the index against a 'highest version wins' model."""

from __future__ import annotations

import zlib

from hypothesis import given, settings
from hypothesis import strategies as st

from dsearch.index.bm25 import CorpusStats
from dsearch.index.inverted_index import DocState, InvertedIndex, SearchHit
from dsearch.index.operation import Operation, OpType
from dsearch.reference.single_node import SingleNodeReference
from tests.strategies import expected_state, history, queries

def build(ops: list[Operation]) -> InvertedIndex:
    index = InvertedIndex()
    for op in ops:
        index.apply(op)
    return index


def fresh_index_of_live_docs(state: dict[str, DocState]) -> InvertedIndex:
    """An index built from scratch with only the final live documents."""
    index = InvertedIndex()
    for doc_id, doc in state.items():
        if not doc.deleted:
            index.apply(Operation(OpType.INSERT, doc_id, doc.version, doc.text))
    return index


@given(history())
def test_state_is_highest_version_regardless_of_order_and_duplicates(h) -> None:
    ops, replay = h
    assert build(replay).state() == expected_state(ops)


@given(history(), st.lists(queries, min_size=1, max_size=5))
def test_any_order_gives_identical_search_results(h, qs) -> None:
    ops, replay = h
    in_order, shuffled = SingleNodeReference(), SingleNodeReference()
    for op in ops:
        in_order.apply(op)
    for op in replay:
        shuffled.apply(op)
    for q in qs:
        assert in_order.search(q, 10) == shuffled.search(q, 10)


@given(history(), st.lists(queries, min_size=1, max_size=5))
def test_postings_and_stats_equal_a_fresh_rebuild(h, qs) -> None:
    """Replaces and deletes leave no stale postings or stats (Invariants 2, 3)."""
    _, replay = h
    index = build(replay)
    fresh = fresh_index_of_live_docs(index.state())
    assert index.postings_snapshot() == fresh.postings_snapshot()
    for q in qs:
        assert index.local_stats(q) == fresh.local_stats(q)
        stats = index.local_stats(q)
        assert index.search(q, stats, 10) == fresh.search(q, stats, 10)


def shard_of(doc_id: str, num_shards: int) -> int:
    return zlib.crc32(doc_id.encode()) % num_shards


@settings(max_examples=200)
@given(history(), st.lists(queries, min_size=1, max_size=5), st.integers(1, 4), st.integers(1, 10))
def test_sharded_top_k_with_merged_stats_equals_reference(h, qs, num_shards, k) -> None:
    """Index-level check of Invariants 6 and 7: merging per-shard top-k scored with
    merged global stats gives exactly the reference's order and scores."""
    _, replay = h
    reference = SingleNodeReference()
    shards = [InvertedIndex() for _ in range(num_shards)]
    for op in replay:
        reference.apply(op)
        shards[shard_of(op.doc_id, num_shards)].apply(op)

    for q in qs:
        global_stats = CorpusStats.merge([s.local_stats(q) for s in shards])
        hits: list[SearchHit] = []
        for shard in shards:
            hits.extend(shard.search(q, global_stats, k))
        merged = sorted(hits, key=lambda hit: (-hit.score, hit.doc_id))[:k]
        assert merged == reference.search(q, k)  # exact float equality (D-006)
