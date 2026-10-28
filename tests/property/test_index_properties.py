"""Hypothesis checks for the index against a 'highest version wins' model.

Generated histories assign one operation per (doc_id, version) (D-003), then
replay them with duplicates and in random order.
"""

from __future__ import annotations

import zlib

from hypothesis import given, settings
from hypothesis import strategies as st

from dsearch.index.bm25 import CorpusStats
from dsearch.index.inverted_index import DocState, InvertedIndex, SearchHit
from dsearch.index.operation import Operation, OpType
from dsearch.reference.single_node import SingleNodeReference

WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]
DOC_IDS = [f"doc{i}" for i in range(8)]

texts = st.lists(st.sampled_from(WORDS), min_size=0, max_size=12).map(" ".join)
queries = st.lists(st.sampled_from(WORDS), min_size=1, max_size=3).map(" ".join)


@st.composite
def unique_ops(draw: st.DrawFn) -> list[Operation]:
    """A set of operations with distinct (doc_id, version) pairs."""
    keys = draw(
        st.lists(
            st.tuples(st.sampled_from(DOC_IDS), st.integers(1, 6)),
            unique=True,
            max_size=30,
        )
    )
    ops = []
    for doc_id, version in keys:
        op_type = draw(st.sampled_from(list(OpType)))
        text = None if op_type is OpType.DELETE else draw(texts)
        ops.append(Operation(op_type, doc_id, version, text))
    return ops


@st.composite
def history(draw: st.DrawFn) -> tuple[list[Operation], list[Operation]]:
    """(unique ops, a shuffled replay of them with duplicates)."""
    ops = draw(unique_ops())
    duplicates = draw(st.lists(st.sampled_from(ops), max_size=10)) if ops else []
    replay = draw(st.permutations(ops + duplicates))
    return ops, replay


def expected_state(ops: list[Operation]) -> dict[str, DocState]:
    latest: dict[str, Operation] = {}
    for op in ops:
        if op.doc_id not in latest or op.version > latest[op.doc_id].version:
            latest[op.doc_id] = op
    return {d: DocState(op.version, op.text) for d, op in latest.items()}


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
