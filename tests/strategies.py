"""Shared Hypothesis strategies: operation histories with one operation per
(doc_id, version) pair (D-003), replayed with duplicates and in random order."""

from __future__ import annotations

from hypothesis import strategies as st

from dsearch.index.inverted_index import DocState
from dsearch.index.operation import Operation, OpType

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
