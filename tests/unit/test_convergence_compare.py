from dsearch.convergence import DiffKind, _compare_docs, _compare_hits, default_queries
from dsearch.index.inverted_index import DocState, SearchHit
from dsearch.index.operation import Operation, OpType
from dsearch.reference.single_node import SingleNodeReference


def test_compare_docs_reports_every_kind_of_difference() -> None:
    expected = {
        "same": DocState(1, "x"),
        "missing": DocState(1, "x"),
        "version": DocState(2, "x"),
        "tomb": DocState(3, None),
        "text": DocState(1, "x"),
    }
    actual = {
        "same": DocState(1, "x"),
        "extra": DocState(1, "y"),
        "version": DocState(1, "x"),
        "tomb": DocState(3, "back from the dead"),
        "text": DocState(1, "different"),
    }
    kinds = {d.detail.split()[0].rstrip(":"): d.kind for d in _compare_docs("c", expected, actual)}
    assert kinds == {
        "missing": DiffKind.MISSING_DOC,
        "extra": DiffKind.EXTRA_DOC,
        "version": DiffKind.VERSION,
        "tomb": DiffKind.TOMBSTONE,
        "text": DiffKind.TEXT,
    }


def test_compare_hits_is_exact() -> None:
    hits = [SearchHit("a", 1.5), SearchHit("b", 1.0)]
    assert _compare_hits(list(hits), hits) is None
    assert "rank 2" in _compare_hits([SearchHit("a", 1.5), SearchHit("b", 1.0000000000000002)], hits)
    assert "rank 1" in _compare_hits([SearchHit("b", 1.0), SearchHit("a", 1.5)], hits)
    assert "1 hits" in _compare_hits(hits[:1], hits)


def test_default_queries_are_deterministic_and_include_an_absent_term() -> None:
    reference = SingleNodeReference()
    for i in range(30):
        reference.apply(Operation(OpType.INSERT, f"d{i}", 1, f"common word{i} other{i % 4}"))
    first = default_queries(reference, count=12)
    assert first == default_queries(reference, count=12)
    assert len(first) == 12
    assert first[0] == "common"
    assert first[-1].endswith("absentterm")
    assert default_queries(SingleNodeReference()) == ["absentterm"]
