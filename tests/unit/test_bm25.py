import math

from dsearch.index.bm25 import B, K1, CorpusStats, idf, score_document


def test_idf_never_negative() -> None:
    assert idf(10, 10) > 0
    assert idf(1, 1) > 0
    assert idf(100, 1) > idf(100, 50)


def test_score_matches_formula() -> None:
    stats = CorpusStats(doc_count=4, total_length=20, doc_freq={"a": 2, "b": 1})
    tf, length = 3, 6
    norm = 1 - B + B * length / 5
    expected = idf(4, 2) * tf * (K1 + 1) / (tf + K1 * norm)
    assert math.isclose(score_document(["a", "b"], {"a": tf}, length, stats), expected)


def test_zero_avg_length_does_not_divide_by_zero() -> None:
    stats = CorpusStats(doc_count=1, total_length=0, doc_freq={"a": 1})
    assert score_document(["a"], {"a": 1}, 0, stats) > 0


def test_merge_sums_parts() -> None:
    merged = CorpusStats.merge(
        [
            CorpusStats(2, 10, {"a": 1, "b": 2}),
            CorpusStats(3, 9, {"a": 2}),
        ]
    )
    assert merged == CorpusStats(5, 19, {"a": 3, "b": 2})
    assert merged.avg_doc_length == 19 / 5
