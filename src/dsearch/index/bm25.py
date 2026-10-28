"""BM25 scoring with cluster-wide statistics (see D-006).

Every node and the reference score through `score_document`, summing term
contributions in sorted term order, so equal inputs give bit-identical floats.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

K1 = 1.2
B = 0.75


@dataclass(frozen=True)
class CorpusStats:
    """Ranking statistics over live (non-tombstoned) documents.

    A shard reports its local values; the coordinator sums them with `merge`
    into the global values used for scoring.
    """

    doc_count: int = 0
    total_length: int = 0
    doc_freq: Mapping[str, int] = field(default_factory=dict)

    @property
    def avg_doc_length(self) -> float:
        return self.total_length / self.doc_count if self.doc_count else 0.0

    @staticmethod
    def merge(parts: list["CorpusStats"]) -> "CorpusStats":
        """Sum per-shard stats into global stats."""
        doc_freq: dict[str, int] = {}
        for part in parts:
            for term, df in part.doc_freq.items():
                doc_freq[term] = doc_freq.get(term, 0) + df
        return CorpusStats(
            doc_count=sum(p.doc_count for p in parts),
            total_length=sum(p.total_length for p in parts),
            doc_freq=doc_freq,
        )


def idf(doc_count: int, doc_freq: int) -> float:
    """Lucene-style IDF, never negative."""
    return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def score_document(
    sorted_terms: list[str],
    term_freqs: Mapping[str, int],
    doc_length: int,
    stats: CorpusStats,
) -> float:
    """BM25 score of one document for unique query terms given in sorted order."""
    avgdl = stats.avg_doc_length
    length_norm = (1.0 - B + B * doc_length / avgdl) if avgdl > 0 else 1.0
    score = 0.0
    for term in sorted_terms:
        tf = term_freqs.get(term, 0)
        df = stats.doc_freq.get(term, 0)
        if tf == 0 or df == 0:
            continue
        score += idf(stats.doc_count, df) * (tf * (K1 + 1.0)) / (tf + K1 * length_norm)
    return score
