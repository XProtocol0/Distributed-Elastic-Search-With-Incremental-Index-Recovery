"""In-memory inverted index holding one shard's documents, versions and tombstones.

Concurrency rule (D-007): the index is only touched from the event-loop thread,
and `apply` and `search` never await, so a replace is atomic with respect to
search. Never call them through `asyncio.to_thread`.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

from dsearch.index.bm25 import CorpusStats, score_document
from dsearch.index.operation import ApplyResult, Operation, OpType
from dsearch.index.tokenizer import query_terms, tokenize


@dataclass(frozen=True)
class DocState:
    """What is stored for one doc id; `text is None` means a tombstone."""

    version: int
    text: str | None

    @property
    def deleted(self) -> bool:
        return self.text is None


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float


@dataclass
class _DocRecord:
    version: int
    text: str | None
    term_freqs: dict[str, int]
    length: int


class InvertedIndex:
    def __init__(self) -> None:
        self._docs: dict[str, _DocRecord] = {}
        self._postings: dict[str, dict[str, int]] = {}  # term -> {doc_id: tf}
        self._live_count = 0
        self._total_length = 0

    # ---- writes ---------------------------------------------------------

    def apply(self, op: Operation) -> ApplyResult:
        """Apply `op` if its version is newer than the stored one (tombstones included).

        Stale, duplicate and reordered writes return STALE and change nothing.
        """
        current = self._docs.get(op.doc_id)
        if current is not None and op.version <= current.version:
            return ApplyResult.STALE

        if current is not None and current.text is not None:
            self._remove_postings(op.doc_id, current)

        if op.op_type is OpType.DELETE:
            self._docs[op.doc_id] = _DocRecord(op.version, None, {}, 0)
        else:
            assert op.text is not None
            tokens = tokenize(op.text)
            record = _DocRecord(op.version, op.text, dict(Counter(tokens)), len(tokens))
            self._docs[op.doc_id] = record
            self._add_postings(op.doc_id, record)
        return ApplyResult.APPLIED

    def _add_postings(self, doc_id: str, record: _DocRecord) -> None:
        for term, tf in record.term_freqs.items():
            self._postings.setdefault(term, {})[doc_id] = tf
        self._live_count += 1
        self._total_length += record.length

    def _remove_postings(self, doc_id: str, record: _DocRecord) -> None:
        for term in record.term_freqs:
            docs = self._postings[term]
            del docs[doc_id]
            if not docs:
                del self._postings[term]
        self._live_count -= 1
        self._total_length -= record.length

    # ---- reads ----------------------------------------------------------

    def get(self, doc_id: str) -> DocState | None:
        record = self._docs.get(doc_id)
        return None if record is None else DocState(record.version, record.text)

    def state(self) -> dict[str, DocState]:
        """Every stored doc id (live and tombstoned) with its version and text."""
        return {doc_id: DocState(r.version, r.text) for doc_id, r in self._docs.items()}

    @property
    def live_count(self) -> int:
        return self._live_count

    def local_stats(self, query: str) -> CorpusStats:
        """This index's share of the ranking stats for `query`'s terms."""
        return CorpusStats(
            doc_count=self._live_count,
            total_length=self._total_length,
            doc_freq={t: len(self._postings.get(t, {})) for t in query_terms(query)},
        )

    def search(self, query: str, stats: CorpusStats, k: int) -> list[SearchHit]:
        """Top-k live docs for `query`, scored with the given (global) stats.

        Ordered by score descending, ties broken by doc_id ascending.
        """
        terms = query_terms(query)
        if k <= 0 or not terms:
            return []
        candidates: set[str] = set()
        for term in terms:
            candidates.update(self._postings.get(term, {}))
        hits: list[SearchHit] = []
        for doc_id in candidates:
            record = self._docs[doc_id]
            score = score_document(terms, record.term_freqs, record.length, stats)
            hits.append(SearchHit(doc_id, score))
        return heapq.nsmallest(k, hits, key=lambda h: (-h.score, h.doc_id))

    def postings_snapshot(self) -> dict[str, dict[str, int]]:
        """Copy of the postings, for tests that compare index structure."""
        return {term: dict(docs) for term, docs in self._postings.items()}
