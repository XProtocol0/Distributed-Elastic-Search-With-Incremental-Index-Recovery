"""Single-node reference: the same index code in one process, used as the oracle.

It is fed the same operations as the cluster. Its stats are already global,
because it holds every document.
"""

from __future__ import annotations

from dsearch.index.inverted_index import DocState, InvertedIndex, SearchHit
from dsearch.index.operation import ApplyResult, Operation


class SingleNodeReference:
    def __init__(self) -> None:
        self.index = InvertedIndex()

    def apply(self, op: Operation) -> ApplyResult:
        return self.index.apply(op)

    def search(self, query: str, k: int) -> list[SearchHit]:
        return self.index.search(query, self.index.local_stats(query), k)

    def state(self) -> dict[str, DocState]:
        return self.index.state()
