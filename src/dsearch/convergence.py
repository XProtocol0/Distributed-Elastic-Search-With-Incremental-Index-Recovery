"""Convergence checker: compare every copy of every shard with the reference (D-022).

For each copy (primary and replicas) it checks:
  * documents, versions and tombstones (DumpState) against the reference's
    state for that shard;
  * applied_seq against the shard's primary (a lagging copy hasn't converged);
  * query results: a full two-phase search reading exactly that copy for its
    shard must equal reference.search in order and exact score.

Every difference is reported; a converged cluster has zero.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import grpc

from dsearch.config import SearchPreference, ShardTopology
from dsearch.coordinator.router import shard_for
from dsearch.coordinator.search import ScatterGather, ShardUnavailableError
from dsearch.index.inverted_index import DocState, SearchHit
from dsearch.index.tokenizer import tokenize
from dsearch.proto import search_pb2
from dsearch.reference.single_node import SingleNodeReference
from dsearch.wire import state_from_proto


class DiffKind(str, Enum):
    UNREACHABLE = "unreachable"
    MISSING_DOC = "missing_doc"
    EXTRA_DOC = "extra_doc"
    VERSION = "version"
    TOMBSTONE = "tombstone"
    TEXT = "text"
    APPLIED_SEQ = "applied_seq"
    SEARCH = "search"


@dataclass(frozen=True)
class Difference:
    kind: DiffKind
    copy: str  # "shard<id> <address>"
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.copy}: {self.detail}"


@dataclass
class ConvergenceReport:
    differences: list[Difference] = field(default_factory=list)
    copies_checked: int = 0
    docs_checked: int = 0
    queries_checked: int = 0

    @property
    def converged(self) -> bool:
        return not self.differences

    def summary(self) -> str:
        head = (
            f"{len(self.differences)} differences "
            f"({self.copies_checked} copies, {self.docs_checked} doc checks, "
            f"{self.queries_checked} queries per copy)"
        )
        return "\n".join([head, *(f"  {d}" for d in self.differences)])


def default_queries(reference: SingleNodeReference, count: int = 20, seed: int = 0) -> list[str]:
    """Deterministic query set from the reference's vocabulary (D-022)."""
    df: Counter[str] = Counter()
    for doc in reference.state().values():
        if doc.text is not None:
            df.update(set(tokenize(doc.text)))
    if not df:
        return ["absentterm"]
    by_df = sorted(df, key=lambda t: (-df[t], t))
    frequent = by_df[:50]
    rare = by_df[-50:]
    rng = random.Random(seed)
    queries = frequent[:5]
    while len(queries) < count - 1:
        size = rng.randint(2, 3)
        terms = rng.sample(frequent, min(size - 1, len(frequent))) + [rng.choice(rare)]
        queries.append(" ".join(terms))
    queries.append(f"{frequent[0]} absentterm")
    return queries


async def check_convergence(
    shards: tuple[ShardTopology, ...],
    reference: SingleNodeReference,
    queries: list[str] | None = None,
    k: int = 10,
    rpc_timeout: float = 5.0,
) -> ConvergenceReport:
    """Compare every copy of every shard with `reference`; see module docstring."""
    queries = default_queries(reference) if queries is None else queries
    shards = tuple(sorted(shards, key=lambda s: s.shard_id))
    num_shards = len(shards)
    expected_by_shard: list[dict[str, DocState]] = [{} for _ in shards]
    for doc_id, doc in reference.state().items():
        expected_by_shard[shard_for(doc_id, num_shards)][doc_id] = doc
    expected_hits = {q: reference.search(q, k) for q in queries}

    report = ConvergenceReport()
    # A whole shard's DumpState easily exceeds gRPC's default 4 MB receive limit.
    scatter = ScatterGather(
        shards, SearchPreference.PRIMARY, rpc_timeout, [("grpc.max_receive_message_length", -1)]
    )
    try:
        for shard in shards:
            primary_seq: int | None = None
            for address in (shard.primary, *shard.replicas):
                name = f"shard{shard.shard_id} {address}"
                report.copies_checked += 1
                try:
                    reply = await scatter.stub(address).DumpState(
                        search_pb2.DumpStateRequest(), timeout=rpc_timeout
                    )
                except grpc.aio.AioRpcError as exc:
                    report.differences.append(Difference(DiffKind.UNREACHABLE, name, exc.code().name))
                    continue

                if address == shard.primary:
                    primary_seq = reply.applied_seq
                elif primary_seq is not None and reply.applied_seq != primary_seq:
                    report.differences.append(
                        Difference(DiffKind.APPLIED_SEQ, name, f"applied_seq={reply.applied_seq}, primary has {primary_seq}")
                    )
                expected = expected_by_shard[shard.shard_id]
                report.differences.extend(_compare_docs(name, expected, state_from_proto(reply.docs)))
                report.docs_checked += len(expected)

                copies = [s.primary for s in shards]
                copies[shard.shard_id] = address
                for query in queries:
                    try:
                        result = await scatter.search_on_copies(query, k, copies)
                    except ShardUnavailableError as exc:
                        report.differences.append(Difference(DiffKind.UNREACHABLE, name, str(exc)))
                        break
                    diff = _compare_hits(result.hits, expected_hits[query])
                    if diff:
                        report.differences.append(Difference(DiffKind.SEARCH, name, f"query {query!r}: {diff}"))
        report.queries_checked = len(queries)
    finally:
        await scatter.close()
    return report


def _compare_docs(name: str, expected: dict[str, DocState], actual: dict[str, DocState]) -> list[Difference]:
    diffs = []
    for doc_id in sorted(expected.keys() | actual.keys()):
        want, got = expected.get(doc_id), actual.get(doc_id)
        if got is None:
            diffs.append(Difference(DiffKind.MISSING_DOC, name, f"{doc_id} (expected v{want.version})"))
        elif want is None:
            diffs.append(Difference(DiffKind.EXTRA_DOC, name, f"{doc_id} v{got.version}"))
        elif want.version != got.version:
            diffs.append(Difference(DiffKind.VERSION, name, f"{doc_id}: v{got.version}, expected v{want.version}"))
        elif want.deleted != got.deleted:
            state = "tombstone" if got.deleted else "live"
            diffs.append(Difference(DiffKind.TOMBSTONE, name, f"{doc_id} v{got.version} is {state}"))
        elif want.text != got.text:
            diffs.append(Difference(DiffKind.TEXT, name, f"{doc_id} v{got.version}: text differs"))
    return diffs


def _compare_hits(actual: list[SearchHit], expected: list[SearchHit]) -> str | None:
    """None if identical (order and exact score), else a description of the first mismatch."""
    if actual == expected:
        return None
    for rank, (got, want) in enumerate(zip(actual, expected), start=1):
        if got != want:
            return f"rank {rank}: got {got.doc_id} {got.score!r}, expected {want.doc_id} {want.score!r}"
    return f"got {len(actual)} hits, expected {len(expected)}"
