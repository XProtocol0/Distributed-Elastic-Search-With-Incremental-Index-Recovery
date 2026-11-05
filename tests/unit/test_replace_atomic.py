"""Invariant 3: a replace is atomic with respect to search.

`apply` and `search` are synchronous, so a search running on the event loop
between a node's awaits (log fsync, applied_seq persist) never sees a doc
with its old postings removed but its new ones not yet added.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path

from dsearch.index.inverted_index import InvertedIndex
from dsearch.index.operation import Operation, OpType
from dsearch.shard.store import ShardStore
from dsearch.storage.log_entry import make_entry

DOCS = [f"d{i}" for i in range(20)]
ROUNDS = 10


def test_index_apply_and_search_never_await() -> None:
    for method in (InvertedIndex.apply, InvertedIndex.search, InvertedIndex.local_stats):
        assert not inspect.iscoroutinefunction(method)


def matching(index: InvertedIndex, term: str) -> set[str]:
    return {h.doc_id for h in index.search(term, index.local_stats(term), len(DOCS))}


def check_every_doc_matches_exactly_one_term(index: InvertedIndex) -> None:
    """Each doc holds either "alpha" or "beta": both, or neither, is a half-applied replace."""
    assert index.local_stats("alpha").doc_count == len(DOCS)
    alpha, beta = matching(index, "alpha"), matching(index, "beta")
    assert not alpha & beta, f"docs matching both old and new text: {sorted(alpha & beta)}"
    assert alpha | beta == set(DOCS), f"docs matching neither: {sorted(set(DOCS) - alpha - beta)}"


async def run_with_concurrent_searches(
    store: ShardStore, writes: Callable[[], Awaitable[None]], check: Callable[[InvertedIndex], None]
) -> int:
    """Run `writes` while a second task runs `check` at every event-loop turn; return the check count."""
    done = asyncio.Event()
    checks = 0

    async def searcher() -> None:
        nonlocal checks
        while not done.is_set():
            check(store.index)
            checks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(searcher())
    try:
        await writes()
    finally:
        done.set()
        await task
    return checks


async def open_loaded_store(path: Path) -> ShardStore:
    store = await ShardStore.open(path)
    for doc_id in DOCS:
        await store.append_new(Operation(OpType.INSERT, doc_id, 1, "alpha"))
    return store


def text_for(version: int) -> str:
    return "beta" if version % 2 == 0 else "alpha"


async def test_searches_interleaved_with_replaces_never_see_a_half_applied_doc(tmp_path: Path) -> None:
    store = await open_loaded_store(tmp_path)

    async def replaces() -> None:
        for version in range(2, ROUNDS + 2):
            for doc_id in DOCS:
                await store.append_new(Operation(OpType.REPLACE, doc_id, version, text_for(version)))

    checks = await run_with_concurrent_searches(store, replaces, check_every_doc_matches_exactly_one_term)
    await store.close()
    assert checks > ROUNDS * len(DOCS)  # the searcher really ran between the writes' awaits


async def test_a_replicated_batch_of_replaces_is_seen_all_or_nothing(tmp_path: Path) -> None:
    store = await open_loaded_store(tmp_path)

    async def batches() -> None:
        for version in range(2, ROUNDS + 2):
            first = store.applied_seq + 1
            ops = [Operation(OpType.REPLACE, d, version, text_for(version)) for d in DOCS]
            await store.write_batch([make_entry(first + i, op) for i, op in enumerate(ops)])

    seen: set[frozenset[str]] = set()

    def check(index: InvertedIndex) -> None:
        check_every_doc_matches_exactly_one_term(index)
        seen.add(frozenset(matching(index, "beta")))

    checks = await run_with_concurrent_searches(store, batches, check)
    await store.close()
    assert checks > ROUNDS
    assert seen <= {frozenset(), frozenset(DOCS)}  # never part of a batch
