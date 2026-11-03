"""Two-phase scatter-gather search with global BM25 statistics (D-018, D-019).

Phase 1: ask one copy of every shard for its share of the stats for the
         query's terms (trying the shard's copies in preference order) and
         sum them into global stats.
Phase 2: ask the same copies for their local top-k scored with the global
         stats, then merge into the global top-k by (-score, doc_id).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from dataclasses import dataclass

import grpc

from dsearch.config import SearchPreference, ShardTopology
from dsearch.index.bm25 import CorpusStats
from dsearch.index.inverted_index import SearchHit
from dsearch.index.tokenizer import query_terms
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.wire import hits_from_proto, stats_from_proto, stats_to_proto

log = logging.getLogger(__name__)


class ShardUnavailableError(Exception):
    """No copy of a shard could serve its part of the search."""


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    copies_read: list[str]  # address of the copy read for each shard, by shard id


def merge_top_k(per_shard: list[list[SearchHit]], k: int) -> list[SearchHit]:
    """Merge per-shard top-k lists (each sorted by -score, doc_id) into the global top-k."""
    merged = heapq.merge(*per_shard, key=lambda h: (-h.score, h.doc_id))
    return list(itertools.islice(merged, k))


class ScatterGather:
    def __init__(
        self,
        shards: tuple[ShardTopology, ...],
        preference: SearchPreference,
        rpc_timeout: float,
        channel_options: list[tuple[str, int]] | None = None,
    ) -> None:
        self._shards = sorted(shards, key=lambda s: s.shard_id)
        self._channel_options = channel_options or []  # keepalive (D-024)
        self._preference = preference
        self._timeout = rpc_timeout
        self._next_copy = {s.shard_id: 0 for s in self._shards}  # round-robin cursors
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._stubs: dict[str, search_pb2_grpc.ShardStub] = {}

    def stub(self, address: str) -> search_pb2_grpc.ShardStub:
        """Shared channel per shard-node address, created on first use."""
        if address not in self._stubs:
            self._channels[address] = grpc.aio.insecure_channel(address, options=self._channel_options)
            self._stubs[address] = search_pb2_grpc.ShardStub(self._channels[address])
        return self._stubs[address]

    async def close(self) -> None:
        for channel in self._channels.values():
            await channel.close()

    def copy_order(self, shard: ShardTopology) -> list[str]:
        """The shard's copies in the order to try them for one query."""
        copies = [shard.primary, *shard.replicas]
        if self._preference is SearchPreference.PRIMARY:
            return copies
        start = self._next_copy[shard.shard_id] % len(copies)
        self._next_copy[shard.shard_id] += 1
        return copies[start:] + copies[:start]

    async def search(self, query: str, k: int) -> SearchResult:
        """Global top-k for `query`, identical to the single-node reference."""
        if k <= 0 or not query_terms(query):
            return SearchResult([], [])
        phase1 = await asyncio.gather(
            *(self._stats_from_first_reachable(shard, query) for shard in self._shards)
        )
        return await self._search_phase2(query, k, phase1)

    async def search_on_copies(self, query: str, k: int, copies: list[str]) -> SearchResult:
        """Like `search`, but read exactly `copies[shard_id]` for each shard, with no
        failover. Used by the convergence checker to test one replica at a time."""
        if k <= 0 or not query_terms(query):
            return SearchResult([], [])
        phase1 = await asyncio.gather(
            *(self._stats_from_copy(shard.shard_id, address, query)
              for shard, address in zip(self._shards, copies))
        )
        return await self._search_phase2(query, k, phase1)

    async def _search_phase2(
        self, query: str, k: int, phase1: list[tuple[str, CorpusStats]]
    ) -> SearchResult:
        """Score on the copies chosen in phase 1 with the summed stats, then merge."""
        copies = [address for address, _ in phase1]
        global_stats = CorpusStats.merge([stats for _, stats in phase1])
        per_shard = await asyncio.gather(
            *(
                self._search_copy(shard.shard_id, address, query, global_stats, k)
                for shard, address in zip(self._shards, copies)
            )
        )
        hits = merge_top_k(per_shard, k)
        log.debug("search q=%r k=%d copies=%s -> %d hits", query, k, copies, len(hits))
        return SearchResult(hits, copies)

    async def _stats_from_copy(self, shard_id: int, address: str, query: str) -> tuple[str, CorpusStats]:
        try:
            reply = await self.stub(address).GetStats(
                search_pb2.StatsRequest(query=query), timeout=self._timeout
            )
        except grpc.aio.AioRpcError as exc:
            raise ShardUnavailableError(f"shard {shard_id} copy {address}: {exc.code().name}") from exc
        return address, stats_from_proto(reply)

    async def _stats_from_first_reachable(
        self, shard: ShardTopology, query: str
    ) -> tuple[str, CorpusStats]:
        for address in self.copy_order(shard):
            try:
                return await self._stats_from_copy(shard.shard_id, address, query)
            except ShardUnavailableError as exc:
                log.warning("%s; trying next copy", exc)
        raise ShardUnavailableError(f"no copy of shard {shard.shard_id} answered")

    async def _search_copy(
        self, shard_id: int, address: str, query: str, global_stats: CorpusStats, k: int
    ) -> list[SearchHit]:
        request = search_pb2.ShardSearchRequest(
            query=query, global_stats=stats_to_proto(global_stats), k=k
        )
        try:
            reply = await self.stub(address).SearchShard(request, timeout=self._timeout)
        except grpc.aio.AioRpcError as exc:
            # Switching copies now would mix one copy's stats with another's postings.
            raise ShardUnavailableError(
                f"shard {shard_id} copy {address} failed between search phases: {exc.code().name}"
            ) from exc
        return hits_from_proto(reply.hits)
