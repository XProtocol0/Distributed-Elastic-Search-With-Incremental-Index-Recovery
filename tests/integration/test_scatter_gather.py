"""Coordinator routing and two-phase scatter-gather over a 3 x (1 + 2) in-process cluster.

Invariants 6 and 7: merged top-k equals the single-node reference in order and
exact score, whichever copies are read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import grpc
import pytest

from dsearch.client import SearchClient
from dsearch.config import SearchPreference
from dsearch.coordinator.router import shard_for
from dsearch.index.operation import Operation, OpType
from dsearch.local_cluster import LocalCluster
from dsearch.proto import search_pb2
from dsearch.reference.single_node import SingleNodeReference

WORDS = "apple banana cherry date elder fig grape honeydew kiwi lemon".split()
QUERIES = ["apple", "banana cherry", "fig grape kiwi", "lemon lemon", "nothing", "date apple elder"]


def corpus_ops() -> list[Operation]:
    """Inserts, then replaces, deletes and stale writes spread over all shards."""
    ops = [
        Operation(OpType.INSERT, f"doc{i}", 1, " ".join(WORDS[(i * j) % len(WORDS)] for j in range(1, 2 + i % 7)))
        for i in range(40)
    ]
    ops += [Operation(OpType.REPLACE, f"doc{i}", 2, f"{WORDS[i % 10]} {WORDS[(i + 3) % 10]}") for i in range(0, 40, 3)]
    ops += [Operation(OpType.DELETE, f"doc{i}", 3) for i in range(0, 40, 5)]
    ops += [Operation(OpType.INSERT, f"doc{i}", 2, "stale apple apple") for i in range(0, 40, 5)]  # blocked
    ops += [Operation(OpType.INSERT, "same1", 1, "tie tie"), Operation(OpType.INSERT, "same0", 1, "tie tie")]
    return ops


async def load(client: SearchClient, reference: SingleNodeReference, ops: list[Operation]) -> None:
    for op in ops:
        await client.write(op)
        reference.apply(op)


async def assert_matches_reference(client: SearchClient, reference: SingleNodeReference) -> None:
    for query in QUERIES + ["tie"]:
        for k in (1, 3, 10, 100):
            assert await client.search(query, k) == reference.search(query, k), (query, k)


@pytest.fixture
async def cluster(tmp_path: Path):
    cluster = await LocalCluster.start(tmp_path, num_shards=3, replicas_per_shard=2)
    client = SearchClient(cluster.coordinator_address)
    yield cluster, client
    await client.close()
    await cluster.stop()


async def test_writes_route_to_the_owning_primary(cluster) -> None:
    cluster, client = cluster
    for i in range(30):
        reply = await client.write(Operation(OpType.INSERT, f"d{i}", 1, "x"))
        assert reply.shard_id == shard_for(f"d{i}", 3)
    total = 0
    for shard_id in range(3):
        docs = cluster.node(shard_id, 0).store.index.state()
        assert all(shard_for(d, 3) == shard_id for d in docs)
        total += len(docs)
    assert total == 30


async def test_search_matches_reference_in_order_and_exact_score(cluster) -> None:
    _, client = cluster
    reference = SingleNodeReference()
    await load(client, reference, corpus_ops())
    await assert_matches_reference(client, reference)
    tie = await client.search("tie", 10)
    assert [h.doc_id for h in tie] == ["same0", "same1"]  # equal scores, doc_id order


async def test_empty_query_or_zero_k_returns_nothing(cluster) -> None:
    _, client = cluster
    await client.write(Operation(OpType.INSERT, "a", 1, "apple"))
    assert await client.search("", 10) == []
    assert await client.search("...", 10) == []
    assert await client.search("apple", 0) == []


async def test_invalid_write_status_passes_through(cluster) -> None:
    _, client = cluster
    bad = search_pb2.WriteRequest(op_type=search_pb2.OP_TYPE_INSERT, doc_id="a", version=0, text="x")
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await client._stub.Write(bad)
    assert err.value.code() is grpc.StatusCode.INVALID_ARGUMENT


async def test_failover_to_replica_when_primary_is_down(cluster) -> None:
    cluster, client = cluster
    reference = SingleNodeReference()
    await load(client, reference, corpus_ops())
    await cluster.wait_replicated()
    await cluster.stop_node(0, 0)

    reply = await client.search_reply("apple", 10)
    assert reply.copies_read[0] == cluster.base.shards[0].replicas[0]
    await assert_matches_reference(client, reference)

    doc_on_shard0 = next(f"x{i}" for i in range(100) if shard_for(f"x{i}", 3) == 0)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await client.write(Operation(OpType.INSERT, doc_on_shard0, 1, "y"))
    assert err.value.code() is grpc.StatusCode.UNAVAILABLE


async def test_search_fails_when_a_shard_has_no_reachable_copy(cluster) -> None:
    cluster, client = cluster
    await client.write(Operation(OpType.INSERT, "a", 1, "apple"))
    for copy in range(3):
        await cluster.stop_node(1, copy)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await client.search("apple", 10)
    assert err.value.code() is grpc.StatusCode.UNAVAILABLE  # never partial results


async def test_round_robin_reads_every_copy_with_identical_results(tmp_path: Path) -> None:
    cluster = await LocalCluster.start(
        tmp_path, num_shards=3, replicas_per_shard=2, search_preference=SearchPreference.ROUND_ROBIN
    )
    client = SearchClient(cluster.coordinator_address)
    try:
        reference = SingleNodeReference()
        await load(client, reference, corpus_ops())
        await cluster.wait_replicated()
        seen: list[set[str]] = [set(), set(), set()]
        for _ in range(3):
            reply = await client.search_reply("apple banana", 10)
            for shard_id, address in enumerate(reply.copies_read):
                seen[shard_id].add(address)
            assert [(h.doc_id, h.score) for h in reply.hits] == [
                (h.doc_id, h.score) for h in reference.search("apple banana", 10)
            ]
        for shard_id, addresses in enumerate(seen):
            topology = cluster.base.shards[shard_id]
            assert addresses == {topology.primary, *topology.replicas}
    finally:
        await client.close()
        await cluster.stop()


async def test_recovered_replica_serves_reference_results_through_coordinator(tmp_path: Path) -> None:
    """Disconnect a replica during replaces and deletes, reconnect, then read
    every copy through the coordinator: all must equal the reference."""
    cluster = await LocalCluster.start(
        tmp_path, num_shards=3, replicas_per_shard=2, search_preference=SearchPreference.ROUND_ROBIN
    )
    client = SearchClient(cluster.coordinator_address)
    try:
        ops = corpus_ops()
        reference = SingleNodeReference()
        await load(client, reference, ops[:40])
        await cluster.wait_replicated()

        lagging = cluster.node(1, 2).replica
        sessions = len(lagging.recoveries)
        await lagging.disconnect()
        await load(client, reference, ops[40:])
        lagging.connect()
        recovery = await asyncio.wait_for(lagging.wait_for_recovery(sessions + 1), 10)
        await cluster.wait_replicated()

        missed_on_shard1 = sum(1 for op in ops[40:] if shard_for(op.doc_id, 3) == 1)
        assert recovery.entries_replayed == missed_on_shard1 > 0
        assert lagging.full_rebuilds == 0
        for _ in range(3):  # round-robin covers every copy of every shard
            await assert_matches_reference(client, reference)
    finally:
        await client.close()
        await cluster.stop()
