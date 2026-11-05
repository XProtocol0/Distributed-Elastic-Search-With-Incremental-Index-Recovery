"""Primary/replica replication and incremental recovery over real gRPC (no Docker)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import grpc
import pytest

from dsearch.index.bm25 import CorpusStats
from dsearch.index.operation import Operation, OpType
from dsearch.proto import search_pb2
from dsearch.reference.single_node import SingleNodeReference
from dsearch.wire import hits_from_proto, operation_to_write_request, stats_from_proto, stats_to_proto
from tests.shard_cluster import (
    ShardClient,
    free_port,
    start_primary,
    start_replica,
    wait_for_recovery,
    wait_for_seq,
)


def ins(doc_id: str, version: int, text: str) -> Operation:
    return Operation(OpType.INSERT, doc_id, version, text)


def rep(doc_id: str, version: int, text: str) -> Operation:
    return Operation(OpType.REPLACE, doc_id, version, text)


def delete(doc_id: str, version: int) -> Operation:
    return Operation(OpType.DELETE, doc_id, version)


INITIAL = [ins(f"d{i}", 1, f"apple banana doc {i}") for i in range(10)]
DURING_OUTAGE = [
    rep("d1", 2, "cherry cherry"),
    delete("d2", 2),
    rep("d3", 2, "banana split"),
    ins("d2", 1, "stale resurrection"),  # older than the tombstone: no-op
    delete("d4", 5),
    rep("d1", 3, "date"),
    ins("d10", 1, "new apple"),
]


@pytest.fixture
async def cluster(tmp_path: Path):
    port = free_port()
    primary = await start_primary(tmp_path, port)
    replica = await start_replica(tmp_path, port)
    client = ShardClient(port)
    yield primary, replica, client, tmp_path, port
    await client.close()
    await replica.stop()
    await primary.stop()


async def write_all(client: ShardClient, reference: SingleNodeReference, ops: list[Operation]) -> int:
    seq = 0
    for op in ops:
        seq = (await client.write(op)).seq
        reference.apply(op)
    return seq


async def test_write_assigns_dense_seqs_and_reports_stale(cluster) -> None:
    _, _, client, _, _ = cluster
    first = await client.write(ins("a", 2, "x"))
    stale = await client.write(ins("a", 1, "older"))
    assert (first.seq, first.applied) == (1, True)
    assert (stale.seq, stale.applied) == (2, False)  # logged no-op


async def test_replica_rejects_writes_and_replicate(cluster) -> None:
    _, replica, _, _, _ = cluster
    replica_client = ShardClient(replica.port)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await replica_client.write(ins("a", 1, "x"))
    assert err.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.aio.AioRpcError) as err:
        async for _ in replica_client.stub.Replicate(search_pb2.ReplicateRequest(from_seq=0)):
            pass
    assert err.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    await replica_client.close()


async def test_invalid_write_rejected(cluster) -> None:
    _, _, client, _, _ = cluster
    bad = search_pb2.WriteRequest(op_type=search_pb2.OP_TYPE_INSERT, doc_id="a", version=0, text="x")
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await client.stub.Write(bad)
    assert err.value.code() is grpc.StatusCode.INVALID_ARGUMENT


async def test_stream_sends_catch_up_then_marker_then_live(cluster) -> None:
    _, _, client, _, _ = cluster
    for i in range(3):
        await client.write(ins(f"d{i}", 1, "x"))
    stream = client.stub.Replicate(search_pb2.ReplicateRequest(replica_id="probe", from_seq=1))
    catch_up, marker = await stream.read(), await stream.read()
    assert [e.seq for e in catch_up.entries.entries] == [2, 3]  # one batch
    assert marker.caught_up_seq == 3
    await client.write(ins("live", 1, "y"))
    live = (await stream.read()).entries.entries
    assert [(e.seq, e.doc_id) for e in live] == [(4, "live")]
    stream.cancel()


async def test_replica_ahead_of_primary_is_rejected(cluster) -> None:
    _, _, client, _, _ = cluster
    stream = client.stub.Replicate(search_pb2.ReplicateRequest(replica_id="ahead", from_seq=5))
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await stream.read()
    assert err.value.code() is grpc.StatusCode.FAILED_PRECONDITION


async def test_live_replication_matches_primary_and_reference(cluster) -> None:
    primary, replica, client, _, _ = cluster
    reference = SingleNodeReference()
    last = await write_all(client, reference, INITIAL + DURING_OUTAGE)
    await wait_for_seq(replica, last)
    replica_client = ShardClient(replica.port)
    assert await replica_client.dump() == await client.dump() == (last, reference.state())
    await replica_client.close()


async def test_disconnect_during_replaces_and_deletes_replays_only_missed(cluster) -> None:
    """Core scenario at shard level: only the missed range is replayed and the
    replica ends identical to the reference."""
    primary, replica, client, _, _ = cluster
    reference = SingleNodeReference()
    before = await write_all(client, reference, INITIAL)
    await wait_for_seq(replica, before)

    sessions = len(replica.replica.recoveries)
    await replica.replica.disconnect()
    last = await write_all(client, reference, DURING_OUTAGE)
    assert replica.store.applied_seq == before  # really missed them

    replica.replica.connect()
    recovery = await wait_for_recovery(replica, sessions + 1)
    assert (recovery.from_seq, recovery.caught_up_seq) == (before, last)
    assert recovery.entries_replayed == len(DURING_OUTAGE)
    assert replica.replica.full_rebuilds == 0
    assert replica.store.index.state() == reference.state()
    for query in ["apple", "banana cherry", "date split", "resurrection"]:
        assert replica.store.index.search(query, replica.store.index.local_stats(query), 10) == reference.search(query, 10)


async def test_replica_restart_replays_local_log_then_only_missed(cluster) -> None:
    primary, replica, client, tmp_path, port = cluster
    reference = SingleNodeReference()
    before = await write_all(client, reference, INITIAL)
    await wait_for_seq(replica, before)

    await replica.stop()  # crash/restart of the replica
    last = await write_all(client, reference, DURING_OUTAGE)

    restarted = await start_replica(tmp_path, port)
    try:
        assert restarted.store.open_report.replayed == before  # local log, no network
        recovery = await wait_for_recovery(restarted, 1)
        assert (recovery.from_seq, recovery.caught_up_seq) == (before, last)
        assert recovery.entries_replayed == len(DURING_OUTAGE)
        assert restarted.store.index.state() == reference.state()
    finally:
        await restarted.stop()


async def test_replica_reconnects_after_primary_restart(cluster) -> None:
    primary, replica, client, tmp_path, port = cluster
    reference = SingleNodeReference()
    before = await write_all(client, reference, INITIAL)
    await wait_for_seq(replica, before)

    sessions = len(replica.replica.recoveries)
    await primary.stop()
    restarted_primary = await start_primary(tmp_path, port)
    try:
        recovery = await wait_for_recovery(replica, sessions + 1)
        assert recovery.from_seq == before
        last = await write_all(client, reference, DURING_OUTAGE)
        await wait_for_seq(replica, last)
        assert replica.store.index.state() == reference.state()
    finally:
        await restarted_primary.stop()


async def test_search_rpcs_return_exact_local_results(cluster) -> None:
    primary, _, client, _, _ = cluster
    reference = SingleNodeReference()
    await write_all(client, reference, INITIAL + DURING_OUTAGE)
    query = "apple banana"
    stats = stats_from_proto(await client.stub.GetStats(search_pb2.StatsRequest(query=query)))
    assert stats == primary.store.index.local_stats(query)
    global_stats = CorpusStats(stats.doc_count + 5, stats.total_length + 40, {"apple": 9, "banana": 4})
    reply = await client.stub.SearchShard(
        search_pb2.ShardSearchRequest(query=query, global_stats=stats_to_proto(global_stats), k=5)
    )
    assert hits_from_proto(reply.hits) == primary.store.index.search(query, global_stats, 5)  # exact floats
    assert reply.applied_seq == primary.store.applied_seq


def slow_appends(node, monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    original = node.store.update_log.append_batch_sync

    def slow(entries):
        time.sleep(seconds)
        original(entries)

    monkeypatch.setattr(node.store.update_log, "append_batch_sync", slow)


async def test_disconnect_in_the_middle_of_a_replica_write(cluster, monkeypatch) -> None:
    """`disconnect()` cancels the replica mid-write; the write finishes, and
    the replica reconnects and converges instead of failing its store."""
    primary, replica, client, _, _ = cluster
    reference = SingleNodeReference()
    await wait_for_recovery(replica, 1)
    slow_appends(replica, monkeypatch, 0.2)
    await write_all(client, reference, INITIAL[:1])
    await asyncio.sleep(0.05)  # replica is now inside the slow fsync of seq 1
    await replica.replica.disconnect()
    replica.replica.connect()  # reconnects while seq 1 may still be finishing
    monkeypatch.undo()

    last = await write_all(client, reference, INITIAL[1:] + DURING_OUTAGE)
    await wait_for_seq(replica, last)
    assert replica.store.index.state() == reference.state()


async def test_client_deadline_does_not_fail_the_primary(cluster, monkeypatch) -> None:
    """A Write whose client deadline expires during the fsync still completes."""
    primary, _, client, _, _ = cluster
    slow_appends(primary, monkeypatch, 0.3)
    request = operation_to_write_request(ins("a", 1, "x"))
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await client.stub.Write(request, timeout=0.05)
    assert err.value.code() is grpc.StatusCode.DEADLINE_EXCEEDED
    monkeypatch.undo()

    reply = await client.write(ins("b", 1, "y"))  # store not failed; seq 1 was kept
    assert (reply.seq, reply.applied) == (2, True)
    assert primary.store.index.get("a") is not None
