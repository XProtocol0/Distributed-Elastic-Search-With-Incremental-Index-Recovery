"""Full rebuild over real gRPC (no Docker): caused by log truncation
or forced for the evaluation baseline. Either way it must be counted,
and the replica must end in exactly the reference state."""

from __future__ import annotations

import random
from pathlib import Path

from dsearch.cluster_status import get_status
from dsearch.index.operation import Operation, OpType
from dsearch.reference.single_node import SingleNodeReference
from dsearch.workload import CorpusDoc, DocVersions, churn_ops, insert_ops
from tests.shard_cluster import ShardClient, free_port, start_primary, start_replica, wait_for_recovery, wait_for_seq

DOCS = [CorpusDoc(f"d{i}", f"alpha beta gamma {i}") for i in range(60)]
TEXTS = ["delta epsilon", "alpha omega", "beta beta zeta", "gamma"]


def workload(count: int, seed: int) -> tuple[list[Operation], list[Operation]]:
    """Initial inserts, then `count` churn ops (replace/delete/re-insert/stale/duplicate)."""
    load = insert_ops(DOCS)
    return load, churn_ops(DocVersions.from_ops(load), TEXTS, count, random.Random(seed))


async def write_all(client: ShardClient, reference: SingleNodeReference, ops: list[Operation]) -> None:
    for op in ops:
        await client.write(op)
        reference.apply(op)


async def test_truncated_log_triggers_a_counted_full_rebuild(tmp_path: Path) -> None:
    port = free_port()
    primary = await start_primary(tmp_path, port, snapshot_every=20, log_retention=10)
    client = ShardClient(port)
    reference = SingleNodeReference()
    load, churn = workload(200, seed=1)
    replica = await start_replica(tmp_path, port)
    try:
        await write_all(client, reference, load)
        await wait_for_seq(replica, 60)
        await replica.stop()  # down at seq 60

        await write_all(client, reference, churn)  # 200 more; primary truncates far past 60
        assert primary.store.update_log.first_seq > 61

        replica = await start_replica(tmp_path, port)
        record = await wait_for_recovery(replica, 1)
        assert replica.replica is not None and replica.replica.full_rebuilds == 1
        assert record.full_rebuild and record.from_seq == 60
        assert record.snapshot_docs == len(reference.state())  # tombstones included
        assert record.caught_up_seq == 260
        assert replica.store.index.state() == reference.state()
        assert replica.store.update_log.first_seq == record.caught_up_seq - record.entries_replayed + 1

        status = await get_status(f"127.0.0.1:{replica.port}")
        assert status.full_rebuilds == 1 and status.recoveries[0].full_rebuild

        # It replicates normally afterwards, and a restart recovers locally (no second rebuild).
        await write_all(client, reference, churn_ops(DocVersions.from_ops(load + churn), TEXTS, 30, random.Random(2)))
        await wait_for_seq(replica, 290)
        await replica.stop()
        replica = await start_replica(tmp_path, port)
        record = await wait_for_recovery(replica, 1)
        assert not record.full_rebuild and record.from_seq == 290 and record.entries_replayed == 0
        assert replica.store.index.state() == reference.state()
    finally:
        await client.close()
        await replica.stop()
        await primary.stop()


async def test_forced_full_rebuild_ends_in_the_same_state_as_incremental(tmp_path: Path) -> None:
    port = free_port()
    primary = await start_primary(tmp_path, port)
    client = ShardClient(port)
    reference = SingleNodeReference()
    load, churn = workload(150, seed=3)
    replica = await start_replica(tmp_path, port, node_id="r1")
    try:
        await write_all(client, reference, load)
        await wait_for_seq(replica, 60)
        await replica.stop()
        await write_all(client, reference, churn)

        incremental = await start_replica(tmp_path, port, node_id="r1")
        inc_record = await wait_for_recovery(incremental, 1)
        await incremental.stop()
        forced = await start_replica(tmp_path, port, node_id="r1", force_full_rebuild=True)
        replica = forced
        full_record = await wait_for_recovery(forced, 1)

        assert not inc_record.full_rebuild and inc_record.entries_replayed == 150
        assert full_record.full_rebuild and full_record.entries_replayed == 0  # nothing after the snapshot
        assert forced.replica is not None and forced.replica.full_rebuilds == 1
        assert forced.store.index.state() == reference.state() == primary.store.index.state()
    finally:
        await client.close()
        await replica.stop()
        await primary.stop()


async def test_full_rebuild_of_a_shard_larger_than_one_grpc_message(tmp_path: Path) -> None:
    """Snapshot chunks are bounded by size, so ~6 MB of state still transfers."""
    port = free_port()
    primary = await start_primary(tmp_path, port)
    client = ShardClient(port)
    reference = SingleNodeReference()
    big = [Operation(OpType.INSERT, f"big{i}", 1, f"word{i} " + "lorem ipsum " * 2000) for i in range(250)]
    try:
        await write_all(client, reference, big)
        replica = await start_replica(tmp_path, port, force_full_rebuild=True)
        record = await wait_for_recovery(replica, 1)
        assert record.full_rebuild and record.snapshot_docs == 250
        assert replica.store.index.state() == reference.state()
        await replica.stop()
    finally:
        await client.close()
        await primary.stop()
