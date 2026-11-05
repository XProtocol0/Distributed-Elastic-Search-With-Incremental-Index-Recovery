"""Silent network partitions between replica and primary, plus GetStatus.

The replica reaches its primary through `PartitionProxy`. While partitioned,
the proxy drops bytes without closing anything, so only keepalive can tell
the replica its stream is dead. After healing, the replica must recover by
replaying exactly the entries it missed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from dsearch.cluster_status import get_status
from dsearch.index.operation import Operation, OpType
from dsearch.reference.single_node import SingleNodeReference
from tests.shard_cluster import ShardClient, free_port, start_primary, start_replica, wait_for_recovery, wait_for_seq
from tests.tcp_proxy import PartitionProxy

KEEPALIVE = 0.3  # seconds; gRPC pings at most once a second, so detection takes ~1.5 s


async def wait_until(predicate: Callable[[], bool], timeout: float) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


async def test_partitioned_replica_detects_dead_stream_and_replays_only_missed(tmp_path: Path) -> None:
    port = free_port()
    primary = await start_primary(tmp_path, port, keepalive=KEEPALIVE)
    proxy = PartitionProxy(port)
    client = ShardClient(port)
    reference = SingleNodeReference()
    replica = None
    try:
        for i in range(20):
            op = Operation(OpType.INSERT, f"d{i}", 1, f"alpha beta {i}")
            await client.write(op)
            reference.apply(op)
        replica = await start_replica(tmp_path, await proxy.start(), keepalive=KEEPALIVE)
        assert replica.replica is not None
        await wait_for_recovery(replica, 1)  # catch-up of seq 1..20

        proxy.partition()
        missed = [
            Operation(OpType.REPLACE, "d1", 2, "gamma"),
            Operation(OpType.DELETE, "d2", 2),
            Operation(OpType.INSERT, "d2", 1, "stale insert under a tombstone"),
            Operation(OpType.REPLACE, "d3", 5, "delta gamma"),
            Operation(OpType.DELETE, "d3", 4),  # older than the replace: no-op
        ]
        for op in missed:
            await client.write(op)
            reference.apply(op)
        # Nothing reaches the replica, and nothing tells it so except keepalive.
        await wait_until(lambda: not replica.replica.live, timeout=10.0)
        assert replica.store.applied_seq == 20

        proxy.heal()
        record = await wait_for_recovery(replica, 2)
        assert (record.from_seq, record.caught_up_seq) == (20, 25)
        assert record.entries_replayed == len(missed)
        assert replica.replica.full_rebuilds == 0
        assert replica.store.index.state() == reference.state()
        assert replica.store.index.state() == primary.store.index.state()

        # The same facts over the network.
        status = await get_status(f"127.0.0.1:{replica.port}")
        assert (status.role, status.applied_seq, status.last_logged_seq, status.live) == ("replica", 25, 25, True)
        assert [(r.from_seq, r.caught_up_seq, r.entries_replayed) for r in status.recoveries] == [
            (0, 20, 20),
            (20, 25, 5),
        ]
        assert status.full_rebuilds == 0
        primary_status = await get_status(f"127.0.0.1:{port}")
        assert (primary_status.role, primary_status.applied_seq, list(primary_status.recoveries)) == (
            "primary",
            25,
            [],
        )
        assert primary_status.log_bytes == primary.store.update_log.size_bytes > 0
    finally:
        await client.close()
        if replica is not None:
            await replica.stop()
        await proxy.stop()
        await primary.stop()


async def test_status_reports_local_replay_after_restart(tmp_path: Path) -> None:
    port = free_port()
    primary = await start_primary(tmp_path, port)
    replica = await start_replica(tmp_path, port)
    client = ShardClient(port)
    try:
        for i in range(7):
            await client.write(Operation(OpType.INSERT, f"d{i}", 1, "text"))
        await wait_for_seq(replica, 7)
        await replica.stop()
        for i in range(3):
            await client.write(Operation(OpType.REPLACE, f"d{i}", 2, "new text"))

        replica = await start_replica(tmp_path, port)
        await wait_for_recovery(replica, 1)
        status = await get_status(f"127.0.0.1:{replica.port}")
        assert status.local_replayed == 7  # rebuilt from its own log, not the network
        assert [(r.from_seq, r.entries_replayed) for r in status.recoveries] == [(7, 3)]
    finally:
        await client.close()
        await replica.stop()
        await primary.stop()
