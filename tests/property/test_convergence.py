"""Hypothesis: random histories (duplicates, reorders, replaces, deletes) on a
2 x (1 + 1) cluster while a random replica is disconnected or crash-restarted for
a random window. Afterwards the checker must report zero differences and the
replica must have replayed exactly the entries it missed (no full rebuild)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dsearch.client import SearchClient
from dsearch.convergence import check_convergence
from dsearch.coordinator.router import shard_for
from dsearch.local_cluster import LocalCluster
from dsearch.reference.single_node import SingleNodeReference
from tests.strategies import history, queries

NUM_SHARDS = 2


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    history(),
    st.data(),
    st.integers(0, NUM_SHARDS - 1),
    st.sampled_from(["disconnect", "restart"]),
    st.lists(queries, min_size=1, max_size=4),
)
def test_cluster_converges_after_replica_outage(h, data, shard_id, fault, qs) -> None:
    _, replay = h
    down_at = data.draw(st.integers(0, len(replay)), label="down_at")
    up_at = data.draw(st.integers(down_at, len(replay)), label="up_at")
    missed = sum(1 for op in replay[down_at:up_at] if shard_for(op.doc_id, NUM_SHARDS) == shard_id)

    async def run(tmp: Path) -> None:
        cluster = await LocalCluster.start(tmp, num_shards=NUM_SHARDS, replicas_per_shard=1)
        client = SearchClient(cluster.coordinator_address)
        reference = SingleNodeReference()
        try:
            for i, op in enumerate(replay + [None]):  # None: a final step to bring it back
                if i == down_at:
                    await cluster.wait_replicated()  # outage starts from a caught-up replica
                    if fault == "disconnect":
                        await cluster.node(shard_id, 1).replica.disconnect()
                    else:
                        await cluster.stop_node(shard_id, 1)
                if i == up_at:
                    if fault == "disconnect":
                        node = cluster.node(shard_id, 1)
                        sessions = len(node.replica.recoveries)
                        node.replica.connect()
                    else:
                        node = await cluster.start_node(shard_id, 1)
                        sessions = 0
                    recovery = await asyncio.wait_for(node.replica.wait_for_recovery(sessions + 1), 10)
                    assert recovery.entries_replayed == missed
                    assert node.replica.full_rebuilds == 0
                if op is not None:
                    await client.write(op)
                    reference.apply(op)
            await cluster.wait_replicated()

            report = await check_convergence(cluster.base.shards, reference, queries=qs)
            assert report.converged, report.summary()
        finally:
            await client.close()
            await cluster.stop()

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp)))
