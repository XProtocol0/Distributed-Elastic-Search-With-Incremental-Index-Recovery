"""Hypothesis: disconnect a replica for a random stretch of a random history
(duplicates and reorders included), reconnect it, and check it converges to the
reference: documents, versions and search results (order and exact scores).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dsearch.reference.single_node import SingleNodeReference
from tests.shard_cluster import (
    ShardClient,
    free_port,
    start_primary,
    start_replica,
    wait_for_recovery,
    wait_for_seq,
)
from tests.strategies import history, queries


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(history(), st.data(), st.lists(queries, min_size=1, max_size=4))
def test_replica_converges_after_disconnect(h, data, qs) -> None:
    _, replay = h
    down_at = data.draw(st.integers(0, len(replay)), label="down_at")
    up_at = data.draw(st.integers(down_at, len(replay)), label="up_at")

    async def run(tmp: Path) -> None:
        port = free_port()
        primary = await start_primary(tmp, port)
        replica = await start_replica(tmp, port)
        client = ShardClient(port)
        reference = SingleNodeReference()
        try:
            await wait_for_recovery(replica, 1)
            for i, op in enumerate(replay):
                if i == down_at:
                    await replica.replica.disconnect()
                if i == up_at:
                    replica.replica.connect()
                await client.write(op)
                reference.apply(op)
            replica.replica.connect()  # no-op if already connected
            await wait_for_seq(replica, len(replay))

            assert replica.store.applied_seq == primary.store.applied_seq == len(replay)
            assert replica.store.index.state() == reference.state()
            assert replica.replica.full_rebuilds == 0
            for q in qs:
                stats = replica.store.index.local_stats(q)
                assert replica.store.index.search(q, stats, 10) == reference.search(q, 10)
        finally:
            await client.close()
            await replica.stop()
            await primary.stop()

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp)))
