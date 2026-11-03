"""Hypothesis: random histories (duplicates and reorders included) written through
the coordinator to 1-3 shards; every search equals the reference exactly."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dsearch.client import SearchClient
from dsearch.local_cluster import LocalCluster
from dsearch.reference.single_node import SingleNodeReference
from tests.strategies import history, queries


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(history(), st.integers(1, 3), st.lists(queries, min_size=1, max_size=4), st.integers(1, 12))
def test_cluster_search_equals_reference(h, num_shards, qs, k) -> None:
    _, replay = h

    async def run(tmp: Path) -> None:
        cluster = await LocalCluster.start(tmp, num_shards=num_shards, replicas_per_shard=0)
        client = SearchClient(cluster.coordinator_address)
        reference = SingleNodeReference()
        try:
            for op in replay:
                await client.write(op)
                reference.apply(op)
            for q in qs:
                assert await client.search(q, k) == reference.search(q, k)
        finally:
            await client.close()
            await cluster.stop()

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp)))
