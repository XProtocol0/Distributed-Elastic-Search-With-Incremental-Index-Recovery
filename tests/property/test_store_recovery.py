"""Hypothesis: crash a store at a random point (including mid-append) and reopen.

The recovered state must equal the reference fed exactly the acknowledged
writes, and the persisted applied_seq must equal the last durable seq. With
log truncation on, the log may start after seq 1 but must still
cover everything after the snapshot.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dsearch.reference.single_node import SingleNodeReference
from dsearch.shard.store import LOG_FILE, ShardStore
from dsearch.storage.log_entry import make_entry
from tests.strategies import history


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(history(), st.data(), st.integers(0, 4), st.integers(0, 3), st.booleans())
def test_reopen_after_crash_matches_reference(h, data, snapshot_every, retention, torn_tail) -> None:
    _, replay = h
    retention = retention if snapshot_every else 0
    crash_at = data.draw(st.integers(0, len(replay)), label="crash_at")
    acked = replay[:crash_at]

    async def run(path: Path) -> None:
        store = await ShardStore.open(path, snapshot_every, retention)
        for seq, op in enumerate(acked, start=1):
            await store.write(make_entry(seq, op))
        await store.close()
        if torn_tail:  # an unacknowledged next write was half-written
            with open(path / LOG_FILE, "ab") as f:
                f.write(b"\x00\x00\x00\x30\x12")

        recovered = await ShardStore.open(path, snapshot_every, retention)
        reference = SingleNodeReference()
        for op in acked:
            reference.apply(op)
        assert recovered.index.state() == reference.state()
        assert recovered.applied_seq == len(acked)
        assert recovered.db.applied_seq_sync() == len(acked)
        assert recovered.update_log.first_seq - 1 <= recovered.snapshot_seq
        if retention and len(acked) >= snapshot_every + retention + 1:
            assert recovered.update_log.first_seq > 1  # something was truncated
        await recovered.close()

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp)))
