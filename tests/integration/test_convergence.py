"""The core convergence check (Goal 6) on a 3 x (1 + 2) cluster, plus proof that
the checker catches lagging, tampered and unreachable copies."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from dsearch.client import SearchClient
from dsearch.config import ShardTopology
from dsearch.convergence import DiffKind, check_convergence
from dsearch.coordinator.router import shard_for
from dsearch.index.operation import Operation, OpType
from dsearch.local_cluster import LocalCluster
from dsearch.ops_file import append_ops
from dsearch.reference.single_node import SingleNodeReference

ROOT = Path(__file__).resolve().parents[2]
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()


def initial_ops(n: int = 60) -> list[Operation]:
    return [
        Operation(OpType.INSERT, f"doc{i}", 1, " ".join(WORDS[(i + j * j) % len(WORDS)] for j in range(3 + i % 5)))
        for i in range(n)
    ]


def churn_ops(n: int = 60) -> list[Operation]:
    """Replaces, deletes, stale re-inserts, duplicates and out-of-order versions."""
    ops: list[Operation] = []
    for i in range(0, n, 2):
        ops.append(Operation(OpType.REPLACE, f"doc{i}", 3, f"{WORDS[i % 12]} {WORDS[(i * 5) % 12]} replaced"))
        ops.append(Operation(OpType.REPLACE, f"doc{i}", 2, "older replace arrives late"))  # stale
    for i in range(1, n, 3):
        ops.append(Operation(OpType.DELETE, f"doc{i}", 4))
        ops.append(Operation(OpType.INSERT, f"doc{i}", 2, "zombie re-insert"))  # blocked by tombstone
        ops.append(Operation(OpType.DELETE, f"doc{i}", 4))  # duplicate delete
    ops += [Operation(OpType.INSERT, f"new{i}", 1, f"{WORDS[i % 12]} fresh") for i in range(10)]
    return ops


async def run(client: SearchClient, reference: SingleNodeReference, ops: list[Operation]) -> None:
    for op in ops:
        await client.write(op)
        reference.apply(op)


def count_for_shard(ops: list[Operation], shard_id: int, num_shards: int = 3) -> int:
    return sum(1 for op in ops if shard_for(op.doc_id, num_shards) == shard_id)


@pytest.fixture
async def cluster(tmp_path: Path):
    cluster = await LocalCluster.start(tmp_path, num_shards=3, replicas_per_shard=2)
    client = SearchClient(cluster.coordinator_address)
    yield cluster, client
    await client.close()
    await cluster.stop()


async def test_replicas_converge_after_disconnect_and_restart_during_churn(cluster) -> None:
    """Disconnect one replica and crash-restart another while replaces and
    deletes continue; afterwards every copy matches the reference with zero
    differences, and only the missed entries were replayed."""
    cluster, client = cluster
    reference = SingleNodeReference()
    first, churn = initial_ops(), churn_ops()
    await run(client, reference, first)
    await cluster.wait_replicated()

    disconnected = cluster.node(1, 1).replica
    sessions = len(disconnected.recoveries)
    await disconnected.disconnect()
    await cluster.stop_node(2, 2)  # crash
    await run(client, reference, churn)

    disconnected.connect()
    restarted = await cluster.start_node(2, 2)
    reconnect = await asyncio.wait_for(disconnected.wait_for_recovery(sessions + 1), 10)
    restart = await asyncio.wait_for(restarted.replica.wait_for_recovery(1), 10)
    await cluster.wait_replicated()

    report = await check_convergence(cluster.base.shards, reference)
    assert report.converged, report.summary()
    assert report.copies_checked == 9

    assert reconnect.entries_replayed == count_for_shard(churn, 1)  # incremental (Invariant 8)
    assert restarted.store.open_report.replayed == count_for_shard(first, 2)  # from its own log
    assert restart.entries_replayed == count_for_shard(churn, 2)
    for shard_id in range(3):
        for copy in (1, 2):
            assert cluster.node(shard_id, copy).replica.full_rebuilds == 0


async def test_checker_reports_a_lagging_replica(cluster) -> None:
    cluster, client = cluster
    reference = SingleNodeReference()
    await run(client, reference, initial_ops())
    await cluster.wait_replicated()
    await cluster.node(0, 2).replica.disconnect()
    await run(client, reference, churn_ops())
    await cluster.wait_replicated()  # the disconnected replica stays behind

    report = await check_convergence(cluster.base.shards, reference)
    assert not report.converged
    lagging = cluster.base.shards[0].replicas[1]
    assert {d.copy for d in report.differences} == {f"shard0 {lagging}"}
    kinds = {d.kind for d in report.differences}
    assert {DiffKind.APPLIED_SEQ, DiffKind.SEARCH} <= kinds
    assert kinds & {DiffKind.VERSION, DiffKind.MISSING_DOC, DiffKind.TOMBSTONE}


async def test_checker_reports_a_tampered_replica(cluster) -> None:
    cluster, client = cluster
    reference = SingleNodeReference()
    await run(client, reference, initial_ops())
    await cluster.wait_replicated()
    victim = next(f"doc{i}" for i in range(60) if shard_for(f"doc{i}", 3) == 2)
    # Bypass the log: the replica's index now holds something the reference doesn't.
    cluster.node(2, 1).store.index.apply(Operation(OpType.REPLACE, victim, 9, "alpha alpha alpha"))

    report = await check_convergence(cluster.base.shards, reference)
    tampered = f"shard2 {cluster.base.shards[2].replicas[0]}"
    assert {d.copy for d in report.differences} == {tampered}
    assert DiffKind.VERSION in {d.kind for d in report.differences}
    assert DiffKind.SEARCH in {d.kind for d in report.differences}  # "alpha" ranking changed


async def test_checker_reports_an_unreachable_copy(cluster) -> None:
    cluster, client = cluster
    reference = SingleNodeReference()
    await run(client, reference, initial_ops(10))
    await cluster.wait_replicated()
    await cluster.stop_node(1, 2)
    report = await check_convergence(cluster.base.shards, reference, rpc_timeout=1.0)
    assert [d.kind for d in report.differences] == [DiffKind.UNREACHABLE]


def topology_toml(shards: tuple[ShardTopology, ...]) -> str:
    lines = [f"num_shards = {len(shards)}", f"replicas_per_shard = {len(shards[0].replicas)}"]
    for s in shards:
        replicas = ", ".join(f'"{r}"' for r in s.replicas)
        lines += ["[[shards]]", f"id = {s.shard_id}", f'primary = "{s.primary}"', f"replicas = [{replicas}]"]
    return "\n".join(lines) + "\n"


async def run_script(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(ROOT / "scripts" / "check_convergence.py"), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode()


async def test_check_convergence_script(cluster, tmp_path: Path) -> None:
    cluster, client = cluster
    reference = SingleNodeReference()
    ops = initial_ops(30) + churn_ops(30)
    await run(client, reference, ops)
    await cluster.wait_replicated()
    config, ops_file = tmp_path / "cluster.toml", tmp_path / "ops.jsonl"
    config.write_text(topology_toml(cluster.base.shards))
    append_ops(ops_file, ops)

    code, out = await run_script("--config", str(config), "--ops", str(ops_file))
    assert code == 0, out
    assert "0 differences" in out

    append_ops(ops_file, [Operation(OpType.INSERT, "never-sent", 1, "alpha")])  # reference only
    code, out = await run_script("--config", str(config), "--ops", str(ops_file))
    assert code == 1
    assert "missing_doc" in out and "never-sent" in out
