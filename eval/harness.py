"""Shared evaluation harness (D-033).

Two cluster backends with the same interface: real processes
(`dsearch.process_cluster.ProcessCluster`) and Docker Compose (`DockerCluster`,
a compose file generated per topology). Every node's data dir is a host path in
both, so experiments can save and restore byte-identical state between trials.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import grpc

from dsearch.client import SearchClient
from dsearch.cluster_status import get_status, wait_for_replicas, wait_until_up
from dsearch.config import ShardTopology
from dsearch.index.operation import Operation
from dsearch.local_cluster import free_port, node_id
from dsearch.process_cluster import ProcessCluster, node_roles, topology_toml
from dsearch.proto import search_pb2, search_pb2_grpc
from dsearch.workload import send_ops

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "eval" / "results" / "raw"
CONTAINER_PORT = 50051
FORCE_FULL_REBUILD = {"DSEARCH_FORCE_FULL_REBUILD": "1"}


class Cluster(Protocol):
    data_dir: Path
    roles: dict[str, tuple[str, int | None]]

    def address(self, name: str) -> str: ...
    @property
    def coordinator_address(self) -> str: ...
    @property
    def shards(self) -> tuple[ShardTopology, ...]: ...
    def node_dir(self, name: str) -> Path: ...
    def copy_dir(self, src: Path, dest: Path) -> None: ...
    def start(self, name: str, env: Mapping[str, str] | None = None) -> None: ...
    def start_all(self) -> None: ...
    def kill(self, name: str) -> None: ...
    def stop_all(self) -> object: ...


class DockerCluster:
    """The cluster as Compose services from the `dsearch:latest` image (build it
    first: `docker compose build`). Each node's port is published on 127.0.0.1
    so the harness runs on the host; nodes reach each other by service name.
    `base` must be in a directory the Docker VM shares (colima and Docker
    Desktop share $HOME by default, not /tmp), or the bind mounts come up empty."""

    def __init__(
        self,
        base: Path,
        num_shards: int,
        replicas_per_shard: int,
        settings: Mapping[str, int | float | str] | None = None,
        project: str = "dsearch-eval",
    ) -> None:
        self.base = base
        self.project = project
        self.data_dir = base / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.roles = node_roles(num_shards, replicas_per_shard)
        self.ports = {name: free_port() for name in self.roles}
        in_cluster = {name: f"{name}:{CONTAINER_PORT}" for name in self.roles}
        (base / "config").mkdir()
        (base / "config" / "cluster.toml").write_text(
            topology_toml(num_shards, replicas_per_shard, in_cluster, "/data", settings)
        )
        host = {name: self.address(name) for name in self.roles}
        self._host_config = base / "host-cluster.toml"
        self._host_config.write_text(topology_toml(num_shards, replicas_per_shard, host, "unused", settings))
        self.compose_file = base / "docker-compose.yml"
        self.compose_file.write_text(self._compose_yaml())

    def _compose_yaml(self) -> str:
        lines = ["services:"]
        for name, (role, shard_id) in self.roles.items():
            module = "dsearch.coordinator" if role == "coordinator" else "dsearch.shard"
            env = {
                "DSEARCH_CONFIG": "/eval/cluster.toml",
                "DSEARCH_NODE_ID": name,
                "DSEARCH_ROLE": role,
                "DSEARCH_HOST": "0.0.0.0",
                "DSEARCH_PORT": str(CONTAINER_PORT),
                "DSEARCH_FORCE_FULL_REBUILD": "${DSEARCH_FORCE_FULL_REBUILD:-0}",  # per start (D-032)
            }
            if shard_id is not None:
                env["DSEARCH_SHARD_ID"] = str(shard_id)
            lines += [
                f"  {name}:",
                "    image: dsearch:latest",
                "    pull_policy: never",
                '    restart: "no"',
                f"    hostname: {name}",
                f'    command: ["python", "-m", "{module}"]',
                f'    ports: ["127.0.0.1:{self.ports[name]}:{CONTAINER_PORT}"]',
                f'    volumes: ["{self.data_dir}:/data", "{self.base / "config"}:/eval:ro"]',
                "    environment:",
                *(f'      {key}: "{value}"' for key, value in env.items()),
            ]
        return "\n".join(lines) + "\n"

    def _compose(self, *args: str, env: Mapping[str, str] | None = None) -> None:
        result = subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "-p", self.project, *args],
            capture_output=True, text=True, env={**os.environ, **(env or {})},
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker compose {' '.join(args)} failed:\n{result.stderr}")

    def address(self, name: str) -> str:
        return f"127.0.0.1:{self.ports[name]}"

    @property
    def coordinator_address(self) -> str:
        return self.address("coordinator")

    @property
    def shards(self) -> tuple[ShardTopology, ...]:
        from dsearch.config import load_config

        return load_config({"DSEARCH_CONFIG": str(self._host_config)}).shards

    def node_dir(self, name: str) -> Path:
        return self.data_dir / name

    def copy_dir(self, src: Path, dest: Path) -> None:
        """Like `copy_dir`, but done inside the Docker VM. Changing bind-mounted
        files from the macOS host leaves colima's virtiofs view briefly stale
        (a restarted node then fails to open its SQLite file); copying through
        the VM keeps its view coherent."""
        root = Path(os.path.commonpath([src.resolve(), dest.resolve(), self.base]))
        rel_src, rel_dest = src.resolve().relative_to(root), dest.resolve().relative_to(root)
        script = f'mkdir -p "/w/{rel_dest}" && find "/w/{rel_dest}" -mindepth 1 -delete && cp -a "/w/{rel_src}/." "/w/{rel_dest}/"'
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{root}:/w", "dsearch:latest", "sh", "-c", script],
            check=True, capture_output=True,
        )

    def start(self, name: str, env: Mapping[str, str] | None = None) -> None:
        """(Re)create and start one node's container. `env` may only set the
        forced-rebuild flag; anything else is fixed in the compose file."""
        if env and set(env) - set(FORCE_FULL_REBUILD):
            raise ValueError(f"DockerCluster.start only supports {list(FORCE_FULL_REBUILD)}, got {list(env)}")
        self._compose("rm", "-sf", name)
        self._compose("up", "-d", "--no-deps", name, env=env)

    def start_all(self) -> None:
        self._compose("up", "-d")

    def kill(self, name: str) -> None:
        self._compose("kill", "-s", "SIGKILL", name)

    def stop_all(self) -> None:
        self._compose("down", "--remove-orphans")


_image_built = False


def build_image() -> None:
    """Build dsearch:latest from the working tree (cached layers make repeats
    cheap), so a Docker run never measures stale code."""
    global _image_built
    if not _image_built:
        subprocess.run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "build", "-q"], check=True)
        _image_built = True


def make_cluster(
    backend: str, base: Path, num_shards: int, replicas_per_shard: int,
    settings: Mapping[str, int | float | str] | None = None,
) -> Cluster:
    # A fresh, never-reused directory per cluster: on colima/virtiofs a bind mount of
    # a path that was just deleted and recreated can briefly show the old (missing) file.
    root = base.resolve()  # compose bind mounts need absolute paths
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("cluster-*"):
        shutil.rmtree(old)
    base = Path(tempfile.mkdtemp(prefix="cluster-", dir=root))
    if backend == "process":
        return ProcessCluster(base, num_shards, replicas_per_shard, settings)
    if backend == "docker":
        build_image()
        return DockerCluster(base, num_shards, replicas_per_shard, settings)
    raise ValueError(f"unknown backend {backend!r}")


# ---- helpers used by the experiments ---------------------------------------


async def wait_up(cluster: Cluster, names: list[str] | None = None, timeout: float = 60.0) -> None:
    await wait_until_up([cluster.address(n) for n in (names or list(cluster.roles))], timeout)


async def send(cluster: Cluster, ops: list[Operation], ops_path: Path, concurrency: int = 32) -> float:
    """Send `ops` through the coordinator; returns writes per second."""
    client = SearchClient(cluster.coordinator_address)
    try:
        stats = await send_ops(client, ops, ops_path, concurrency=concurrency)
    finally:
        await client.close()
    return stats.sent / stats.seconds if stats.seconds else 0.0


async def wait_replicated(cluster: Cluster, timeout: float = 300.0) -> None:
    await wait_for_replicas(cluster.shards, timeout)


async def dump(address: str) -> search_pb2.DumpStateReply:
    async with grpc.aio.insecure_channel(address, options=[("grpc.max_receive_message_length", 1 << 30)]) as channel:
        return await search_pb2_grpc.ShardStub(channel).DumpState(search_pb2.DumpStateRequest(), timeout=120)


async def measure_recovery(
    cluster: Cluster, replica: str, primary: str, env: Mapping[str, str] | None = None, timeout: float = 600.0
) -> dict[str, float | int | bool]:
    """Start `replica` and time it until it is live with the primary's applied_seq
    (polled every 20 ms). The primary must not be taking writes. Also verifies
    the recovered copy is identical to the primary's."""
    target = (await get_status(cluster.address(primary))).applied_seq
    started = time.monotonic()
    cluster.start(replica, env)
    while True:
        try:
            status = await get_status(cluster.address(replica), timeout=1.0)
            if status.live and status.applied_seq == target:
                break
        except grpc.aio.AioRpcError:
            pass  # still starting
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"{replica} did not recover within {timeout}s")
        await asyncio.sleep(0.02)
    wall = time.monotonic() - started

    session = status.recoveries[-1]
    replica_state, primary_state = await dump(cluster.address(replica)), await dump(cluster.address(primary))
    if replica_state != primary_state:
        raise AssertionError(f"{replica} recovered to a state different from {primary}")
    return {
        "wall_s": wall,
        "open_s": status.open_seconds,
        "catchup_s": session.seconds,
        "full_rebuild": session.full_rebuild,
        "entries_replayed": session.entries_replayed,
        "snapshot_docs": session.snapshot_docs,
        "local_replayed": status.local_replayed,
        "applied_seq": status.applied_seq,
        "full_rebuilds": status.full_rebuilds,
    }


def primary_of(shard_id: int) -> str:
    return node_id(shard_id, 0)


def replica_of(shard_id: int, copy: int = 1) -> str:
    return node_id(shard_id, copy)
