"""A cluster of real node processes on localhost.

Each node runs the same entry point as in a container (`python -m
dsearch.shard` / `dsearch.coordinator`), configured by a generated TOML
topology plus DSEARCH_* env vars. Used by tests that need real
process crashes (SIGKILL) without Docker, and by the evaluation.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from dsearch.config import ShardTopology, load_config
from dsearch.local_cluster import free_port, node_id


def topology_toml(
    num_shards: int,
    replicas_per_shard: int,
    address: Mapping[str, str],
    data_dir: str,
    settings: Mapping[str, int | float | str] | None = None,
) -> str:
    """TOML for a cluster whose node `name` is reachable at `address[name]`."""
    lines = [
        f"num_shards = {num_shards}",
        f"replicas_per_shard = {replicas_per_shard}",
        f'data_dir = "{data_dir}"',
    ]
    for key, value in (settings or {}).items():
        lines.append(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}")
    for shard_id in range(num_shards):
        names = [node_id(shard_id, copy) for copy in range(replicas_per_shard + 1)]
        replicas = ", ".join(f'"{address[n]}"' for n in names[1:])
        lines += ["", "[[shards]]", f"id = {shard_id}", f'primary = "{address[names[0]]}"', f"replicas = [{replicas}]"]
    return "\n".join(lines) + "\n"


def node_roles(num_shards: int, replicas_per_shard: int) -> dict[str, tuple[str, int | None]]:
    """Node name -> (role, shard id), coordinator first."""
    roles: dict[str, tuple[str, int | None]] = {"coordinator": ("coordinator", None)}
    for shard_id in range(num_shards):
        for copy in range(replicas_per_shard + 1):
            roles[node_id(shard_id, copy)] = ("primary" if copy == 0 else "replica", shard_id)
    return roles


class ProcessCluster:
    def __init__(
        self,
        base: Path,
        num_shards: int,
        replicas_per_shard: int,
        settings: Mapping[str, int | float | str] | None = None,
    ) -> None:
        """`settings` are extra top-level config keys for every node (e.g. snapshot_every)."""
        self.base = base
        self.config_path = base / "cluster.toml"
        self.data_dir = base / "data"
        self.logs = base / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.roles = node_roles(num_shards, replicas_per_shard)
        self.ports = {name: free_port() for name in self.roles}
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        address = {name: self.address(name) for name in self.roles}
        self.config_path.write_text(
            topology_toml(num_shards, replicas_per_shard, address, str(self.data_dir), settings)
        )

    def address(self, name: str) -> str:
        return f"127.0.0.1:{self.ports[name]}"

    @property
    def coordinator_address(self) -> str:
        return self.address("coordinator")

    @property
    def shards(self) -> tuple[ShardTopology, ...]:
        return load_config({"DSEARCH_CONFIG": str(self.config_path)}).shards

    def node_dir(self, name: str) -> Path:
        """The node's update log and SQLite state (Config.node_data_dir)."""
        return self.data_dir / name

    def copy_dir(self, src: Path, dest: Path) -> None:
        """Make `dest` an exact copy of `src` (e.g. restore a saved data dir)."""
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

    def start(self, name: str, env: Mapping[str, str] | None = None) -> None:
        """Start one node; `env` adds DSEARCH_* overrides for this start only."""
        role, shard_id = self.roles[name]
        full_env = {
            **os.environ,
            "DSEARCH_CONFIG": str(self.config_path),
            "DSEARCH_NODE_ID": name,
            "DSEARCH_ROLE": role,
            "DSEARCH_HOST": "127.0.0.1",
            "DSEARCH_PORT": str(self.ports[name]),
            **(env or {}),
        }
        if shard_id is not None:
            full_env["DSEARCH_SHARD_ID"] = str(shard_id)
        module = "dsearch.coordinator" if role == "coordinator" else "dsearch.shard"
        with open(self.logs / f"{name}.log", "ab") as log:
            self.procs[name] = subprocess.Popen([sys.executable, "-m", module], env=full_env, stderr=log)

    def start_all(self) -> None:
        for name in self.roles:
            self.start(name)

    def kill(self, name: str) -> None:
        """Crash the node: SIGKILL, no shutdown code runs."""
        proc = self.procs.pop(name)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)

    def stop(self, name: str) -> int:
        """Clean shutdown (SIGTERM); returns the exit code."""
        proc = self.procs.pop(name)
        proc.send_signal(signal.SIGTERM)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()

    def stop_all(self) -> dict[str, int]:
        """SIGTERM every node and return their exit codes (0 = clean shutdown)."""
        return {name: self.stop(name) for name in list(self.procs)}
