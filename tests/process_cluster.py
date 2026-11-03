"""A cluster of real node processes on localhost, for tests that need actual
process crashes (SIGKILL) without Docker. Each node runs the same entry
point as in a container (`python -m dsearch.shard` / `dsearch.coordinator`),
configured by a TOML topology plus DSEARCH_* env vars (D-002)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from dsearch.local_cluster import free_port, node_id


class ProcessCluster:
    def __init__(self, base: Path, num_shards: int, replicas_per_shard: int) -> None:
        self.base = base
        self.config_path = base / "cluster.toml"
        self.logs = base / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.roles: dict[str, tuple[str, int | None]] = {"coordinator": ("coordinator", None)}
        self.ports: dict[str, int] = {"coordinator": free_port()}
        self.procs: dict[str, subprocess.Popen[bytes]] = {}

        lines = [
            f"num_shards = {num_shards}",
            f"replicas_per_shard = {replicas_per_shard}",
            f'data_dir = "{base / "data"}"',
            'log_level = "INFO"',
        ]
        for shard_id in range(num_shards):
            names = [node_id(shard_id, copy) for copy in range(replicas_per_shard + 1)]
            for copy, name in enumerate(names):
                self.roles[name] = ("primary" if copy == 0 else "replica", shard_id)
                self.ports[name] = free_port()
            replicas = ", ".join(f'"{self.address(n)}"' for n in names[1:])
            lines += ["", "[[shards]]", f"id = {shard_id}", f'primary = "{self.address(names[0])}"', f"replicas = [{replicas}]"]
        self.config_path.write_text("\n".join(lines) + "\n")

    def address(self, name: str) -> str:
        return f"127.0.0.1:{self.ports[name]}"

    @property
    def coordinator_address(self) -> str:
        return self.address("coordinator")

    def start(self, name: str) -> None:
        role, shard_id = self.roles[name]
        env = {
            **os.environ,
            "DSEARCH_CONFIG": str(self.config_path),
            "DSEARCH_NODE_ID": name,
            "DSEARCH_ROLE": role,
            "DSEARCH_HOST": "127.0.0.1",
            "DSEARCH_PORT": str(self.ports[name]),
        }
        if shard_id is not None:
            env["DSEARCH_SHARD_ID"] = str(shard_id)
        module = "dsearch.coordinator" if role == "coordinator" else "dsearch.shard"
        with open(self.logs / f"{name}.log", "ab") as log:
            self.procs[name] = subprocess.Popen([sys.executable, "-m", module], env=env, stderr=log)

    def start_all(self) -> None:
        for name in self.roles:
            self.start(name)

    def kill(self, name: str) -> None:
        """Crash the node: SIGKILL, no shutdown code runs."""
        proc = self.procs.pop(name)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)

    def stop_all(self) -> dict[str, int]:
        """SIGTERM every node and return their exit codes (0 = clean shutdown)."""
        for proc in self.procs.values():
            proc.send_signal(signal.SIGTERM)
        codes = {}
        for name, proc in self.procs.items():
            try:
                codes[name] = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                codes[name] = proc.wait()
        self.procs.clear()
        return codes
