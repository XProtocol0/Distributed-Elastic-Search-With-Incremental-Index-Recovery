"""Inject faults into the Docker Compose cluster.

    python scripts/inject_fault.py kill shard1-replica2        # crash (SIGKILL); data volume survives
    python scripts/inject_fault.py start shard1-replica2       # restart after a crash
    python scripts/inject_fault.py disconnect shard0-replica1  # network partition (no TCP reset)
    python scripts/inject_fault.py connect shard0-replica1     # heal the partition
    python scripts/inject_fault.py freeze|unfreeze <service>   # SIGSTOP/SIGCONT the whole container

Containers use `restart: "no"`, so a fault stays until it is undone here.
--project must match the Compose project name (default "dsearch").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETWORK = "cluster"  # network name inside docker-compose.yml
ACTIONS = ("kill", "start", "disconnect", "connect", "freeze", "unfreeze")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("service", help="Compose service, e.g. shard0-replica1")
    parser.add_argument("--project", default="dsearch")
    return parser.parse_args(argv)


def compose(project: str, *args: str) -> list[str]:
    return ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "-p", project, *args]


def container_id(project: str, service: str) -> str:
    out = subprocess.run(
        compose(project, "ps", "--all", "-q", service), check=True, capture_output=True, text=True
    ).stdout.strip()
    if not out:
        raise SystemExit(f"no container for service {service!r} in project {project!r}")
    return out.splitlines()[0]


def fault_command(action: str, service: str, project: str) -> list[str]:
    """The docker command that performs `action` on `service`."""
    if action == "kill":
        return compose(project, "kill", "-s", "SIGKILL", service)
    if action == "start":
        return compose(project, "start", service)
    if action == "freeze":
        return compose(project, "pause", service)
    if action == "unfreeze":
        return compose(project, "unpause", service)
    network = f"{project}_{NETWORK}"
    container = container_id(project, service)
    if action == "disconnect":
        return ["docker", "network", "disconnect", network, container]
    # Reconnecting must restore the service-name alias, or the other nodes
    # can no longer resolve e.g. "shard0-replica1".
    return ["docker", "network", "connect", "--alias", service, network, container]


def inject(action: str, service: str, project: str = "dsearch") -> None:
    command = fault_command(action, service, project)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        inject(args.action, args.service, args.project)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
