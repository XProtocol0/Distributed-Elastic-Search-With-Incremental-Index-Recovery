"""Docker commands behind each injected fault. Runs without Docker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import inject_fault  # noqa: E402


@pytest.fixture(autouse=True)
def fake_container_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inject_fault, "container_id", lambda project, service: f"{project}-{service}-id")


def tail(command: list[str]) -> list[str]:
    """The command after `docker compose -f <file> -p <project>`."""
    return command[6:] if command[:2] == ["docker", "compose"] else command


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("kill", ["kill", "-s", "SIGKILL", "shard1-replica2"]),
        ("start", ["start", "shard1-replica2"]),
        ("freeze", ["pause", "shard1-replica2"]),
        ("unfreeze", ["unpause", "shard1-replica2"]),
    ],
)
def test_process_faults_use_compose(action: str, expected: list[str]) -> None:
    command = inject_fault.fault_command(action, "shard1-replica2", "dsearch-test")
    assert command[:2] == ["docker", "compose"] and command[4:6] == ["-p", "dsearch-test"]
    assert tail(command) == expected


def test_partition_disconnects_from_the_project_network() -> None:
    assert inject_fault.fault_command("disconnect", "shard0-replica1", "dsearch") == [
        "docker", "network", "disconnect", "dsearch_cluster", "dsearch-shard0-replica1-id",
    ]


def test_heal_restores_the_service_alias() -> None:
    """Without --alias the other nodes can't resolve the replica's hostname again."""
    assert inject_fault.fault_command("connect", "shard0-replica1", "dsearch") == [
        "docker", "network", "connect", "--alias", "shard0-replica1", "dsearch_cluster",
        "dsearch-shard0-replica1-id",
    ]
