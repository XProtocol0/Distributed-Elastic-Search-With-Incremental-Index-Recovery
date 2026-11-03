"""The core convergence scenario on the real Docker Compose cluster (D-028).

Excluded from a plain `pytest` run; run with `pytest -m docker` (or
`make test-docker`). Skipped when Docker or data/corpus.jsonl is missing.
Uses its own Compose project and coordinator port, so a cluster you are
running by hand is left alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "dsearch-test"

pytestmark = pytest.mark.docker


def docker_running() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture
def compose_project():
    if not docker_running():
        pytest.skip("Docker is not installed or not running")
    if not (ROOT / "data" / "corpus.jsonl").exists():
        pytest.skip("data/corpus.jsonl missing: run scripts/prepare_corpus.py")
    env = {**os.environ, "DSEARCH_COORDINATOR_PORT": "50060"}
    yield env
    subprocess.run(
        ["docker", "compose", "-p", PROJECT, "down", "-v", "--remove-orphans"], cwd=ROOT, env=env, check=False
    )
    (ROOT / "data" / "run" / f"{PROJECT}-ops.jsonl").unlink(missing_ok=True)


def test_partition_and_crash_during_churn_converge(compose_project: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/fault_scenario.py", "--project", PROJECT, "--fresh",
         "--docs", "5000", "--churn", "8000", "--rate", "300"],
        cwd=ROOT, env=compose_project, timeout=900,
    )
    assert result.returncode == 0
