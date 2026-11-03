"""The cluster scripts against real node processes, with a real crash (no Docker).

Every node is its own OS process. A replica is SIGKILLed while
run_workload.py sends replaces and deletes, then restarted on its data dir.
It must rebuild from its own log, catch up over the network from exactly
where that left off, and converge with the reference (check_convergence.py).
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from dsearch.cluster_status import wait_until_up
from tests.process_cluster import ProcessCluster

ROOT = Path(__file__).resolve().parents[2]
WORDS = "river mountain city music science history planet ocean forest king war language art film".split()


def script(name: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / name), *args]


def write_corpus(path: Path, count: int) -> None:
    rng = random.Random(0)
    with open(path, "w") as f:
        for i in range(count):
            text = " ".join(rng.choices(WORDS, k=rng.randint(5, 25)))
            f.write(json.dumps({"doc_id": f"doc{i}", "title": f"Doc {i}", "text": text}) + "\n")


def test_crash_and_restart_a_replica_during_churn(tmp_path: Path) -> None:
    cluster = ProcessCluster(tmp_path, num_shards=2, replicas_per_shard=2)
    corpus, ops = tmp_path / "corpus.jsonl", tmp_path / "ops.jsonl"
    write_corpus(corpus, 400)
    coordinator = ["--coordinator", cluster.coordinator_address]
    victim = "shard0-replica1"
    cluster.start_all()
    try:
        asyncio.run(wait_until_up([cluster.address(n) for n in cluster.roles], timeout=30))
        load = subprocess.run(script("load_data.py", *coordinator, "--corpus", str(corpus), "--ops", str(ops)))
        assert load.returncode == 0

        churn = subprocess.Popen(
            script("run_workload.py", *coordinator, "--corpus", str(corpus), "--ops", str(ops),
                   "--count", "1500", "--rate", "400")
        )
        time.sleep(1.0)
        cluster.kill(victim)
        time.sleep(1.5)  # writes to shard 0 continue while it is down
        cluster.start(victim)
        assert churn.wait(timeout=60) == 0

        status = subprocess.run(
            script("cluster_status.py", "--config", str(cluster.config_path), "--wait-replicas", "30", "--json"),
            capture_output=True, text=True,
        )
        assert status.returncode == 0, status.stderr
        restarted = json.loads(status.stdout)[cluster.address(victim)]
        session = restarted["recoveries"][0]
        from_seq, caught_up, replayed = (int(session[k]) for k in ("from_seq", "caught_up_seq", "entries_replayed"))
        # Local replay of its own log first, then the network catch-up starts exactly there.
        assert int(restarted["local_replayed"]) == from_seq > 0
        assert replayed == caught_up - from_seq > 0
        assert int(restarted["full_rebuilds"]) == 0

        check = subprocess.run(
            script("check_convergence.py", "--config", str(cluster.config_path), "--ops", str(ops)),
            capture_output=True, text=True,
        )
        assert check.returncode == 0, check.stdout + check.stderr
        assert "0 differences" in check.stdout

        queries = subprocess.run(script("run_queries.py", *coordinator, "river music", "--k", "3"),
                                 capture_output=True, text=True)
        assert queries.returncode == 0 and "3 hits" in queries.stdout
    finally:
        exit_codes = cluster.stop_all()
    assert set(exit_codes.values()) == {0}  # SIGTERM shuts every node down cleanly
