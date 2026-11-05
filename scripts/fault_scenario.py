"""The core convergence scenario on the Docker Compose cluster.

    python scripts/fault_scenario.py [--fresh] [--docs 20000] [--churn 15000] [--rate 400]

1. (--fresh) wipe and start the cluster; load `--docs` corpus docs.
2. Run replace/delete churn at `--rate` ops/s in the background.
3. A quarter of the way in: partition shard0-replica1 from the network and
   crash shard1-replica2 (SIGKILL).
4. At about two thirds: heal the partition and restart the crashed replica.
5. When the churn ends: wait for every replica to catch up, check that both
   faulted replicas recovered by incremental replay (no full rebuild), and
   run the convergence checker against the single-node reference.

Exit code 0 only if every step passed and the checker found zero differences.
Run from the repo root on a machine with Docker; needs data/corpus.jsonl
(scripts/prepare_corpus.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from inject_fault import compose, inject

PARTITIONED = "shard0-replica1"
CRASHED = "shard1-replica2"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", default="dsearch")
    parser.add_argument("--fresh", action="store_true", help="docker compose down -v, then up --build")
    parser.add_argument("--docs", type=int, default=20_000)
    parser.add_argument("--churn", type=int, default=15_000)
    parser.add_argument("--rate", type=float, default=400.0, help="churn ops per second")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def step(message: str) -> None:
    print(f"\n=== {message}", flush=True)


def tools(project: str, *script_args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a script inside the `tools` container, on the cluster network."""
    command = compose(project, "run", "--rm", "-T", "tools", "python", *script_args)
    return subprocess.run(command, text=True, capture_output=capture)


def check_recovery(statuses: dict[str, dict]) -> list[str]:
    """Problems with how the two faulted replicas recovered (empty = fine)."""
    problems = []
    for address, status in statuses.items():
        if int(status["full_rebuilds"]):
            problems.append(f"{address}: {status['full_rebuilds']} full rebuild(s)")
    partitioned = statuses[f"{PARTITIONED}:50051"]
    # The partition breaks the first stream (keepalive); healing starts a second one.
    sessions = partitioned["recoveries"]
    if len(sessions) < 2 or not any(int(s["entries_replayed"]) > 0 for s in sessions[1:]):
        problems.append(f"{PARTITIONED}: expected a catch-up session that replayed missed entries, got {sessions}")
    crashed = statuses[f"{CRASHED}:50051"]
    first = crashed["recoveries"][0] if crashed["recoveries"] else None
    if first is None or int(first["from_seq"]) == 0:
        problems.append(f"{CRASHED}: restart should resume from its persisted applied_seq, got {first}")
    for name, status in ((PARTITIONED, partitioned), (CRASHED, crashed)):
        for s in status["recoveries"]:
            if int(s["entries_replayed"]) != int(s["caught_up_seq"]) - int(s["from_seq"]):
                problems.append(f"{name}: session {s} replayed more or less than it missed")
    return problems


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    project = args.project
    ops = Path("data/run") / f"{project}-ops.jsonl"

    if args.fresh:
        step("fresh cluster")
        subprocess.run(compose(project, "down", "-v", "--remove-orphans"), check=True)
        subprocess.run(compose(project, "up", "-d", "--build"), check=True)
        ops.unlink(missing_ok=True)

    step(f"load {args.docs} docs")
    if tools(project, "scripts/load_data.py", "--ops", str(ops), "--limit", str(args.docs)).returncode:
        return 1

    duration = args.churn / args.rate
    step(f"churn: {args.churn} ops at {args.rate:.0f}/s (~{duration:.0f}s) with faults")
    churn = subprocess.Popen(
        compose(project, "run", "--rm", "-T", "tools", "python", "scripts/run_workload.py",
                "--ops", str(ops), "--count", str(args.churn), "--rate", str(args.rate),
                "--seed", str(args.seed))
    )
    started = time.monotonic()
    time.sleep(duration * 0.25)
    inject("disconnect", PARTITIONED, project)
    inject("kill", CRASHED, project)
    time.sleep(max(0.0, started + duration * 0.65 - time.monotonic()))
    inject("connect", PARTITIONED, project)
    inject("start", CRASHED, project)
    if churn.wait() != 0:
        print("churn failed", file=sys.stderr)
        return 1

    step("wait for replicas to catch up")
    status = tools(project, "scripts/cluster_status.py", "--wait-replicas", "120", "--json", capture=True)
    if status.returncode:
        print(status.stdout, status.stderr, file=sys.stderr)
        return 1
    tools(project, "scripts/cluster_status.py")
    problems = check_recovery(json.loads(status.stdout))
    for problem in problems:
        print(f"RECOVERY PROBLEM: {problem}", file=sys.stderr)

    step("convergence check against the single-node reference")
    converged = tools(project, "scripts/check_convergence.py", "--ops", str(ops)).returncode == 0
    ok = converged and not problems
    print(f"\nscenario {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
