"""Recovery time: incremental log replay vs full rebuild.

    python eval/recovery_time.py --backend process|docker [--trials 3]

One shard (primary + replica) plus the coordinator. For each index size I a
base index is loaded once and saved as a template. For each number of missed
updates M, the replica stays down while M churn ops (replace/delete/re-insert,
plus stale/duplicate re-sends) go to the primary. The replica is then
restarted from the same saved directory twice per trial: normally
(incremental replay) and with DSEARCH_FORCE_FULL_REBUILD. Every
recovered copy is checked identical to the primary.

Sweep 1: M varies at a fixed index size. Sweep 2: index size varies at fixed M.
Writes one CSV row per (I, M, trial, mode) to eval/results/raw/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import sys
import time
from pathlib import Path

from harness import FORCE_FULL_REBUILD, RAW, ROOT, make_cluster, measure_recovery, send, wait_replicated, wait_up

from dsearch.workload import DocVersions, churn_ops, insert_ops, read_corpus

PRIMARY, REPLICA = "shard0-primary", "shard0-replica1"


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["process", "docker"], required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    parser.add_argument("--base-size", type=int, default=20_000, help="index size for the missed-updates sweep")
    parser.add_argument("--missed", type=ints, default=ints("100,500,1000,2000,5000,10000,20000"))
    parser.add_argument("--sizes", type=ints, default=ints("5000,20000,50000,100000"))
    parser.add_argument("--fixed-missed", type=int, default=1000, help="missed updates for the index-size sweep")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--snapshot-every", type=int, default=10_000)
    parser.add_argument("--workdir", type=Path, help="default: data/eval/recovery-<backend>")
    parser.add_argument("--out", type=Path, help="default: eval/results/raw/recovery_<backend>.csv")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> None:
    workdir = args.workdir or ROOT / "data" / "eval" / f"recovery-{args.backend}"
    out = args.out or RAW / f"recovery_{args.backend}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus = read_corpus(args.corpus)
    texts = [d.text for d in corpus]
    points = {size: set() for size in sorted(set(args.sizes) | {args.base_size})}
    points[args.base_size] |= set(args.missed)
    for size in args.sizes:
        points[size].add(args.fixed_missed)

    rows = []
    for size, missed_values in points.items():
        if size > len(corpus):
            raise SystemExit(f"index size {size} > corpus size {len(corpus)}")
        cluster = make_cluster(args.backend, workdir / "cluster", 1, 1, {"snapshot_every": args.snapshot_every})
        load = insert_ops(corpus[:size])
        print(f"[I={size}] loading", flush=True)
        cluster.start_all()
        await wait_up(cluster)
        rate = await send(cluster, load, workdir / "load-ops.jsonl")
        await wait_replicated(cluster)
        cluster.stop_all()
        template = workdir / f"template-{size}"
        cluster.copy_dir(cluster.data_dir, template)
        print(f"[I={size}] loaded at {rate:.0f} writes/s", flush=True)

        for missed in sorted(missed_values):
            cluster.copy_dir(template, cluster.data_dir)
            cluster.start(PRIMARY)
            cluster.start("coordinator")
            await wait_up(cluster, [PRIMARY, "coordinator"])
            churn = churn_ops(DocVersions.from_ops(load), texts, missed, random.Random(missed))
            await send(cluster, churn, workdir / "churn-ops.jsonl")
            for trial in range(args.trials):
                # Alternate which mode goes first so neither always gets warm caches.
                modes = ["incremental", "full"] if trial % 2 == 0 else ["full", "incremental"]
                for mode in modes:
                    cluster.copy_dir(template / REPLICA, cluster.node_dir(REPLICA))
                    env = FORCE_FULL_REBUILD if mode == "full" else None
                    result = await measure_recovery(cluster, REPLICA, PRIMARY, env, timeout=120)
                    cluster.kill(REPLICA)
                    if result["full_rebuild"] != (mode == "full") or (mode == "incremental" and result["entries_replayed"] != missed):
                        raise AssertionError(f"unexpected recovery for {mode}: {result}")
                    rows.append({"backend": args.backend, "index_docs": size, "missed": missed,
                                 "trial": trial, "mode": mode, **result})
                    print(f"[I={size} M={missed} t={trial}] {mode:<11} wall {result['wall_s']:.3f}s "
                          f"(open {result['open_s']:.3f}s, catch-up {result['catchup_s']:.3f}s)", flush=True)
            cluster.stop_all()
            write_csv(out, rows)  # after every point, so a long run keeps partial results


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    asyncio.run(run(args))
    print(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
