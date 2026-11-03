"""Query latency as shards are added (D-033).

    python eval/query_latency.py --backend process|docker [--shards 1,2,3,4] [--docs 50000]

For each shard count: a fresh cluster (1 replica per shard, primary reads),
the same first N corpus docs, and the same query set drawn from the corpus
vocabulary (D-022 generator). Each query runs once as warm-up, then
`--reps` times sequentially from one client; every request's latency is a
CSV row. Because ranking uses global statistics, the top-k must be
identical for every shard count; the script checks that and fails otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

from harness import RAW, ROOT, make_cluster, send, wait_replicated, wait_up

from dsearch.client import SearchClient
from dsearch.convergence import default_queries
from dsearch.index.inverted_index import SearchHit
from dsearch.reference.single_node import SingleNodeReference
from dsearch.workload import insert_ops, read_corpus


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["process", "docker"], required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    parser.add_argument("--shards", type=ints, default=ints("1,2,3,4"))
    parser.add_argument("--docs", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workdir", type=Path, help="default: data/eval/latency-<backend>")
    parser.add_argument("--out", type=Path, help="default: eval/results/raw/latency_<backend>.csv")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> None:
    workdir = args.workdir or ROOT / "data" / "eval" / f"latency-{args.backend}"
    out = args.out or RAW / f"latency_{args.backend}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    load = insert_ops(read_corpus(args.corpus, args.docs))
    reference = SingleNodeReference()
    for op in load:
        reference.apply(op)
    queries = default_queries(reference, count=args.queries, seed=0)
    expected = {q: reference.search(q, args.k) for q in queries}

    rows = []
    for num_shards in args.shards:
        cluster = make_cluster(args.backend, workdir / "cluster", num_shards, 1)
        try:
            cluster.start_all()
            await wait_up(cluster)
            rate = await send(cluster, load, workdir / "ops.jsonl")
            await wait_replicated(cluster)
            print(f"[shards={num_shards}] loaded {len(load)} docs at {rate:.0f} writes/s", flush=True)
            client = SearchClient(cluster.coordinator_address)
            try:
                for query in queries:  # warm-up, and the identical-results check
                    hits = await client.search(query, args.k, timeout=30)
                    if hits != expected[query]:
                        raise AssertionError(f"shards={num_shards}: results for {query!r} differ from the reference")
                for rep in range(args.reps):
                    for i, query in enumerate(queries):
                        started = time.perf_counter()
                        hits: list[SearchHit] = await client.search(query, args.k, timeout=30)
                        latency_ms = (time.perf_counter() - started) * 1000
                        rows.append({"backend": args.backend, "shards": num_shards, "docs": len(load),
                                     "query": i, "terms": len(query.split()), "rep": rep,
                                     "hits": len(hits), "latency_ms": latency_ms})
            finally:
                await client.close()
        finally:
            cluster.stop_all()
        latencies = sorted(r["latency_ms"] for r in rows if r["shards"] == num_shards)
        print(f"[shards={num_shards}] p50 {latencies[len(latencies) // 2]:.2f} ms, "
              f"p99 {latencies[int(len(latencies) * 0.99)]:.2f} ms ({len(latencies)} searches)", flush=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str]) -> int:
    started = time.monotonic()
    asyncio.run(run(parse_args(argv)))
    print(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
