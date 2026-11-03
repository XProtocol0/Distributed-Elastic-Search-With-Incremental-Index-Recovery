"""Run search queries through the coordinator and print hits and latency.

    python scripts/run_queries.py "solar system" "river" [--k 10] [--repeat 5]
    python scripts/run_queries.py --queries queries.txt --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import grpc

from dsearch.client import SearchClient


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query", nargs="*")
    parser.add_argument("--coordinator", default=os.environ.get("DSEARCH_COORDINATOR", "127.0.0.1:50050"))
    parser.add_argument("--queries", type=Path, help="one query per line")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="run each query this many times")
    parser.add_argument("--quiet", action="store_true", help="print only the latency summary")
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    queries = list(args.query)
    if args.queries is not None:
        queries += [q.strip() for q in args.queries.read_text().splitlines() if q.strip()]
    if not queries:
        print("no queries given", file=sys.stderr)
        return 2

    client = SearchClient(args.coordinator)
    latencies_ms: list[float] = []
    try:
        for query in queries:
            for _ in range(args.repeat):
                started = time.perf_counter()
                hits = await client.search(query, args.k, timeout=30.0)
                latencies_ms.append((time.perf_counter() - started) * 1000)
            if not args.quiet:
                print(f"{query!r}: {len(hits)} hits")
                for rank, hit in enumerate(hits, 1):
                    print(f"  {rank:2d}. {hit.score:9.4f}  {hit.doc_id}")
    except grpc.aio.AioRpcError as exc:
        print(f"search failed: {exc.code().name} {exc.details()}", file=sys.stderr)
        return 1
    finally:
        await client.close()

    latencies_ms.sort()
    p95 = latencies_ms[min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))]
    print(
        f"{len(latencies_ms)} searches: p50 {statistics.median(latencies_ms):.2f} ms, "
        f"p95 {p95:.2f} ms, max {latencies_ms[-1]:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
