"""Send replace/delete churn to a loaded cluster (run this while injecting faults).

    python scripts/run_workload.py --count 20000 [--rate 500] [--seed 1]

Versions continue from the ops file written by load_data.py (and earlier
workload runs); the new ops are appended to it. The mix of replaces,
deletes, re-inserts and deliberately stale or duplicate re-sends is
`dsearch.workload.CHURN_MIX`, drawn from a seeded RNG.
Exit code: 0 on success, 1 if a write had to be abandoned, 2 on bad input.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

from dsearch.client import SearchClient
from dsearch.cluster_status import wait_until_up
from dsearch.ops_file import read_ops
from dsearch.workload import DocVersions, WriteAbandonedError, churn_ops, read_corpus, send_ops


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coordinator", default=os.environ.get("DSEARCH_COORDINATOR", "127.0.0.1:50050"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"), help="texts for replacements")
    parser.add_argument("--ops", type=Path, default=Path("data/run/ops.jsonl"))
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rate", type=float, default=0.0, help="ops per second (0 = as fast as possible)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retry-for", type=float, default=60.0, help="seconds to keep retrying one write")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the coordinator")
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.ops.exists():
        print(f"{args.ops} not found; run load_data.py first", file=sys.stderr)
        return 2
    state = DocVersions.from_ops(read_ops(args.ops))
    texts = [d.text for d in read_corpus(args.corpus)]
    if not state.history or not texts:
        print("need a loaded cluster and a non-empty corpus", file=sys.stderr)
        return 2
    ops = churn_ops(state, texts, args.count, random.Random(args.seed))
    kinds = Counter(op.op_type.value for op in ops)

    await wait_until_up([args.coordinator], args.wait)
    client = SearchClient(args.coordinator)
    try:
        stats = await send_ops(
            client, ops, args.ops, concurrency=args.concurrency, rate=args.rate, retry_for=args.retry_for
        )
    except WriteAbandonedError as exc:
        print(f"abandoned write, ops file is incomplete: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()
    print(
        f"sent {stats.sent} ops ({dict(kinds)}): {stats.applied} applied, {stats.noops} no-ops, "
        f"{stats.retries} retries in {stats.seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
