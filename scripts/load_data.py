"""Load a corpus into a fresh cluster through the coordinator.

    python scripts/load_data.py --corpus data/corpus.jsonl --ops data/run/ops.jsonl [--limit N]

Every doc is inserted at version 1, and each acknowledged write is recorded
in the ops file (D-023) that the convergence checker replays into the
reference. A load starts a new run, so the ops file must not exist yet.
Exit code: 0 on success, 1 if a write had to be abandoned (D-027), 2 on bad input.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dsearch.client import SearchClient
from dsearch.cluster_status import wait_until_up
from dsearch.workload import WriteAbandonedError, insert_ops, read_corpus, send_ops


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coordinator", default=os.environ.get("DSEARCH_COORDINATOR", "127.0.0.1:50050"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    parser.add_argument("--ops", type=Path, default=Path("data/run/ops.jsonl"))
    parser.add_argument("--limit", type=int, help="load only the first N docs")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retry-for", type=float, default=60.0, help="seconds to keep retrying one write")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the coordinator")
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.ops.exists() and args.ops.stat().st_size > 0:
        print(f"{args.ops} already exists; a load starts a new run on a fresh cluster", file=sys.stderr)
        return 2
    docs = read_corpus(args.corpus, args.limit)
    if not docs:
        print(f"no documents in {args.corpus}", file=sys.stderr)
        return 2

    await wait_until_up([args.coordinator], args.wait)
    client = SearchClient(args.coordinator)
    try:
        stats = await send_ops(
            client, insert_ops(docs), args.ops, concurrency=args.concurrency, retry_for=args.retry_for
        )
    except WriteAbandonedError as exc:
        print(f"abandoned write, ops file is incomplete: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()
    print(
        f"loaded {stats.applied} docs ({stats.noops} no-ops, {stats.retries} retries) "
        f"in {stats.seconds:.1f}s = {stats.sent / stats.seconds:.0f} writes/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
