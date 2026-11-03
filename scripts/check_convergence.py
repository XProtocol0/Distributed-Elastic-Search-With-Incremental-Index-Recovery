"""Check a running cluster against the single-node reference.

    python scripts/check_convergence.py --config cluster.toml --ops ops.jsonl [--k 10] [--queries q.txt]

The reference is built by replaying the ops file (JSON lines, D-023) that the
workload driver recorded. Every copy of every shard is compared (D-022).
Exit code: 0 if there are zero differences, 1 otherwise, 2 on bad input.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dsearch.config import ConfigError, load_config
from dsearch.convergence import check_convergence
from dsearch.ops_file import read_ops
from dsearch.reference.single_node import SingleNodeReference


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="cluster TOML (default: $DSEARCH_CONFIG)")
    parser.add_argument("--ops", type=Path, required=True, help="JSON-lines ops file the workload wrote")
    parser.add_argument("--queries", type=Path, help="one query per line (default: derived from the corpus)")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    env = dict(os.environ)
    if args.config is not None:
        env["DSEARCH_CONFIG"] = str(args.config)
    try:
        config = load_config(env)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if len(config.shards) != config.num_shards:
        print("config error: the [[shards]] topology is required", file=sys.stderr)
        return 2

    reference = SingleNodeReference()
    ops = read_ops(args.ops)
    for op in ops:
        reference.apply(op)
    queries = None
    if args.queries is not None:
        queries = [q.strip() for q in args.queries.read_text().splitlines() if q.strip()]

    report = await check_convergence(config.shards, reference, queries, args.k, config.rpc_timeout)
    print(f"reference: {len(ops)} operations, {reference.index.live_count} live docs")
    print(report.summary())
    return 0 if report.converged else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
