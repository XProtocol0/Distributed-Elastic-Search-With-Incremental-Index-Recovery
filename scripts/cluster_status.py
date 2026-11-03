"""Print every shard copy's replication and recovery status (D-025).

    python scripts/cluster_status.py --config cluster.toml [--wait-replicas 60]

With --wait-replicas, first wait (up to that many seconds) until every
replica is live and has applied everything its primary has; exit 1 on timeout.
--json prints {address: status} instead of a table (down nodes are omitted).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict

from dsearch.cluster_status import get_status, wait_for_replicas
from dsearch.config import ConfigError, load_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="cluster TOML (default: $DSEARCH_CONFIG)")
    parser.add_argument("--wait-replicas", type=float, metavar="SECONDS")
    parser.add_argument("--json", action="store_true")
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

    exit_code = 0
    if args.wait_replicas is not None:
        try:
            await wait_for_replicas(config.shards, args.wait_replicas)
        except TimeoutError as exc:
            print(exc, file=sys.stderr)
            exit_code = 1

    if args.json:
        statuses = {}
        for shard in config.shards:
            for address in (shard.primary, *shard.replicas):
                try:
                    status = await get_status(address)
                except grpc.aio.AioRpcError:
                    continue
                statuses[address] = MessageToDict(
                    status, always_print_fields_with_no_presence=True, preserving_proto_field_name=True
                )
        print(json.dumps(statuses, indent=2))
        return exit_code

    print(f"{'node':<16} {'role':<8} {'applied':>8} {'logged':>8} {'log KiB':>8} {'live':<5} recoveries")
    for shard in config.shards:
        for address in (shard.primary, *shard.replicas):
            try:
                s = await get_status(address)
            except grpc.aio.AioRpcError as exc:
                print(f"{address:<16} DOWN ({exc.code().name})")
                continue
            sessions = ", ".join(
                f"{r.from_seq}->{r.caught_up_seq} ({r.entries_replayed} in {r.seconds:.2f}s)" for r in s.recoveries
            )
            live = "yes" if s.live else ("-" if s.role == "primary" else "no")
            print(
                f"{s.node_id:<16} {s.role:<8} {s.applied_seq:>8} {s.last_logged_seq:>8} "
                f"{s.log_bytes / 1024:>8.0f} {live:<5} {sessions}"
                + (f"  full_rebuilds={s.full_rebuilds}" if s.full_rebuilds else "")
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
