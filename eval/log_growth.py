"""Update-log growth against the number of updates.

    python eval/log_growth.py [--updates 50000]

Runs directly on a `ShardStore` (the same log code every node uses). Log
bytes are deterministic for a given op stream, so the result does not
depend on the process or Docker environment. Scenarios, all starting from
the same 5,000 loaded docs except insert-only:

  insert-only     new docs only
  churn           the workload mix (replace-heavy)
  delete-heavy    mostly deletes and re-inserts
  churn+truncate  churn with snapshot_every=5000, log_retention=5000

Every 1,000 updates it records the log size, the SQLite state size and the
live doc count.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import sys
import tempfile
from pathlib import Path

from harness import RAW, ROOT

from dsearch.index.operation import Operation
from dsearch.shard.store import DB_FILE, ShardStore
from dsearch.workload import CHURN_MIX, DocVersions, churn_ops, insert_ops, read_corpus

LOADED = 5_000
DELETE_HEAVY = {"replace": 0.10, "delete": 0.45, "reinsert": 0.40, "stale": 0.03, "duplicate": 0.02}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--every", type=int, default=1_000)
    parser.add_argument("--out", type=Path, default=RAW / "log_growth.csv")
    return parser.parse_args(argv)


def db_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.parent.glob(DB_FILE + "*"))


async def run_scenario(
    name: str, preload: list[Operation], updates: list[Operation], every: int,
    snapshot_every: int, log_retention: int,
) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        store = await ShardStore.open(Path(tmp), snapshot_every, log_retention)
        for op in preload:
            await store.append_new(op)
        base_seq = store.applied_seq
        for i, op in enumerate(updates, start=1):
            await store.append_new(op)
            if i % every == 0:
                rows.append({
                    "scenario": name, "updates": i, "log_bytes": store.update_log.size_bytes,
                    "log_entries": store.update_log.last_seq - store.update_log.first_seq + 1,
                    "db_bytes": db_bytes(Path(tmp) / DB_FILE), "live_docs": store.index.live_count,
                    "preloaded_seq": base_seq,
                })
        await store.close()
    print(f"{name}: {rows[-1]['log_bytes'] / 1e6:.1f} MB log after {len(updates)} updates", flush=True)
    return rows


async def run(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.corpus)
    texts = [d.text for d in corpus]
    load = insert_ops(corpus[:LOADED])
    fresh = insert_ops(corpus[LOADED : LOADED + args.updates])
    churn = churn_ops(DocVersions.from_ops(load), texts, args.updates, random.Random(1), CHURN_MIX)
    deletes = churn_ops(DocVersions.from_ops(load), texts, args.updates, random.Random(1), DELETE_HEAVY)

    rows = []
    rows += await run_scenario("insert-only", [], fresh, args.every, 0, 0)
    rows += await run_scenario("churn", load, churn, args.every, 0, 0)
    rows += await run_scenario("delete-heavy", load, deletes, args.every, 0, 0)
    rows += await run_scenario("churn+truncate", load, churn, args.every, 5_000, 5_000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    asyncio.run(run(parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
