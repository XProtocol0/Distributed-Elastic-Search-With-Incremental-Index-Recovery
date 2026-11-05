"""Workload driver: load a corpus, generate replace/delete churn, send it.

The driver owns document versions: it remembers the highest version it has
sent for each doc (rebuilt from the ops file between runs), so every new
write gets the next one. Writes go out with bounded concurrency, so two
writes to the same doc can arrive in either order; versions make the final
state the same either way.

A write that fails with UNAVAILABLE or DEADLINE_EXCEEDED has an unknown
outcome. It is retried with the *same* version until acknowledged (a
duplicate is a no-op), so every op recorded in the ops file is exactly one
the cluster applied or ignored by version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO

import grpc

from dsearch.index.operation import Operation, OpType
from dsearch.ops_file import operation_to_json
from dsearch.proto import search_pb2

log = logging.getLogger(__name__)

RETRYABLE = frozenset({grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED})

# Share of each kind of churn operation (see `churn_ops`).
CHURN_MIX: dict[str, float] = {
    "replace": 0.55,
    "delete": 0.20,
    "reinsert": 0.10,
    "stale": 0.10,  # an older op of the doc sent again (a delayed request): must be a no-op
    "duplicate": 0.05,  # the doc's latest op sent again: must be a no-op
}


# ---- corpus -------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str


def read_corpus(path: Path, limit: int | None = None) -> list[CorpusDoc]:
    """Read a JSON-lines corpus of `{"doc_id": ..., "text": ...}` records."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(docs) >= limit:
                break
            if line.strip():
                record = json.loads(line)
                docs.append(CorpusDoc(str(record["doc_id"]), record["text"]))
    return docs


def insert_ops(docs: Iterable[CorpusDoc]) -> list[Operation]:
    """Initial load: every doc inserted at version 1."""
    return [Operation(OpType.INSERT, d.doc_id, 1, d.text) for d in docs]


# ---- churn generation ---------------------------------------------------


class _Pool:
    """A set of doc ids with O(1) add, remove and random choice."""

    def __init__(self, items: Iterable[str] = ()) -> None:
        self._items: list[str] = []
        self._index: dict[str, int] = {}
        for item in items:
            self.add(item)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: str) -> None:
        if item not in self._index:
            self._index[item] = len(self._items)
            self._items.append(item)

    def remove(self, item: str) -> None:
        i = self._index.pop(item)
        last = self._items.pop()
        if i < len(self._items):
            self._items[i] = last
            self._index[last] = i

    def choice(self, rng: random.Random) -> str:
        return self._items[rng.randrange(len(self._items))]


@dataclass
class DocVersions:
    """Every distinct op the driver has sent, per doc, in version order.

    Re-sends (stale or duplicate) reuse an op from here unchanged, so each
    (doc_id, version) always names exactly one operation.
    """

    history: dict[str, list[Operation]] = field(default_factory=dict)

    @classmethod
    def from_ops(cls, ops: Iterable[Operation]) -> "DocVersions":
        state = cls()
        for op in sorted(ops, key=lambda o: o.version):  # ops files are in ack order
            state.record(op)
        return state

    def latest(self, doc_id: str) -> Operation:
        return self.history[doc_id][-1]

    def record(self, op: Operation) -> None:
        ops = self.history.setdefault(op.doc_id, [])
        if not ops or op.version > ops[-1].version:
            ops.append(op)


def churn_ops(
    state: DocVersions,
    texts: Sequence[str],
    count: int,
    rng: random.Random,
    mix: Mapping[str, float] = CHURN_MIX,
) -> list[Operation]:
    """`count` replace/delete/re-insert ops plus deliberate stale and duplicate
    re-sends, drawn by `mix` (keys as in CHURN_MIX). Updates `state` as it goes."""
    ordered = sorted(state.history)  # fixed order so a seed reproduces the run
    live = _Pool(d for d in ordered if state.latest(d).op_type is not OpType.DELETE)
    dead = _Pool(d for d in ordered if state.latest(d).op_type is OpType.DELETE)
    kinds = list(mix)
    weights = list(mix.values())
    ops: list[Operation] = []
    while len(ops) < count:
        kind = rng.choices(kinds, weights)[0]
        if kind == "reinsert" and not dead:
            kind = "replace"
        if kind != "reinsert" and not live:
            if not dead:
                raise ValueError("churn needs at least one loaded document")
            kind = "reinsert"
        doc_id = (dead if kind == "reinsert" else live).choice(rng)
        history = state.history[doc_id]
        latest = history[-1]

        if kind == "replace":
            op = Operation(OpType.REPLACE, doc_id, latest.version + 1, rng.choice(texts))
        elif kind == "delete":
            op = Operation(OpType.DELETE, doc_id, latest.version + 1, None)
            live.remove(doc_id)
            dead.add(doc_id)
        elif kind == "reinsert":
            op = Operation(OpType.INSERT, doc_id, latest.version + 1, rng.choice(texts))
            dead.remove(doc_id)
            live.add(doc_id)
        elif kind == "stale" and len(history) > 1:
            op = rng.choice(history[:-1])
        else:  # duplicate, or stale on a doc that has no older op
            op = latest
        state.record(op)
        ops.append(op)
    return ops


# ---- sending ------------------------------------------------------------


class Writer(Protocol):
    async def write(self, op: Operation, timeout: float | None = None) -> search_pb2.CoordinatorWriteReply: ...


class WriteAbandonedError(Exception):
    """A write kept failing past the retry budget; its outcome is unknown."""

    def __init__(self, op: Operation, cause: grpc.aio.AioRpcError) -> None:
        super().__init__(
            f"{op.op_type.value} doc={op.doc_id} v={op.version} failed: {cause.code().name} {cause.details()}"
        )
        self.op = op


@dataclass
class SendStats:
    sent: int = 0
    applied: int = 0
    noops: int = 0  # acknowledged but not newer than the stored version
    retries: int = 0
    seconds: float = 0.0


class _Pacer:
    """Spaces sends `1/rate` seconds apart across all workers (rate 0 = no limit)."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate if rate > 0 else 0.0
        self._next = time.monotonic()

    async def wait(self) -> None:
        if not self._interval:
            return
        now = time.monotonic()
        slot = max(self._next, now)
        self._next = slot + self._interval
        await asyncio.sleep(slot - now)


async def send_ops(
    client: Writer,
    ops: Sequence[Operation],
    ops_path: Path,
    *,
    concurrency: int = 32,
    rate: float = 0.0,
    write_timeout: float = 10.0,
    retry_for: float = 60.0,
    retry_delay: float = 0.2,
) -> SendStats:
    """Send `ops` and append each acknowledged one to the ops file.

    Raises WriteAbandonedError if a write can't be acknowledged within
    `retry_for` seconds; ops acknowledged until then are still recorded, but
    the ops file no longer describes the cluster exactly.
    """
    stats = SendStats()
    pacer = _Pacer(rate)
    pending = iter(ops)
    failures: list[WriteAbandonedError] = []
    started = time.monotonic()

    async def send_one(op: Operation) -> search_pb2.CoordinatorWriteReply:
        give_up_at = time.monotonic() + retry_for
        while True:
            try:
                return await client.write(op, timeout=write_timeout)
            except grpc.aio.AioRpcError as exc:
                if exc.code() not in RETRYABLE or time.monotonic() >= give_up_at:
                    raise WriteAbandonedError(op, exc) from exc
                stats.retries += 1
                log.warning("retrying doc=%s v=%d after %s", op.doc_id, op.version, exc.code().name)
                await asyncio.sleep(retry_delay)

    async def worker(out: TextIO) -> None:
        for op in pending:  # shared iterator: each op is taken by exactly one worker
            if failures:
                return
            await pacer.wait()
            stats.sent += 1
            try:
                reply = await send_one(op)
            except WriteAbandonedError as exc:
                failures.append(exc)
                return
            if reply.applied:
                stats.applied += 1
            else:
                stats.noops += 1
            out.write(operation_to_json(op) + "\n")

    ops_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ops_path, "a", encoding="utf-8") as out:
        await asyncio.gather(*(worker(out) for _ in range(max(1, concurrency))))
    stats.seconds = time.monotonic() - started
    if failures:
        raise failures[0]
    return stats
