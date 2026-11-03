"""Workload driver (D-027): churn generation and retrying sends, against a fake coordinator."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import grpc
import pytest

from dsearch.index.operation import ApplyResult, Operation, OpType
from dsearch.ops_file import read_ops
from dsearch.proto import search_pb2
from dsearch.reference.single_node import SingleNodeReference
from dsearch.workload import (
    CorpusDoc,
    DocVersions,
    WriteAbandonedError,
    churn_ops,
    insert_ops,
    read_corpus,
    send_ops,
)

DOCS = [CorpusDoc(f"d{i}", f"text number {i}") for i in range(50)]
TEXTS = [f"replacement {i} alpha beta" for i in range(20)]


def rpc_error(code: grpc.StatusCode) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata(), details="injected")


class FakeCoordinator:
    """Applies writes to a reference index after a random delay (so concurrent
    writes reorder). `failures[(doc_id, version)]` lists errors to raise on
    successive attempts; `applied_then_fail` ops are applied before raising,
    like a write whose ack was lost."""

    def __init__(self, seed: int = 0) -> None:
        self.index = SingleNodeReference()
        self.failures: dict[tuple[str, int], list[grpc.StatusCode]] = {}
        self.applied_then_fail: set[tuple[str, int]] = set()
        self.attempts: Counter[tuple[str, int]] = Counter()
        self._rng = random.Random(seed)

    async def write(self, op: Operation, timeout: float | None = None) -> search_pb2.CoordinatorWriteReply:
        key = (op.doc_id, op.version)
        self.attempts[key] += 1
        await asyncio.sleep(self._rng.random() * 0.002)
        pending = self.failures.get(key)
        if pending:
            if key in self.applied_then_fail:
                self.index.apply(op)
            raise rpc_error(pending.pop(0))
        result = self.index.apply(op)
        return search_pb2.CoordinatorWriteReply(applied=result is ApplyResult.APPLIED)


def test_read_corpus_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps({"doc_id": d.doc_id, "title": "t", "text": d.text}) + "\n" for d in DOCS))
    assert read_corpus(path, limit=3) == DOCS[:3]
    assert read_corpus(path) == DOCS


def test_churn_is_reproducible() -> None:
    def run() -> list[Operation]:
        return churn_ops(DocVersions.from_ops(insert_ops(DOCS)), TEXTS, 500, random.Random(7))

    assert run() == run()
    assert run() != churn_ops(DocVersions.from_ops(insert_ops(DOCS)), TEXTS, 500, random.Random(8))


def test_churn_versions_follow_the_rules() -> None:
    """New ops raise the doc's version by one and fit its state (replace/delete
    a live doc, re-insert a deleted one); re-sends repeat an earlier op unchanged
    and are no-ops. So every (doc_id, version) names exactly one operation."""
    state = DocVersions.from_ops(insert_ops(DOCS))
    ops = churn_ops(state, TEXTS, 3000, random.Random(1))
    reference = SingleNodeReference()
    sent: dict[tuple[str, int], Operation] = {}
    for op in insert_ops(DOCS):
        reference.apply(op)
        sent[(op.doc_id, op.version)] = op

    kinds: Counter[str] = Counter()
    for op in ops:
        before = reference.state()[op.doc_id]
        result = reference.apply(op)
        if (op.doc_id, op.version) in sent:
            assert sent[(op.doc_id, op.version)] == op
            assert result is ApplyResult.STALE
            kinds["duplicate" if op.version == before.version else "stale"] += 1
            continue
        sent[(op.doc_id, op.version)] = op
        assert op.version == before.version + 1
        assert result is ApplyResult.APPLIED
        was_live = before.text is not None
        if op.op_type is OpType.INSERT:
            assert not was_live
        else:
            assert was_live
        kinds[op.op_type.value] += 1

    assert set(kinds) == {"insert", "replace", "delete", "stale", "duplicate"}
    assert {d: state.latest(d).version for d in state.history} == {
        d: s.version for d, s in reference.state().items()
    }


def test_churn_continues_versions_from_an_ops_file(tmp_path: Path) -> None:
    first = insert_ops(DOCS) + churn_ops(DocVersions.from_ops(insert_ops(DOCS)), TEXTS, 300, random.Random(1))
    shuffled = first[:]
    random.Random(2).shuffle(shuffled)  # ops files are in ack order, not version order
    second = churn_ops(DocVersions.from_ops(shuffled), TEXTS, 300, random.Random(1))
    seen = {(op.doc_id, op.version): op for op in first}
    for op in second:
        assert seen.setdefault((op.doc_id, op.version), op) == op


async def test_send_records_every_acked_op_once(tmp_path: Path) -> None:
    ops = insert_ops(DOCS) + churn_ops(DocVersions.from_ops(insert_ops(DOCS)), TEXTS, 400, random.Random(3))
    fake = FakeCoordinator()
    ops_path = tmp_path / "ops.jsonl"
    stats = await send_ops(fake, ops, ops_path, concurrency=16)

    assert sorted(read_ops(ops_path), key=repr) == sorted(ops, key=repr)
    assert stats.sent == len(ops) and stats.retries == 0
    reference = SingleNodeReference()
    for op in read_ops(ops_path):
        reference.apply(op)
    assert fake.index.state() == reference.state()


async def test_failed_writes_are_retried_with_the_same_version(tmp_path: Path) -> None:
    ops = insert_ops(DOCS[:10]) + [Operation(OpType.REPLACE, "d1", 2, "new"), Operation(OpType.DELETE, "d2", 2)]
    fake = FakeCoordinator()
    fake.failures[("d1", 2)] = [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]
    fake.failures[("d2", 2)] = [grpc.StatusCode.DEADLINE_EXCEEDED]
    fake.applied_then_fail.add(("d2", 2))  # applied, but the ack was lost
    ops_path = tmp_path / "ops.jsonl"

    stats = await send_ops(fake, ops, ops_path, concurrency=4, retry_delay=0.001)

    assert stats.retries == 3
    assert fake.attempts[("d1", 2)] == 3 and fake.attempts[("d2", 2)] == 2
    assert stats.applied == len(ops) - 1 and stats.noops == 1  # d2's retry found it already applied
    recorded = read_ops(ops_path)
    assert sorted(recorded, key=repr) == sorted(ops, key=repr)  # once each, same versions
    reference = SingleNodeReference()
    for op in recorded:
        reference.apply(op)
    assert fake.index.state() == reference.state()


async def test_non_retryable_error_abandons_without_recording_the_op(tmp_path: Path) -> None:
    ops = insert_ops(DOCS[:20])
    fake = FakeCoordinator()
    fake.failures[("d5", 1)] = [grpc.StatusCode.INVALID_ARGUMENT]
    ops_path = tmp_path / "ops.jsonl"

    with pytest.raises(WriteAbandonedError) as info:
        await send_ops(fake, ops, ops_path, concurrency=2)

    assert info.value.op == ops[5]
    assert fake.attempts[("d5", 1)] == 1
    recorded = read_ops(ops_path)
    assert ops[5] not in recorded
    assert {(op.doc_id, op.version) for op in recorded} == {k for k, n in fake.attempts.items() if k != ("d5", 1)}


async def test_retry_budget_runs_out(tmp_path: Path) -> None:
    fake = FakeCoordinator()
    fake.failures[("d0", 1)] = [grpc.StatusCode.UNAVAILABLE] * 1000
    with pytest.raises(WriteAbandonedError):
        await send_ops(fake, insert_ops(DOCS[:1]), tmp_path / "ops.jsonl", retry_for=0.05, retry_delay=0.01)
    assert 2 <= fake.attempts[("d0", 1)] < 1000


async def test_rate_limit_spaces_sends(tmp_path: Path) -> None:
    started = time.monotonic()
    await send_ops(FakeCoordinator(), insert_ops(DOCS[:21]), tmp_path / "ops.jsonl", concurrency=8, rate=200)
    assert time.monotonic() - started >= 20 / 200
