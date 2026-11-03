from dsearch.config import SearchPreference, ShardTopology
from dsearch.coordinator.search import ScatterGather, merge_top_k
from dsearch.index.inverted_index import SearchHit

SHARD = ShardTopology(0, "p:1", ("r1:1", "r2:1"))


def test_merge_orders_by_score_then_doc_id_and_cuts_at_k() -> None:
    a = [SearchHit("a1", 3.0), SearchHit("a2", 1.0)]
    b = [SearchHit("b1", 3.0), SearchHit("b2", 2.0)]
    c: list[SearchHit] = []
    assert [h.doc_id for h in merge_top_k([a, b, c], 3)] == ["a1", "b1", "b2"]
    assert merge_top_k([a, b], 0) == []
    assert len(merge_top_k([a, b], 10)) == 4


def test_primary_preference_tries_primary_first_then_replicas() -> None:
    sg = ScatterGather((SHARD,), SearchPreference.PRIMARY, 1.0)
    assert sg.copy_order(SHARD) == ["p:1", "r1:1", "r2:1"]
    assert sg.copy_order(SHARD) == ["p:1", "r1:1", "r2:1"]


def test_round_robin_rotates_and_keeps_others_as_fallback() -> None:
    sg = ScatterGather((SHARD,), SearchPreference.ROUND_ROBIN, 1.0)
    orders = [sg.copy_order(SHARD) for _ in range(4)]
    assert [o[0] for o in orders] == ["p:1", "r1:1", "r2:1", "p:1"]
    assert all(sorted(o) == ["p:1", "r1:1", "r2:1"] for o in orders)
