from collections import Counter

from dsearch.coordinator.router import shard_for


def test_known_values_are_stable() -> None:
    # crc32 is fixed forever, unlike Python's per-process salted hash().
    assert [shard_for(d, 3) for d in ["doc1", "doc2", "doc3", "Café"]] == [2, 1, 2, 0]


def test_in_range_and_roughly_even() -> None:
    counts = Counter(shard_for(f"doc{i}", 4) for i in range(4000))
    assert set(counts) == {0, 1, 2, 3}
    assert all(800 < c < 1200 for c in counts.values())


def test_single_shard_gets_everything() -> None:
    assert {shard_for(f"d{i}", 1) for i in range(50)} == {0}
