"""Write routing (D-017)."""

from __future__ import annotations

import zlib


def shard_for(doc_id: str, num_shards: int) -> int:
    """The shard that owns `doc_id`: crc32(utf-8 bytes) % num_shards.

    Stable across processes and runs, unlike Python's salted hash().
    """
    return zlib.crc32(doc_id.encode("utf-8")) % num_shards
