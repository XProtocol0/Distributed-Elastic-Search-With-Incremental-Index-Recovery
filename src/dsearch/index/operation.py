"""Write operations applied to an index (see D-003)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OpType(str, Enum):
    INSERT = "insert"
    REPLACE = "replace"
    DELETE = "delete"


@dataclass(frozen=True)
class Operation:
    """One versioned write. `(doc_id, version)` must name exactly one operation.

    INSERT and REPLACE are both version-gated upserts; the type is kept for
    logging and metrics. DELETE carries no text and leaves a tombstone.
    """

    op_type: OpType
    doc_id: str
    version: int
    text: str | None = None

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")
        if self.op_type is OpType.DELETE:
            if self.text is not None:
                raise ValueError("delete must not carry text")
        elif self.text is None:
            raise ValueError(f"{self.op_type.value} needs text")


class ApplyResult(str, Enum):
    APPLIED = "applied"
    STALE = "stale"  # version not newer than the stored one: no-op
