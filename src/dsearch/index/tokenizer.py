"""Tokenizer shared by every node and the reference (see D-005)."""

from __future__ import annotations

import re

# Runs of Unicode letters or digits (\w without the underscore).
_TOKEN = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase `text` and split it into letter/digit tokens, keeping duplicates."""
    return _TOKEN.findall(text.lower())


def query_terms(query: str) -> list[str]:
    """Unique query terms in sorted order, the fixed order BM25 sums in (D-006)."""
    return sorted(set(tokenize(query)))
