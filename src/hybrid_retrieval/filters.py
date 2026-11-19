"""Metadata filtering with a small Mongo-style query language.

    {"year": {"$gte": 2020}, "source": {"$in": ["arxiv", "pubmed"]}, "lang": "en"}

A filter compiles to a boolean mask over the corpus. Retrievers apply the mask
*before* top-k selection (pre-filtering), so a restrictive filter never returns
fewer than k results while matching documents exist. Post-filtering would.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from hybrid_retrieval.types import Document

Filter = dict[str, Any]

_MISSING = object()

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": operator.eq,
    "$ne": operator.ne,
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
    "$in": lambda v, arg: v in arg,
    "$nin": lambda v, arg: v not in arg,
}


def _match_field(value: Any, cond: Any) -> bool:
    if not isinstance(cond, dict):
        return value is not _MISSING and value == cond
    for op, arg in cond.items():
        if op == "$exists":
            if (value is not _MISSING) != bool(arg):
                return False
            continue
        if op not in _OPS:
            raise ValueError(f"Unknown filter operator: {op}")
        if value is _MISSING:
            return False
        try:
            if not _OPS[op](value, arg):
                return False
        except TypeError:  # e.g. comparing str with int
            return False
    return True


def matches(metadata: dict[str, Any], flt: Filter) -> bool:
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(metadata, f) for f in cond):
                return False
        elif key == "$or":
            if not any(matches(metadata, f) for f in cond):
                return False
        elif not _match_field(metadata.get(key, _MISSING), cond):
            return False
    return True


def build_mask(docs: Sequence[Document], flt: Filter | None) -> np.ndarray | None:
    """Return a boolean mask of documents that satisfy ``flt``, or None for no filter."""
    if not flt:
        return None
    return np.fromiter((matches(d.metadata, flt) for d in docs), dtype=bool, count=len(docs))
