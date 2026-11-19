"""Ways to merge ranked lists from the sparse and dense retrievers.

* Reciprocal Rank Fusion ignores raw scores and uses only ranks, so it needs no
  calibration between BM25 scores (unbounded) and cosine similarity ([-1, 1]).
* Convex combination min-max normalises each list and mixes them with ``alpha``;
  it can beat RRF when tuned, but the best ``alpha`` is corpus-specific.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Ranked = tuple[np.ndarray, np.ndarray]  # (doc indices, scores), best first


def reciprocal_rank_fusion(
    lists: Sequence[Ranked], k: int = 60, weights: Sequence[float] | None = None
) -> Ranked:
    weights = weights or [1.0] * len(lists)
    fused: dict[int, float] = {}
    for (idx, _), w in zip(lists, weights, strict=True):
        for rank, doc in enumerate(idx.tolist()):
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank + 1)
    return _sorted(fused)


def convex_fusion(sparse: Ranked, dense: Ranked, alpha: float = 0.5) -> Ranked:
    """``alpha`` weights the dense list; ``1 - alpha`` weights the sparse list."""
    fused: dict[int, float] = {}
    for (idx, sc), w in ((sparse, 1 - alpha), (dense, alpha)):
        if len(idx) == 0:
            continue
        lo, hi = float(sc.min()), float(sc.max())
        norm = (sc - lo) / (hi - lo) if hi > lo else np.ones_like(sc)
        for doc, s in zip(idx.tolist(), norm.tolist(), strict=True):
            fused[doc] = fused.get(doc, 0.0) + w * s
    return _sorted(fused)


def _sorted(fused: dict[int, float]) -> Ranked:
    if not fused:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    items = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    idx = np.fromiter((d for d, _ in items), dtype=np.int64, count=len(items))
    sc = np.fromiter((s for _, s in items), dtype=np.float32, count=len(items))
    return idx, sc
