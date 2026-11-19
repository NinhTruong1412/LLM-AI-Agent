from __future__ import annotations

import numpy as np


def top_k(
    scores: np.ndarray, k: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and scores of the k highest scores, best first, honouring an optional mask."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=scores.dtype)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx, scores[idx]
