import numpy as np

from hybrid_retrieval.fusion import convex_fusion, reciprocal_rank_fusion
from hybrid_retrieval.topk import top_k


def r(*idx):
    return np.array(idx), np.linspace(1, 0.1, len(idx))


def test_rrf_rewards_agreement():
    idx, _ = reciprocal_rank_fusion([r(1, 2, 3), r(3, 1, 4)])
    assert idx[0] == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert set(idx) == {1, 2, 3, 4}


def test_convex_alpha_extremes():
    sparse, dense = r(1, 2), r(3, 4)
    assert convex_fusion(sparse, dense, alpha=0.0)[0][0] == 1
    assert convex_fusion(sparse, dense, alpha=1.0)[0][0] == 3


def test_fusion_handles_empty_lists():
    empty = (np.empty(0, dtype=np.int64), np.empty(0))
    assert len(reciprocal_rank_fusion([empty, empty])[0]) == 0
    assert list(convex_fusion(empty, r(5))[0]) == [5]


def test_top_k_respects_mask():
    scores = np.array([5.0, 4.0, 3.0, 2.0])
    idx, _ = top_k(scores, 3, mask=np.array([False, True, False, True]))
    assert list(idx) == [1, 3]
