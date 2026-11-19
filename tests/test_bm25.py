import numpy as np

from hybrid_retrieval.bm25 import BM25Index
from hybrid_retrieval.text import tokenize


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Aspirin and the HEART") == ["aspirin", "heart"]


def test_bm25_ranks_matching_documents_first(docs):
    idx = BM25Index().fit([d.text for d in docs])
    top, scores = idx.search("aspirin heart attack", k=3)
    assert docs[top[0]].id == "d1"
    assert all(np.diff(scores) <= 0)


def test_bm25_returns_nothing_for_unknown_terms(docs):
    idx = BM25Index().fit([d.text for d in docs])
    top, _ = idx.search("zzzz qqqq", k=5)
    assert len(top) == 0


def test_rare_terms_weigh_more_than_common_ones():
    texts = ["common rare", "common", "common", "common"]
    s = BM25Index().fit(texts).scores("common rare")
    assert s[0] > s[1] > 0
