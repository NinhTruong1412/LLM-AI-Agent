import pytest

from agentcore.tracing import Tracer
from hybrid_retrieval.dense import LSAEmbedder
from hybrid_retrieval.evaluate import mrr_at_k, ndcg_at_k, recall_at_k
from hybrid_retrieval.pipeline import HybridRetriever, RetrievalConfig
from hybrid_retrieval.rerank import CallableReranker


@pytest.fixture
def retriever(docs):
    return HybridRetriever(docs, embedder=LSAEmbedder(dim=4), tracer=Tracer())


@pytest.mark.parametrize("mode", ["sparse", "dense", "hybrid"])
def test_every_mode_returns_ranked_hits(retriever, mode):
    res = retriever.search("aspirin heart", config=RetrievalConfig(mode=mode, top_k=3))
    assert 0 < len(res.hits) <= 3
    assert [h.rank for h in res.hits] == list(range(len(res.hits)))
    assert "total" in res.timings_ms


def test_filter_is_applied_before_top_k(retriever):
    res = retriever.search(
        "documents retrieval",
        filter={"source": "arxiv"},
        config=RetrievalConfig(mode="sparse", top_k=10),
    )
    ids = res.doc_ids
    assert ids and all(i in {"d3", "d4", "d7", "d8"} for i in ids)


def test_reranker_reorders_head(docs):
    # A reranker that strongly prefers d5 should move it to the top.
    rr = CallableReranker(lambda q, ds: [10.0 if "Myocardial" in d else 0.0 for d in ds])
    ret = HybridRetriever(docs, embedder=LSAEmbedder(dim=4), reranker=rr)
    res = ret.search("aspirin", config=RetrievalConfig(rerank_depth=5, top_k=3))
    assert res.doc_ids[0] == "d5"
    assert "rerank" in res.timings_ms


def test_dense_mode_without_embedder_raises(docs):
    with pytest.raises(ValueError):
        HybridRetriever(docs).search("x", config=RetrievalConfig(mode="dense"))


def test_metrics():
    ranked, rel = ["a", "b", "c"], {"b", "z"}
    assert mrr_at_k(ranked, rel) == 0.5
    assert recall_at_k(ranked, rel, 3) == 0.5
    assert 0 < ndcg_at_k(ranked, rel) < 1
    assert ndcg_at_k(["b", "z"], rel) == pytest.approx(1.0)
