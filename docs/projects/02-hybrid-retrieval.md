# Project 2: Hybrid Reranking Retrieval Stack

**Goal:** retrieval that actually finds things, with a measured answer to
"how much accuracy does each extra millisecond buy?"

Code: [`src/hybrid_retrieval/`](../../src/hybrid_retrieval/) ·
Benchmark: [`benchmarks/bench_retrieval.py`](../../benchmarks/bench_retrieval.py)

## Architecture

```
                 ┌── BM25 (sparse, top N) ──┐
query + filter ──┤                          ├── fuse ── cross-encoder rerank (top D) ── top k
                 └── bi-encoder (dense, N) ─┘   RRF or
                                                convex α
      metadata filter → boolean mask applied inside both retrievers
```

| Component | File | Notes |
|---|---|---|
| BM25 | `bm25.py` | From scratch. Per-(doc, term) weights precomputed into a sparse CSC matrix, so a query is a column gather and a sum. |
| Dense | `dense.py` | `Embedder` protocol. `LSAEmbedder` runs offline; `SentenceTransformerEmbedder` wraps any pretrained bi-encoder. Exact cosine search. |
| Fusion | `fusion.py` | Reciprocal Rank Fusion (rank-only, no calibration) and convex min-max fusion (`alpha` weights dense). |
| Rerank | `rerank.py` | `CrossEncoderReranker` over the top `rerank_depth` fused candidates. |
| Filters | `filters.py` | Mongo-style `$eq $ne $gt $gte $lt $lte $in $nin $exists $and $or`. |
| Pipeline | `pipeline.py` | `HybridRetriever.search(query, filter=..., config=...)` returns hits plus per-stage timings. |

### Design decisions

- **Pre-filtering, not post-filtering.** The filter becomes a mask applied
  *before* top-k inside each retriever. Post-filtering a top-100 list can
  return zero results for a selective filter even when matching documents
  exist. See [ADR 0002](../adr/0002-metadata-pre-filtering.md).
- **Rerank depth is the latency dial.** Cross-encoder cost grows linearly
  with the number of candidates it scores, so the benchmark sweeps it.
- **Fusion feeds the reranker.** Hybrid retrieval's job is mostly to raise
  *recall at depth*, so the reranker sees the right document. nDCG of the
  fused list matters less once a reranker sits on top.

## Results: SciFact dev (5,183 abstracts, 300 claims)

Offline run with the corpus-fitted LSA embedder (256 dims), no reranker.
Reproduce with `python benchmarks/bench_retrieval.py`.

| Config | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25 only | 0.642 | 0.769 | 0.866 | 0.606 | 0.1 | 0.1 |
| Dense only (LSA) | 0.466 | 0.656 | 0.875 | 0.414 | 39.8 | 48.1 |
| Hybrid, RRF | 0.559 | 0.722 | 0.918 | 0.514 | 34.3 | 48.8 |
| Hybrid, convex α=0.3 | **0.649** | **0.779** | 0.918 | **0.613** | 35.5 | 48.1 |
| Hybrid, convex α=0.5 | 0.618 | 0.772 | 0.918 | 0.575 | 38.3 | 48.0 |
| Hybrid, convex α=0.7 | 0.558 | 0.725 | **0.928** | 0.513 | 45.5 | 49.2 |

**What this shows**

1. **Hybrid's main win is recall at depth.** Recall@100 rises from 0.866
   (BM25) to 0.918 to 0.928. That is the pool a reranker picks from.
2. **Fusion method matters when one retriever is weak.** RRF treats both lists
   as equals, so a weak dense model drags nDCG@10 down (0.642 → 0.559).
   Convex fusion with a low dense weight keeps BM25's precision and still gains
   recall.
3. **Latency is dominated by query encoding**, not search: BM25 takes
   0.1 ms and fusion 0.2 ms, while LSA query encoding takes about 35 ms.

### With pretrained models

The `Retrieval benchmark` GitHub Actions workflow runs the same script with
`BAAI/bge-small-en-v1.5` and the `ms-marco-MiniLM-L-6-v2` cross-encoder, and
sweeps rerank depth over 20, 50 and 100. Its table will replace this note once
it has run.

## Running it

```python
from hybrid_retrieval import Document, HybridRetriever, LSAEmbedder, RetrievalConfig

docs = [Document("1", "Aspirin lowers heart attack risk", {"year": 2021}), ...]
r = HybridRetriever(docs, embedder=LSAEmbedder())
res = r.search(
    "does aspirin help the heart?",
    filter={"year": {"$gte": 2020}},
    config=RetrievalConfig(fusion="convex", alpha=0.3, top_k=5),
)
res.doc_ids, res.timings_ms
```

## Next

Project 1 (Context Assembler) consumes `HybridRetriever` as its retrieval
source and adds token budgeting on top.
