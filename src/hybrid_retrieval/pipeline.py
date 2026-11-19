"""The hybrid retrieval pipeline.

query ─┬─ BM25 (top N) ──┐
       │                 ├─ fuse (RRF or convex) ─ rerank top D ─ top k
       └─ dense (top N) ─┘
metadata filter → mask applied inside both retrievers (pre-filtering)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from agentcore.tracing import Tracer
from hybrid_retrieval.bm25 import BM25Index
from hybrid_retrieval.dense import DenseIndex, Embedder
from hybrid_retrieval.filters import Filter, build_mask
from hybrid_retrieval.fusion import convex_fusion, reciprocal_rank_fusion
from hybrid_retrieval.rerank import Reranker
from hybrid_retrieval.types import Document, Hit, SearchResult

Mode = Literal["sparse", "dense", "hybrid"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Mode = "hybrid"
    fusion: Literal["rrf", "convex"] = "rrf"
    alpha: float = 0.5  # convex fusion: weight on dense
    rrf_k: int = 60
    candidates: int = 100  # per retriever, before fusion
    rerank_depth: int = 0  # 0 disables reranking
    top_k: int = 10

    def with_(self, **changes) -> RetrievalConfig:
        return replace(self, **changes)


class HybridRetriever:
    def __init__(
        self,
        docs: Sequence[Document],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.docs = list(docs)
        self.config = config or RetrievalConfig()
        self.reranker = reranker
        self.tracer = tracer or Tracer()
        texts = [d.text for d in self.docs]

        with self.tracer.span("index.bm25", n_docs=len(texts)):
            self.sparse = BM25Index().fit(texts)
        self.dense: DenseIndex | None = None
        if embedder is not None:
            with self.tracer.span("index.dense", n_docs=len(texts), embedder=embedder.name):
                self.dense = DenseIndex(embedder).fit(texts)

    def search(
        self, query: str, *, filter: Filter | None = None, config: RetrievalConfig | None = None
    ) -> SearchResult:
        cfg = config or self.config
        timings: dict[str, float] = {}

        def timed(name: str):
            span = self.tracer.span(name)
            return _Recorder(span, timings, name)

        with timed("total"):
            with timed("filter"):
                mask = build_mask(self.docs, filter)

            ranked_lists = []
            if cfg.mode in ("sparse", "hybrid"):
                with timed("sparse"):
                    ranked_lists.append(self.sparse.search(query, cfg.candidates, mask))
            if cfg.mode in ("dense", "hybrid"):
                if self.dense is None:
                    raise ValueError("dense or hybrid mode needs an embedder")
                with timed("dense"):
                    ranked_lists.append(self.dense.search(query, cfg.candidates, mask))

            with timed("fusion"):
                if len(ranked_lists) == 1:
                    idx, scores = ranked_lists[0]
                elif cfg.fusion == "rrf":
                    idx, scores = reciprocal_rank_fusion(ranked_lists, k=cfg.rrf_k)
                else:
                    idx, scores = convex_fusion(*ranked_lists, alpha=cfg.alpha)

            if cfg.rerank_depth > 0 and self.reranker is not None and len(idx):
                with timed("rerank"):
                    idx, scores = self._rerank(query, idx, scores, cfg.rerank_depth)

        hits = [
            Hit(doc_id=self.docs[i].id, score=float(s), rank=r)
            for r, (i, s) in enumerate(zip(idx[: cfg.top_k], scores[: cfg.top_k], strict=True))
        ]
        return SearchResult(hits=hits, timings_ms=timings)

    def _rerank(
        self, query: str, idx: np.ndarray, scores: np.ndarray, depth: int
    ) -> tuple[np.ndarray, np.ndarray]:
        head, tail = idx[:depth], idx[depth:]
        rr = self.reranker.score(query, [self.docs[i].text for i in head])
        order = np.argsort(-rr, kind="stable")
        head, head_scores = head[order], rr[order].astype(np.float32)
        # Keep un-reranked candidates below every reranked one.
        tail_scores = np.full(len(tail), float(head_scores.min()) - 1.0, dtype=np.float32)
        return np.concatenate([head, tail]), np.concatenate([head_scores, tail_scores])


class _Recorder:
    def __init__(self, span_cm, timings: dict[str, float], name: str) -> None:
        self._cm = span_cm
        self._timings = timings
        self._name = name

    def __enter__(self):
        self._span = self._cm.__enter__()
        return self._span

    def __exit__(self, *exc):
        out = self._cm.__exit__(*exc)
        self._timings[self._name] = self._span.duration_ms
        return out
