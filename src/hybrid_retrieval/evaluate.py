"""IR metrics and a small benchmark harness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from hybrid_retrieval.pipeline import HybridRetriever, RetrievalConfig


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1 / (i + 1)
    return 0.0


@dataclass
class BenchResult:
    name: str
    ndcg10: float
    recall10: float
    recall100: float
    mrr10: float
    p50_ms: float
    p95_ms: float
    stage_p50_ms: dict[str, float] = field(default_factory=dict)

    def row(self) -> str:
        return (
            f"| {self.name} | {self.ndcg10:.3f} | {self.recall10:.3f} | {self.recall100:.3f} "
            f"| {self.mrr10:.3f} | {self.p50_ms:.1f} | {self.p95_ms:.1f} |"
        )


TABLE_HEADER = (
    "| Config | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |\n"
    "|---|---|---|---|---|---|---|"
)


def run_benchmark(
    retriever: HybridRetriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, set[str]],
    config: RetrievalConfig,
    name: str,
) -> BenchResult:
    # Retrieve 100 so Recall@100 is measurable; metrics @10 use the head of the list.
    cfg = config.with_(top_k=max(config.top_k, 100), candidates=max(config.candidates, 100))
    ndcg, r10, r100, mrr, totals = [], [], [], [], []
    stages: dict[str, list[float]] = {}
    for qid, text in queries.items():
        res = retriever.search(text, config=cfg)
        ranked, rel = res.doc_ids, qrels[qid]
        ndcg.append(ndcg_at_k(ranked, rel))
        r10.append(recall_at_k(ranked, rel, 10))
        r100.append(recall_at_k(ranked, rel, 100))
        mrr.append(mrr_at_k(ranked, rel))
        totals.append(res.timings_ms["total"])
        for stage, ms in res.timings_ms.items():
            stages.setdefault(stage, []).append(ms)
    return BenchResult(
        name=name,
        ndcg10=float(np.mean(ndcg)),
        recall10=float(np.mean(r10)),
        recall100=float(np.mean(r100)),
        mrr10=float(np.mean(mrr)),
        p50_ms=float(np.percentile(totals, 50)),
        p95_ms=float(np.percentile(totals, 95)),
        stage_p50_ms={s: float(np.percentile(v, 50)) for s, v in stages.items()},
    )
