"""Benchmark the Hybrid Reranking Retrieval Stack on SciFact.

Offline (no model downloads):
    python benchmarks/bench_retrieval.py

With pretrained models (needs Hugging Face access):
    pip install -e ".[models]"
    python benchmarks/bench_retrieval.py --embedder bge-small --reranker minilm

Writes a markdown table to stdout and full results to benchmarks/results/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from hybrid_retrieval.datasets import load_scifact
from hybrid_retrieval.dense import LSAEmbedder
from hybrid_retrieval.evaluate import TABLE_HEADER, run_benchmark
from hybrid_retrieval.pipeline import HybridRetriever, RetrievalConfig

RESULTS = Path(__file__).parent / "results"


def make_embedder(name: str):
    if name == "lsa":
        return LSAEmbedder(dim=256)
    from hybrid_retrieval.dense import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder({"bge-small": "BAAI/bge-small-en-v1.5"}.get(name, name))


def make_reranker(name: str):
    if name == "none":
        return None
    from hybrid_retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker({"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2"}.get(name, name))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embedder", default="lsa", help="lsa | bge-small | any sentence-transformers id"
    )
    p.add_argument("--reranker", default="none", help="none | minilm | any cross-encoder id")
    p.add_argument("--split", default="dev")
    p.add_argument("--rerank-depths", default="20,50,100")
    args = p.parse_args()

    docs, queries, qrels = load_scifact(args.split)
    t0 = time.perf_counter()
    retriever = HybridRetriever(
        docs, embedder=make_embedder(args.embedder), reranker=make_reranker(args.reranker)
    )
    index_s = time.perf_counter() - t0

    base = RetrievalConfig()
    configs = {
        "BM25 only": base.with_(mode="sparse"),
        f"Dense only ({args.embedder})": base.with_(mode="dense"),
        "Hybrid, RRF": base.with_(fusion="rrf"),
        **{f"Hybrid, convex a={a}": base.with_(fusion="convex", alpha=a) for a in (0.3, 0.5, 0.7)},
    }
    if retriever.reranker is not None:
        for d in (int(x) for x in args.rerank_depths.split(",")):
            configs[f"Hybrid RRF + rerank top {d}"] = base.with_(rerank_depth=d)

    results = [run_benchmark(retriever, queries, qrels, cfg, name) for name, cfg in configs.items()]

    print(
        f"SciFact {args.split}: {len(docs)} docs, {len(queries)} queries. "
        f"Index build {index_s:.1f}s. Embedder={args.embedder}, reranker={args.reranker}.\n"
    )
    print(TABLE_HEADER)
    for r in results:
        print(r.row())

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"scifact-{args.split}-{args.embedder}-{args.reranker}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": f"scifact-{args.split}",
                "n_docs": len(docs),
                "n_queries": len(queries),
                "embedder": args.embedder,
                "reranker": args.reranker,
                "index_build_s": index_s,
                "machine": platform.platform(),
                "python": platform.python_version(),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
