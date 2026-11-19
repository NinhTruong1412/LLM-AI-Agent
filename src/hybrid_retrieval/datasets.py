"""Benchmark datasets.

SciFact (Wadden et al., 2020): 5,183 scientific abstracts and expert-written
claims, each labelled with the abstracts that support or refute it. It is small
enough to run on a laptop, and its vocabulary mismatch between claims and
abstracts makes it a fair test of sparse versus dense retrieval.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

from hybrid_retrieval.types import Document

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _ensure_scifact(data_dir: Path) -> Path:
    root = data_dir / "scifact" / "data"
    if (root / "corpus.jsonl").exists():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "data.tar.gz"
    urllib.request.urlretrieve(SCIFACT_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root.parent, filter="data")
    archive.unlink()
    return root


def load_scifact(
    split: str = "dev", data_dir: Path = DATA_DIR
) -> tuple[list[Document], dict[str, str], dict[str, set[str]]]:
    """Return (corpus, queries, qrels). Claims without cited abstracts are dropped."""
    root = _ensure_scifact(data_dir)
    docs = []
    with (root / "corpus.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            docs.append(
                Document(
                    id=str(r["doc_id"]),
                    text=r["title"] + "\n" + " ".join(r["abstract"]),
                    metadata={"structured": bool(r.get("structured", False))},
                )
            )
    queries, qrels = {}, {}
    with (root / f"claims_{split}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            cited = {str(d) for d in r.get("cited_doc_ids", [])}
            if cited:
                queries[str(r["id"])] = r["claim"]
                qrels[str(r["id"])] = cited
    return docs, queries, qrels
