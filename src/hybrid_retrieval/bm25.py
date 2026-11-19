"""BM25 (Okapi) implemented from scratch on a sparse matrix.

At index time each (doc, term) weight is precomputed:

    w(d, t) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

so a query is just a sparse column gather plus a sum, which keeps query
latency in the low milliseconds for corpora of tens of thousands of documents.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from hybrid_retrieval.text import tokenize
from hybrid_retrieval.topk import top_k


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self._weights: sparse.csc_matrix | None = None

    def fit(self, texts: Sequence[str]) -> BM25Index:
        rows, cols, tfs = [], [], []
        doc_lens = np.zeros(len(texts), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_lens[i] = sum(counts.values())
            for term, tf in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(tf)

        n_docs, n_terms = len(texts), len(self.vocab)
        tf = np.asarray(tfs, dtype=np.float32)
        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)

        df = np.bincount(cols_a, minlength=n_terms).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avgdl = float(doc_lens.mean()) if n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * doc_lens[rows_a] / max(avgdl, 1e-9))
        w = idf[cols_a] * tf * (self.k1 + 1) / (tf + norm)

        self._weights = sparse.csc_matrix((w, (rows_a, cols_a)), shape=(n_docs, n_terms))
        return self

    def scores(self, query: str) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("BM25Index.fit() must be called first")
        q = Counter(t for t in tokenize(query) if t in self.vocab)
        if not q:
            return np.zeros(self._weights.shape[0], dtype=np.float32)
        cols = np.fromiter((self.vocab[t] for t in q), dtype=np.int64)
        qtf = np.fromiter(q.values(), dtype=np.float32)
        return np.asarray(self._weights[:, cols] @ qtf).ravel()

    def search(
        self, query: str, k: int, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        idx, sc = top_k(s, k, mask)
        keep = sc > 0  # documents with no query term are not sparse matches
        return idx[keep], sc[keep]
