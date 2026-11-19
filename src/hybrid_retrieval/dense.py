"""Dense retrieval: pluggable embedders plus an exact cosine-similarity index.

* ``LSAEmbedder`` is a TF-IDF + truncated SVD model fitted on the corpus. It needs
  no downloads, so tests and CI run anywhere, and it captures some synonymy BM25 misses.
* ``SentenceTransformerEmbedder`` wraps any pretrained bi-encoder (default
  ``BAAI/bge-small-en-v1.5``). Install with ``pip install -e ".[models]"``.

The index is brute-force matrix multiplication, which is exact and fast enough
below roughly a million vectors; an ANN backend can slot in behind ``search``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from hybrid_retrieval.topk import top_k


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


class LSAEmbedder:
    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = f"lsa-{dim}"
        self._tfidf = TfidfVectorizer(
            sublinear_tf=True, stop_words="english", min_df=2, ngram_range=(1, 2)
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=seed)

    def fit(self, texts: Sequence[str]) -> None:
        x = self._tfidf.fit_transform(texts)
        n_components = min(self._svd.n_components, x.shape[1] - 1, x.shape[0] - 1)
        self._svd.n_components = max(n_components, 1)
        self._svd.fit(x)

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        return _normalize(self._svd.transform(self._tfidf.transform(texts)))


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    def fit(self, texts: Sequence[str]) -> None:  # pretrained: nothing to fit
        return None

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if is_query and self._query_prefix:
            texts = [self._query_prefix + t for t in texts]
        emb = self._model.encode(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True
        )
        return np.asarray(emb, dtype=np.float32)


class DenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._vectors: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> DenseIndex:
        self.embedder.fit(texts)
        self._vectors = self.embedder.encode(texts)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self._vectors is None:
            raise RuntimeError("DenseIndex.fit() must be called first")
        q = self.embedder.encode([query], is_query=True)[0]
        return self._vectors @ q

    def search(
        self, query: str, k: int, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return top_k(self.scores(query), k, mask)
