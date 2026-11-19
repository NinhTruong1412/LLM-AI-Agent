"""Second-stage rerankers.

A cross-encoder reads the query and document together, so it is far more
accurate than either first-stage retriever and far slower. The pipeline only
reranks the top ``rerank_depth`` fused candidates, which is the main
latency/accuracy dial this project measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    name: str

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name)
        self._batch_size = batch_size

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        pairs = [(query, d) for d in docs]
        return np.asarray(self._model.predict(pairs, batch_size=self._batch_size))


class CallableReranker:
    """Wraps any ``(query, docs) -> scores`` function. Used in tests and for custom models."""

    def __init__(self, fn, name: str = "custom") -> None:
        self._fn = fn
        self.name = name

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return np.asarray(self._fn(query, list(docs)), dtype=np.float32)
