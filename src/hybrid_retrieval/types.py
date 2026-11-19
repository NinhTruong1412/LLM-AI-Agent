from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]

    @property
    def doc_ids(self) -> list[str]:
        return [h.doc_id for h in self.hits]
