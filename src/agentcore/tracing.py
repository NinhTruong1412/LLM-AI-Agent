"""Minimal, dependency-free tracing.

Every project records per-stage timings through a ``Tracer`` so that latency
numbers in READMEs and benchmarks come from the same code path as production.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    start: float
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects spans in memory and optionally appends them to a JSONL file."""

    def __init__(self, sink: str | Path | None = None) -> None:
        self.spans: list[Span] = []
        self._sink = Path(sink) if sink else None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        s = Span(name=name, start=time.time(), attributes=dict(attributes))
        t0 = time.perf_counter()
        try:
            yield s
        finally:
            s.duration_ms = (time.perf_counter() - t0) * 1000
            self.spans.append(s)
            if self._sink:
                with self._sink.open("a") as f:
                    f.write(json.dumps(asdict(s)) + "\n")

    def durations(self, name: str) -> list[float]:
        return [s.duration_ms for s in self.spans if s.name == name]

    def reset(self) -> None:
        self.spans.clear()
