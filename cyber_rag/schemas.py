from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkReference:
    chunk_id: str
    source: str
    title: str | None = None
    page: int | None = None
    score: float | None = None
    excerpt: str = ""


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[ChunkReference] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationExample:
    question: str
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
