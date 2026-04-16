from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LayoutBlockRef:
    """Reference to a layout block from DotsOCR structured parsing."""

    page_number: int
    category: str | None = None
    category_raw: str = ""
    text: str | None = None
    bbox: list[float] | None = None
    order_in_page: int = 0


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
    answers: dict[str, str] | None = None   # MCQ choices  ← NEW
    solution: str | None = None             # MCQ correct key ← NEW
    metadata: dict[str, Any] = field(default_factory=dict)
