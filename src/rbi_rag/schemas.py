from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_answer: str
    accepted_pages: tuple[int, ...]
    category: str = "unspecified"
    split: str = "development"
    notes: str = ""


@dataclass(frozen=True)
class MetricEvaluationResult:
    success: bool
    score: float | None
    reason: str | None
    attempts: int
    error_type: str | None
    error: str | None
    status: str


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    query_type: str
    report_ids: tuple[str, ...]
    topic: str | None
    comparison_dimension: str | None
    requires_calculation: bool
    confidence: float
    routing_reason: str


@dataclass(frozen=True)
class Citation:
    report_id: str
    report_period: str
    page: int
    chunk_id: str
    excerpt: str

