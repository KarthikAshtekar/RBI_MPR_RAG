from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

from ..schemas import MetricEvaluationResult


RETRYABLE_MARKERS = ("429", "rate limit", "timeout", "tempor", "json", "connection")


def measure_with_retry(
    metric_factory: Callable[[], object], test_case, *, max_attempts: int = 5,
    base_delay_seconds: float = 20.0, sleep: Callable[[float], None] = time.sleep,
) -> MetricEvaluationResult:
    errors: list[str] = []
    error_type = None
    for attempt in range(1, max_attempts + 1):
        metric = metric_factory()
        try:
            metric.measure(test_case)
            score = float(metric.score)
            if not math.isfinite(score):
                raise ValueError(f"non-finite metric score: {score}")
            return MetricEvaluationResult(
                True, score, getattr(metric, "reason", None), attempt, None, None, "success"
            )
        except Exception as exc:
            error_type = type(exc).__name__
            message = f"{error_type}: {exc}"
            errors.append(message)
            retryable = any(marker in message.lower() for marker in RETRYABLE_MARKERS)
            if attempt >= max_attempts or not retryable:
                return MetricEvaluationResult(
                    False, None, None, attempt, error_type, " | ".join(errors), "failed"
                )
            jitter = random.uniform(0.0, max(0.01, base_delay_seconds * 0.1))
            sleep(base_delay_seconds * attempt + jitter)
    raise AssertionError("unreachable")

