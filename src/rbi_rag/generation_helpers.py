from __future__ import annotations

from typing import Any


def mean_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float, bool))]
    return sum(numeric) / len(numeric) if numeric else None


def source_label(block: dict[str, Any]) -> str:
    return f"[SOURCE: {block['report_period']} MPR | page {block['page_number']} | chunk {block['chunk_id']}]"
