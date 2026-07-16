from __future__ import annotations

from pathlib import Path
from typing import Any

from .multi_evaluation import load_jsonl


def expected_case_lookup(root: Path = Path("."), split: str = "dev") -> dict[str, dict[str, Any]]:
    path = root / f"data/evaluation/multi_report_{split}.jsonl"
    return {case["question_id"]: case for case in load_jsonl(path)}
