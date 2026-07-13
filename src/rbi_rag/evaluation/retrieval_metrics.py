from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import RAGConfig, file_sha256
from ..schemas import EvaluationQuestion


def load_evaluation_items(path: Path) -> list[EvaluationQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["questions"] if isinstance(payload, dict) else payload
    items = []
    for row in rows:
        pages = row.get("accepted_pages", row.get("ground_truth_pages"))
        items.append(EvaluationQuestion(
            row["question_id"], row["question"],
            row.get("expected_answer", row.get("expected_answer_summary")),
            tuple(map(int, pages)), row.get("category", "unspecified"),
            row.get("split", "development"), row.get("notes", ""),
        ))
    if not items or len({item.question_id for item in items}) != len(items):
        raise ValueError("evaluation dataset must be non-empty with unique question IDs")
    return items


def _document(item):
    return item[0] if isinstance(item, tuple) else item


def _score(item):
    return item[1] if isinstance(item, tuple) and item[1] is not None else None


def score_retrieval(strategy_name: str, retrieve, items: list[EvaluationQuestion]):
    details = []
    for item in items:
        started = time.perf_counter()
        retrieved = retrieve(item.question)
        latency_ms = (time.perf_counter() - started) * 1000
        documents = [_document(value) for value in retrieved]
        pages = [int(doc.metadata.get("page_number", doc.metadata["page"])) for doc in documents]
        ranks = [i for i, page in enumerate(pages, 1) if page in item.accepted_pages]
        rank = min(ranks) if ranks else None
        details.append({
            "pipeline": strategy_name, "question_id": item.question_id,
            "question": item.question, "accepted_pages": list(item.accepted_pages),
            "retrieved_chunk_ids": [doc.metadata["chunk_id"] for doc in documents],
            "retrieved_pages": pages, "raw_scores": [_score(value) for value in retrieved],
            "rank_first_accepted_page": rank, "hit": rank is not None,
            "reciprocal_rank": 1.0 / rank if rank else 0.0, "latency_ms": latency_ms,
        })
    count = len(details)
    return {
        "strategy": strategy_name, "question_count": count,
        "hit_rate_at_k": sum(row["hit"] for row in details) / count,
        "mrr": sum(row["reciprocal_rank"] for row in details) / count,
        "mean_latency_ms": sum(row["latency_ms"] for row in details) / count,
        "details": details,
    }


def run_retrieval_baseline(suite, items, config: RAGConfig):
    digest = file_sha256(config.pdf_path)
    return {
        "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"{config.report_period} only", "pdf_sha256": digest,
        "config_fingerprint": config.fingerprint(digest), "config": config.public_dict(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "strategies": [score_retrieval(name, fn, items) for name, fn in suite.strategies().items()],
    }

