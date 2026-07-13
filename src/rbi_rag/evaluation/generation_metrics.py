from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median, stdev

from ..config import RAGConfig, file_sha256
from ..schemas import EvaluationQuestion
from .reporting import atomic_write_json
from .retry import measure_with_retry


def summarize_generation_rows(rows: list[dict]) -> dict:
    names = sorted({name for row in rows for name in row.get("metrics", {})})
    summary = {}
    for name in names:
        values = [row["metrics"][name] for row in rows if name in row.get("metrics", {})]
        scores = [float(value["score"]) for value in values if value.get("success")]
        total = len(values)
        summary[name] = {
            "mean": mean(scores) if scores else None,
            "median": median(scores) if scores else None,
            "standard_deviation": stdev(scores) if len(scores) > 1 else None,
            "successful_cases": len(scores), "failed_cases": total - len(scores),
            "total_cases": total, "coverage_percentage": 100 * len(scores) / total if total else 0.0,
        }
    return summary


def _payload(rows, config):
    digest = file_sha256(config.pdf_path)
    return {
        "schema_version": 2, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"{config.report_period} only", "retrieval_strategy": "hybrid_reranked",
        "pdf_sha256": digest, "config_fingerprint": config.fingerprint(digest),
        "config": config.public_dict(), "summary": summarize_generation_rows(rows), "rows": rows,
    }


def run_generation_evaluation(
    *, items: list[EvaluationQuestion], retrieve, generate, metric_factories,
    checkpoint_path: Path, config: RAGConfig, max_attempts: int | None = None,
    rerun_successful: bool = False,
):
    fingerprint = config.fingerprint(file_sha256(config.pdf_path))
    rows_by_id: dict[str, dict] = {}
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("config_fingerprint") != fingerprint:
            raise ValueError("checkpoint configuration does not match this run")
        rows_by_id = {row["question_id"]: row for row in saved.get("rows", [])}
    from deepeval.test_case import LLMTestCase

    for item in items:
        row = rows_by_id.get(item.question_id)
        if row is None:
            started = datetime.now(timezone.utc)
            retrieved = retrieve(item.question)
            documents = [value[0] if isinstance(value, tuple) else value for value in retrieved]
            generated = generate(item.question, documents)
            row = {
                "question_id": item.question_id, "question": item.question,
                "expected_answer": item.expected_answer, "generated_answer": generated["answer"],
                "retrieved_context": [doc.page_content for doc in documents],
                "retrieved_chunk_ids": [doc.metadata["chunk_id"] for doc in documents],
                "retrieved_pages": [doc.metadata["page_number"] for doc in documents],
                "source_excerpts": [doc.page_content[:500] for doc in documents],
                "llm_model": config.generator_model, "judge_model": config.judge_model,
                "prompt_version": config.prompt_version,
                "generation_started_at_utc": started.isoformat(),
                "generation_latency_ms": generated.get("latency_ms"), "metrics": {},
            }
            rows_by_id[item.question_id] = row
        test_case = LLMTestCase(
            input=item.question, actual_output=row["generated_answer"],
            expected_output=item.expected_answer, retrieval_context=row["retrieved_context"],
        )
        for name, factory in metric_factories.items():
            prior = row["metrics"].get(name)
            if prior and prior.get("success") and not rerun_successful:
                continue
            result = measure_with_retry(
                factory, test_case, max_attempts=max_attempts or config.max_retries,
                base_delay_seconds=config.retry_base_delay_seconds,
            )
            row["metrics"][name] = asdict(result)
            atomic_write_json(checkpoint_path, _payload(list(rows_by_id.values()), config))
        atomic_write_json(checkpoint_path, _payload(list(rows_by_id.values()), config))
    return _payload(list(rows_by_id.values()), config)

