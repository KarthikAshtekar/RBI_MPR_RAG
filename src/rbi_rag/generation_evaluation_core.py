from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .artifact_io import now_iso
from .security import contains_key_material as actual_key_values_serialized


DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.0

METRIC_NAMES = [
    "factual_correctness",
    "faithfulness_to_context",
    "contextual_relevancy",
    "contextual_recall",
    "citation_correctness",
    "citation_completeness",
    "temporal_attribution_correctness",
    "comparative_correctness",
    "abstention_correctness",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "by",
    "as", "at", "was", "were", "is", "are", "be", "been", "it", "that", "this",
    "from", "per", "cent", "percent", "report", "mpr", "rbi",
}


def safe_error_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def parse_citations(answer: str, context_item: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {block["chunk_id"]: block for block in context_item.get("context_blocks", [])}
    found_ids = []
    for chunk_id in re.findall(r"rbi_mpr_\d{4}_\d{2}_p\d{3}_c\d{3}", answer or ""):
        if chunk_id not in found_ids:
            found_ids.append(chunk_id)
    citations = []
    for chunk_id in found_ids:
        block = allowed.get(chunk_id)
        if block is None:
            citations.append({"chunk_id": chunk_id, "valid_supplied_chunk": False})
        else:
            citations.append({
                "chunk_id": chunk_id,
                "report_id": block["report_id"],
                "report_period": block["report_period"],
                "page": block["page_number"],
                "valid_supplied_chunk": True,
            })
    return citations


def invalid_citations(row: dict[str, Any], context_item: dict[str, Any]) -> list[dict[str, Any]]:
    supplied = set(context_item.get("selected_chunk_ids", []))
    invalid = []
    for citation in row.get("citations", []):
        chunk_id = citation.get("chunk_id")
        if chunk_id not in supplied:
            invalid.append(citation)
    return invalid


class GroqGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        from langchain_groq import ChatGroq

        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGroq(model=model_name, temperature=temperature)

    def invoke(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt).content)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z0-9.]+", (text or "").lower())
        if token and token not in STOPWORDS and len(token) > 1
    }


def _contains_abstention(answer: str) -> bool:
    answer_l = (answer or "").lower()
    return (
        "insufficient" in answer_l
        or "could not find" in answer_l
        or "not supplied" in answer_l
        or "cannot be determined" in answer_l
        or "cannot determine" in answer_l
        or "not enough evidence" in answer_l
    )


def _score_record(
    score: float | None,
    *,
    success: bool = True,
    applicable: bool = True,
    method: str = "deterministic_heuristic",
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "score": score,
        "success": success,
        "applicable": applicable,
        "method": method,
        "error_type": error_type,
        "error_message": error_message,
    }


def evaluate_generation_rows(
    generation_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    retrieval_by_id = {row["question_id"]: row for row in retrieval_rows}
    context_by_id = {row["question_id"]: row for row in contexts}
    eval_rows = []
    failures = []
    for row in generation_rows:
        retrieval = retrieval_by_id.get(row["question_id"], {})
        context = context_by_id.get(row["question_id"], {})
        supplied_chunks = {
            block["chunk_id"]: block
            for block in context.get("context_blocks", [])
        }
        cited_chunks = [citation.get("chunk_id") for citation in row.get("citations", [])]
        answer = row.get("generated_answer") or ""
        expected = row.get("expected_answer") or ""
        answered = row.get("generation_success") and not _contains_abstention(answer)
        metrics: dict[str, dict[str, Any]] = {}
        if not row.get("generation_success"):
            failures.append({
                "question_id": row["question_id"],
                "stage": "generation",
                "error_type": row.get("generation_error_type"),
                "error_message": row.get("generation_error_message"),
            })
            for name in METRIC_NAMES:
                metrics[name] = _score_record(
                    None,
                    success=False,
                    error_type="generation_failed",
                    error_message=row.get("generation_error_message"),
                )
        else:
            expected_tokens = _tokens(expected)
            answer_tokens = _tokens(answer)
            if expected_tokens and answered:
                metrics["factual_correctness"] = _score_record(len(expected_tokens & answer_tokens) / len(expected_tokens))
            elif _contains_abstention(answer):
                metrics["factual_correctness"] = _score_record(None, applicable=False)
            else:
                metrics["factual_correctness"] = _score_record(None, success=False, error_type="missing_expected_answer")
            context_tokens = _tokens(context.get("source_labelled_context", ""))
            claim_tokens = answer_tokens - _tokens("Answer Citations SOURCE page chunk report period")
            metrics["faithfulness_to_context"] = _score_record(
                1.0 if _contains_abstention(answer) else (len(claim_tokens & context_tokens) / len(claim_tokens) if claim_tokens else 0.0)
            )
            metrics["contextual_relevancy"] = _score_record(float(retrieval.get("evidence_recall") or 0.0))
            metrics["contextual_recall"] = _score_record(float(bool(retrieval.get("complete_evidence_recall"))))
            citation_subset = bool(cited_chunks) and all(chunk_id in supplied_chunks for chunk_id in cited_chunks)
            metrics["citation_correctness"] = _score_record(float(citation_subset))
            required = set(row.get("required_report_ids") or [])
            cited_reports = {
                supplied_chunks[chunk_id]["report_id"]
                for chunk_id in cited_chunks
                if chunk_id in supplied_chunks
            }
            if required and answered and row.get("query_type") != "unsupported_period":
                metrics["citation_completeness"] = _score_record(len(required & cited_reports) / len(required))
            else:
                metrics["citation_completeness"] = _score_record(None, applicable=False)
            temporal_ok = citation_subset and cited_reports <= required if required else citation_subset
            metrics["temporal_attribution_correctness"] = _score_record(float(temporal_ok))
            if row.get("query_type") in {"pairwise_comparison", "trend_all_reports"}:
                periods = {
                    block["report_id"]: block["report_period"].lower()
                    for block in context.get("context_blocks", [])
                }
                mentioned = sum(1 for report_id in required if periods.get(report_id, "").lower() in answer.lower())
                mention_score = mentioned / len(required) if required else 0.0
                citation_score = metrics["citation_completeness"]["score"] or 0.0
                metrics["comparative_correctness"] = _score_record(min(mention_score, citation_score))
            else:
                metrics["comparative_correctness"] = _score_record(None, applicable=False)
            retrieval_complete = bool(retrieval.get("complete_evidence_recall"))
            abstention_expected = not retrieval_complete
            abstention_actual = _contains_abstention(answer)
            metrics["abstention_correctness"] = _score_record(float(abstention_actual == abstention_expected))
        eval_rows.append({
            "question_id": row["question_id"],
            "split": row.get("split"),
            "query_type": row.get("query_type"),
            "required_report_ids": row.get("required_report_ids"),
            "generation_success": row.get("generation_success"),
            "metrics": metrics,
        })
    coverage = summarise_metric_coverage(eval_rows)
    summary = summarise_eval_metrics(eval_rows)
    return eval_rows, summary, coverage, failures


def summarise_metric_coverage(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for name in METRIC_NAMES:
        values = [row["metrics"].get(name, {}) for row in eval_rows]
        applicable = [item for item in values if item.get("applicable", True)]
        successful = [item for item in applicable if item.get("success") and item.get("score") is not None]
        failed = [item for item in applicable if not item.get("success")]
        coverage[name] = {
            "total_rows": len(eval_rows),
            "applicable_rows": len(applicable),
            "successful_evaluations": len(successful),
            "failed_evaluations": len(failed),
            "not_applicable": len(values) - len(applicable),
            "coverage": len(successful) / len(applicable) if applicable else None,
        }
    return coverage


def summarise_eval_metrics(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for name in METRIC_NAMES:
        scores = [
            row["metrics"][name]["score"]
            for row in eval_rows
            if row["metrics"].get(name, {}).get("success") and row["metrics"][name].get("score") is not None
        ]
        metrics[name] = {
            "mean_score": sum(scores) / len(scores) if scores else None,
            "successful_count": len(scores),
        }
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "row_count": len(eval_rows),
        "metrics": metrics,
    }
