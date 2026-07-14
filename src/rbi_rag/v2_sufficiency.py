from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .env_loading import load_project_dotenv
from .evidence_sufficiency import classify_all, write_sufficiency_classification
from .final_evaluation import file_sha, make_checksum_manifest, write_csv, write_json
from .multi_evaluation import load_jsonl
from .v2_generation_contexts import build_generation_contexts, validate_selected_v2_config, validate_v2_retrieval_input
from .v2_generation_evaluation import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    METRIC_NAMES,
    GroqGenerator,
    _contains_abstention,
    actual_key_values_serialized,
    evaluate_generation_rows,
    expected_case_lookup,
    invalid_citations,
    load_json_file,
    now_iso,
    parse_citations,
    safe_error_message,
    summarise_eval_metrics,
    summarise_metric_coverage,
)


ROOT = Path(".")
V2_SUFF_OUT = Path("reports/v2_sufficiency")
PROMPT_VERSION = "v2_sufficiency_prompt_v1"
CHECKSUM_TARGETS = [
    Path("configs/v2_selected_retrieval.yaml"),
    Path("reports/v2_unstructured_cohere"),
    Path("reports/v2_generation"),
    Path("data/evaluation"),
]


def ensure_dirs(root: Path = ROOT) -> None:
    (root / V2_SUFF_OUT).mkdir(parents=True, exist_ok=True)


def write_pre_sufficiency_checksums(root: Path = ROOT) -> dict[str, Any]:
    ensure_dirs(root)
    payload = make_checksum_manifest(root, CHECKSUM_TARGETS)
    write_json(root / V2_SUFF_OUT / "pre_sufficiency_checksums.json", payload)
    lines = [
        "# Pre-Sufficiency Checksums",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Files captured: {payload['entry_count']}",
        "",
    ]
    if payload.get("missing_targets"):
        lines += ["## Missing targets", ""]
        lines.extend(f"- `{target}`" for target in payload["missing_targets"])
        lines.append("")
    lines += ["## Captured files", ""]
    lines.extend(f"- `{entry['path']}`: `{entry['sha256']}`" for entry in payload["entries"])
    (root / V2_SUFF_OUT / "pre_sufficiency_checksums.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def env_key_available(name: str, root: Path = ROOT) -> bool:
    load_project_dotenv(root)
    return bool(os.getenv(name))


def write_environment_readiness(root: Path = ROOT) -> dict[str, Any]:
    load_project_dotenv(root)
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "groq_api_key_available": env_key_available("GROQ_API_KEY", root),
        "cohere_api_key_available": env_key_available("COHERE_API_KEY", root),
        "unstructured_api_key_available": env_key_available("UNSTRUCTURED_API_KEY", root),
        "generation_may_proceed": bool(os.getenv("GROQ_API_KEY")),
    }
    write_json(root / V2_SUFF_OUT / "environment_readiness.json", payload)
    (root / V2_SUFF_OUT / "environment_readiness.md").write_text(
        "# V2 Sufficiency Environment Readiness\n\n"
        f"Groq key available: {payload['groq_api_key_available']}\n"
        f"Cohere key available: {payload['cohere_api_key_available']}\n"
        f"Unstructured key available: {payload['unstructured_api_key_available']}\n"
        f"Generation may proceed: {payload['generation_may_proceed']}\n",
        encoding="utf-8",
    )
    return payload


def write_input_validation(root: Path, experiment_id: str, config_path: Path) -> dict[str, Any]:
    selected = validate_selected_v2_config(root, config_path, experiment_id)
    retrieval = validate_v2_retrieval_input(root, experiment_id)
    issues = selected.get("issues", []) + retrieval.get("issues", [])
    old_generation_exists = (root / "reports/v2_generation/dev_generation_raw_results.json").exists()
    if not old_generation_exists:
        issues.append("missing_old_v2_generation_results")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "selected_config_validation": selected,
        "retrieval_input_validation": retrieval,
        "old_v2_generation_exists": old_generation_exists,
        "issues": issues,
        "heldout_used": False,
        "retrieval_rerun": False,
    }
    write_json(root / V2_SUFF_OUT / "v2_sufficiency_input_validation.json", payload)
    lines = [
        "# V2 Sufficiency Input Validation",
        "",
        f"Status: {payload['status']}",
        f"Selected experiment: `{selected.get('selected_experiment_id')}`",
        f"Retrieval rows: {retrieval.get('row_count')}",
        f"Old V2 generation exists: {old_generation_exists}",
    ]
    if issues:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in issues)
    (root / V2_SUFF_OUT / "v2_sufficiency_input_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def required_periods(context_item: dict[str, Any]) -> list[str]:
    periods = {
        block["report_id"]: block["report_period"]
        for block in context_item.get("context_blocks", [])
    }
    return [periods.get(report_id, report_id) for report_id in context_item.get("required_report_ids") or []]


def build_sufficiency_prompt(
    question: str,
    context: str,
    query_type: str,
    required_periods_value: list[str],
    sufficiency: dict[str, Any],
) -> str:
    periods = ", ".join(required_periods_value) if required_periods_value else "none"
    status = sufficiency["sufficiency_status"]
    behavior = sufficiency["required_generation_behavior"]
    reasons = ", ".join(sufficiency.get("sufficiency_reasons") or ["none"])
    if status == "sufficient":
        behaviour_instruction = "Answer from the supplied context and cite every factual claim."
    elif status == "partially_sufficient":
        behaviour_instruction = (
            "Answer only the parts supported by the supplied context. Clearly state which parts cannot be determined "
            "from the supplied context. Do not invent missing report comparisons."
        )
    else:
        behaviour_instruction = (
            "Do not answer substantively. State that the supplied context is insufficient. "
            "Mention which required report or evidence is missing if known."
        )
    return f"""You are answering a question for a Temporal multi-document RAG for RBI Monetary Policy Reports.

Use only the supplied source-labelled context. Do not use outside knowledge.
Preserve report-specific attribution. Do not mix April 2025, October 2025, and April 2026 evidence.
Do not invent numbers. Do not infer unsupported policy changes. Do not use generic sentiment language.
Use policy stance, inflation outlook, growth projection, risk assessment, or policy stance and narrative evolution as appropriate.

Required reports: {periods}
Query type: {query_type}
Evidence sufficiency status: {status}
Sufficiency reasons: {reasons}
Required generation behavior: {behavior}

Behaviour instruction:
{behaviour_instruction}

Citation rules:
- Every factual claim must cite supplied chunks.
- Citations must use report period, page, and chunk ID.
- No citation may refer to a chunk not supplied in the context.
- Copy chunk_id values exactly from SOURCE labels.

Output format:
Answer:
...

Citations:
- [report period, page, chunk_id]

Context:
{context}

Question: {question}
"""


def deterministic_abstention(context_item: dict[str, Any], sufficiency: dict[str, Any]) -> str:
    missing = sufficiency.get("missing_required_reports") or []
    reasons = sufficiency.get("sufficiency_reasons") or []
    detail = []
    if missing:
        detail.append("missing required reports: " + ", ".join(missing))
    if reasons:
        detail.append("reasons: " + ", ".join(reasons))
    detail_text = " (" + "; ".join(detail) + ")" if detail else ""
    return (
        "Answer:\n"
        f"The supplied context is insufficient to answer this question safely{detail_text}. "
        "I will not infer unsupported policy stance, inflation outlook, growth projection, risk assessment, or narrative evolution beyond the supplied evidence.\n\n"
        "Citations:\n"
    )


def apply_sufficiency_postprocessing(answer: str, sufficiency: dict[str, Any]) -> str:
    if sufficiency.get("required_generation_behavior") != "answer_with_caveat":
        return answer
    if _contains_abstention(answer):
        return answer
    if answer.lstrip().lower().startswith("answer:"):
        return answer.replace(
            "Answer:",
            "Answer:\nThe supplied context is partially insufficient; some requested evidence cannot be determined from the supplied context. The supported answer is:",
            1,
        )
    return (
        "Answer:\n"
        "The supplied context is partially insufficient; some requested evidence cannot be determined from the supplied context. "
        "The supported answer is:\n"
        f"{answer}"
    )


def run_sufficiency_generation_cases(
    contexts: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    classifications: list[dict[str, Any]],
    *,
    generator: Any,
    model_provider: str = "Groq",
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    checkpoint_path: Path | None = None,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    classification_by_id = {row["question_id"]: row for row in classifications}
    context_by_id = {item["question_id"]: item for item in contexts}
    rows: list[dict[str, Any]] = []
    if checkpoint_path and checkpoint_path.exists():
        existing = load_json_file(checkpoint_path)
        for row in existing:
            context_item = context_by_id.get(row.get("question_id"), {})
            sufficiency = classification_by_id.get(row.get("question_id"), {})
            if row.get("prompt_version") != PROMPT_VERSION:
                continue
            if row.get("generation_success"):
                row["generated_answer"] = apply_sufficiency_postprocessing(row.get("generated_answer") or "", sufficiency)
                row["citations"] = parse_citations(row["generated_answer"], context_item)
            if row.get("generation_success") and invalid_citations(row, context_item):
                continue
            rows.append(row)
        if checkpoint_path:
            write_json(checkpoint_path, rows)
    completed = {row["question_id"] for row in rows}
    for item in contexts:
        qid = item["question_id"]
        if qid in completed:
            continue
        case = cases.get(qid, {})
        sufficiency = classification_by_id[qid]
        prompt = build_sufficiency_prompt(
            item.get("original_query") or case.get("question", ""),
            item.get("source_labelled_context") or "",
            str(item.get("query_type")),
            required_periods(item),
            sufficiency,
        )
        attempts = 0
        retry_warnings: list[dict[str, Any]] = []
        error_type = None
        error_message = None
        started = time.perf_counter()
        if sufficiency["required_generation_behavior"] == "abstain":
            generated_answer = deterministic_abstention(item, sufficiency)
            success = True
            model_provider_used = "deterministic_sufficiency_gate"
            attempts = 0
        else:
            generated_answer = ""
            success = False
            model_provider_used = model_provider
            for attempt in range(1, max_retries + 1):
                attempts = attempt
                try:
                    generated_answer = apply_sufficiency_postprocessing(str(generator.invoke(prompt)), sufficiency)
                    provisional = {"citations": parse_citations(generated_answer, item), "generation_success": True}
                    invalid = invalid_citations(provisional, item)
                    if invalid and attempt < max_retries:
                        retry_warnings.append({
                            "attempt": attempt,
                            "type": "InvalidCitation",
                            "invalid_chunk_ids": [citation.get("chunk_id") for citation in invalid],
                        })
                        prompt += "\n\nRegenerate: cite only exact chunk IDs that appear in the supplied SOURCE labels."
                        continue
                    success = True
                    error_type = None
                    error_message = None
                    break
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_message = safe_error_message(exc)
                    retry_warnings.append({"attempt": attempt, "type": error_type})
                    if attempt < max_retries:
                        time.sleep(min(2**attempt, 8))
        latency = (time.perf_counter() - started) * 1000
        citations = parse_citations(generated_answer, item)
        row = {
            "question_id": qid,
            "split": item.get("split"),
            "query_type": item.get("query_type"),
            "required_report_ids": item.get("required_report_ids"),
            "original_query": item.get("original_query") or case.get("question"),
            "normalised_query": item.get("normalised_query"),
            "retrieval_experiment_id": item.get("retrieval_experiment_id"),
            "retrieval_config_checksum": item.get("retrieval_config_checksum"),
            "selected_chunk_ids": item.get("selected_chunk_ids"),
            "selected_pages": item.get("selected_pages"),
            "sufficiency_status": sufficiency["sufficiency_status"],
            "sufficiency_reasons": sufficiency["sufficiency_reasons"],
            "required_generation_behavior": sufficiency["required_generation_behavior"],
            "source_labelled_context": item.get("source_labelled_context"),
            "prompt_version": PROMPT_VERSION,
            "model_provider": model_provider_used,
            "model_name": model_name,
            "temperature": temperature,
            "generated_answer": generated_answer,
            "citations": citations,
            "generation_latency_ms": latency,
            "generation_success": success,
            "generation_error_type": error_type,
            "generation_error_message": error_message,
            "generation_attempts": attempts,
            "generation_retry_warnings": retry_warnings,
            "expected_answer": case.get("expected_answer"),
            "category": case.get("category"),
            "table_or_numeric_question": sufficiency.get("table_or_numeric_question"),
        }
        rows.append(row)
        if checkpoint_path:
            write_json(checkpoint_path, rows)
    return rows


def write_generation_outputs(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = root / V2_SUFF_OUT
    write_json(out / "dev_generation_sufficiency_raw_results.json", rows)
    write_csv(out / "dev_generation_sufficiency_results.csv", rows)
    successes = [row for row in rows if row.get("generation_success")]
    latencies = [float(row["generation_latency_ms"]) for row in rows if isinstance(row.get("generation_latency_ms"), (int, float))]
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "split": "dev",
        "row_count": len(rows),
        "generation_success_count": len(successes),
        "generation_failure_count": len(rows) - len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "model_name": rows[0].get("model_name") if rows else None,
        "temperature": rows[0].get("temperature") if rows else None,
        "prompt_version": PROMPT_VERSION,
        "mean_generation_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_generation_latency_ms": statistics.median(latencies) if latencies else None,
        "behavior_counts": dict(Counter(row.get("required_generation_behavior") for row in rows)),
        "sufficiency_status_counts": dict(Counter(row.get("sufficiency_status") for row in rows)),
        "hard_gate_abstention_count": sum(1 for row in rows if row.get("model_provider") == "deterministic_sufficiency_gate"),
    }
    write_json(out / "dev_generation_sufficiency_summary.json", summary)
    lines = [
        "# V2 Sufficiency-Gated Development Generation Summary",
        "",
        f"Rows: {summary['row_count']}",
        f"Successful generations: {summary['generation_success_count']}",
        f"Failed generations: {summary['generation_failure_count']}",
        f"Prompt version: `{PROMPT_VERSION}`",
        f"Hard-gated abstentions: {summary['hard_gate_abstention_count']}",
        f"Median latency ms: {summary['median_generation_latency_ms']}",
    ]
    (out / "dev_generation_sufficiency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_eval_outputs(root: Path, eval_rows: list[dict[str, Any]], summary: dict[str, Any], coverage: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    out = root / V2_SUFF_OUT
    write_json(out / "dev_sufficiency_eval_raw_results.json", eval_rows)
    write_json(out / "dev_sufficiency_eval_summary.json", summary)
    write_json(out / "dev_sufficiency_metric_coverage.json", coverage)
    write_csv(out / "dev_sufficiency_generation_failures.csv", failures)
    lines = ["# V2 Sufficiency Evaluation Summary", "", "| Metric | Mean score | Successful count |", "|---|---:|---:|"]
    for name, item in summary["metrics"].items():
        lines.append(f"| {name} | {item['mean_score']} | {item['successful_count']} |")
    (out / "dev_sufficiency_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(eval_row: dict[str, Any], name: str) -> float | None:
    metric = (eval_row.get("metrics") or {}).get(name) or {}
    return metric.get("score") if metric.get("success") and metric.get("score") is not None else None


def _mean(values: list[Any]) -> float | None:
    vals = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(vals) / len(vals) if vals else None


def comparison_outputs(
    retrieval_rows: list[dict[str, Any]],
    old_generation_rows: list[dict[str, Any]],
    old_eval_rows: list[dict[str, Any]],
    new_generation_rows: list[dict[str, Any]],
    new_eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_gen = {row["question_id"]: row for row in old_generation_rows}
    new_gen = {row["question_id"]: row for row in new_generation_rows}
    old_eval = {row["question_id"]: row for row in old_eval_rows}
    new_eval = {row["question_id"]: row for row in new_eval_rows}
    rows = []
    for retrieval in retrieval_rows:
        qid = retrieval["question_id"]
        old_answered = old_gen.get(qid, {}).get("generation_success") and not _contains_abstention(old_gen.get(qid, {}).get("generated_answer", ""))
        new_answered = new_gen.get(qid, {}).get("generation_success") and not _contains_abstention(new_gen.get(qid, {}).get("generated_answer", ""))
        row = {
            "question_id": qid,
            "query_type": retrieval.get("query_type"),
            "retrieval_complete_evidence_recall": retrieval.get("complete_evidence_recall"),
            "retrieval_evidence_recall": retrieval.get("evidence_recall"),
            "old_abstained": bool(old_gen.get(qid, {}).get("generation_success")) and not old_answered,
            "new_abstained": bool(new_gen.get(qid, {}).get("generation_success")) and not new_answered,
            "old_answered": bool(old_answered),
            "new_answered": bool(new_answered),
            "sufficiency_status": new_gen.get(qid, {}).get("sufficiency_status"),
            "required_generation_behavior": new_gen.get(qid, {}).get("required_generation_behavior"),
        }
        for metric_name in METRIC_NAMES:
            old_score = _metric(old_eval.get(qid, {}), metric_name)
            new_score = _metric(new_eval.get(qid, {}), metric_name)
            row[f"old_{metric_name}"] = old_score
            row[f"new_{metric_name}"] = new_score
            row[f"delta_{metric_name}"] = new_score - old_score if isinstance(old_score, (int, float)) and isinstance(new_score, (int, float)) else None
        rows.append(row)
    incomplete = [row for row in rows if row["retrieval_complete_evidence_recall"] is not True]
    complete = [row for row in rows if row["retrieval_complete_evidence_recall"] is True]
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "row_count": len(rows),
        "metric_deltas": {
            metric: {
                "old_mean": _mean([row[f"old_{metric}"] for row in rows]),
                "new_mean": _mean([row[f"new_{metric}"] for row in rows]),
                "delta": (
                    (_mean([row[f"new_{metric}"] for row in rows]) or 0) - (_mean([row[f"old_{metric}"] for row in rows]) or 0)
                    if _mean([row[f"old_{metric}"] for row in rows]) is not None and _mean([row[f"new_{metric}"] for row in rows]) is not None else None
                ),
            }
            for metric in METRIC_NAMES
        },
        "old_incomplete_retrieval_abstention_rate": _mean([float(row["old_abstained"]) for row in incomplete]),
        "new_incomplete_retrieval_abstention_rate": _mean([float(row["new_abstained"]) for row in incomplete]),
        "old_incomplete_unsupported_answer_rate": _mean([float(row["old_answered"]) for row in incomplete]),
        "new_incomplete_unsupported_answer_rate": _mean([float(row["new_answered"]) for row in incomplete]),
        "old_complete_retrieval_factual": _mean([row["old_factual_correctness"] for row in complete]),
        "new_complete_retrieval_factual": _mean([row["new_factual_correctness"] for row in complete]),
        "old_incomplete_retrieval_factual": _mean([row["old_factual_correctness"] for row in incomplete]),
        "new_incomplete_retrieval_factual": _mean([row["new_factual_correctness"] for row in incomplete]),
    }
    return rows, summary


def write_comparison_outputs(root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out = root / V2_SUFF_OUT
    write_csv(out / "old_vs_sufficiency_generation_comparison.csv", rows)
    write_json(out / "old_vs_sufficiency_generation_comparison.json", {"summary": summary, "rows": rows})
    lines = [
        "# Old V2 vs Sufficiency-Gated Generation Comparison",
        "",
        f"Rows: {summary['row_count']}",
        f"Old incomplete-retrieval abstention rate: {summary['old_incomplete_retrieval_abstention_rate']}",
        f"New incomplete-retrieval abstention rate: {summary['new_incomplete_retrieval_abstention_rate']}",
        f"Old incomplete unsupported-answer rate: {summary['old_incomplete_unsupported_answer_rate']}",
        f"New incomplete unsupported-answer rate: {summary['new_incomplete_unsupported_answer_rate']}",
        f"Old complete-retrieval factual correctness: {summary['old_complete_retrieval_factual']}",
        f"New complete-retrieval factual correctness: {summary['new_complete_retrieval_factual']}",
        "",
        "## Metric deltas",
        "",
        "| Metric | Old mean | New mean | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in summary["metric_deltas"].items():
        lines.append(f"| {metric} | {values['old_mean']} | {values['new_mean']} | {values['delta']} |")
    (out / "old_vs_sufficiency_generation_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status(root: Path, generation_summary: dict[str, Any]) -> dict[str, Any]:
    if generation_summary.get("generation_success_count") == generation_summary.get("row_count"):
        status = "dev_sufficiency_generation_complete"
    elif generation_summary.get("generation_success_count", 0) > 0:
        status = "dev_sufficiency_generation_partial"
    else:
        status = "dev_sufficiency_generation_failed"
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": status,
        "generation_summary": generation_summary,
        "heldout_generation_run": False,
        "heldout_retrieval_run": False,
    }
    write_json(root / V2_SUFF_OUT / "v2_sufficiency_status.json", payload)
    (root / V2_SUFF_OUT / "v2_sufficiency_status.md").write_text(
        "# V2 Sufficiency Status\n\n"
        f"Status: `{status}`\n\n"
        "Held-out retrieval/generation was not run.\n",
        encoding="utf-8",
    )
    return payload


def validate_v2_sufficiency_artifacts(root: Path = ROOT) -> dict[str, Any]:
    load_project_dotenv(root)
    issues: list[str] = []
    out = root / V2_SUFF_OUT
    pre_path = out / "pre_sufficiency_checksums.json"
    if not pre_path.exists():
        issues.append("missing_pre_sufficiency_checksums")
        pre_entries = {}
    else:
        pre = load_json_file(pre_path)
        pre_entries = {entry["path"]: entry["sha256"] for entry in pre.get("entries", [])}
    for rel, sha in pre_entries.items():
        path = root / rel
        if not path.exists() or file_sha(path) != sha:
            issues.append(f"checksum_changed:{rel}")
    required = [
        "dev_sufficiency_classification.json",
        "dev_generation_sufficiency_raw_results.json",
        "dev_sufficiency_eval_raw_results.json",
        "old_vs_sufficiency_generation_comparison.json",
        "sufficiency_results_for_presentation.md",
        "v2_sufficiency_status.json",
    ]
    for name in required:
        if not (out / name).exists():
            issues.append(f"missing_artifact:{name}")
    rows = load_json_file(out / "dev_generation_sufficiency_raw_results.json") if (out / "dev_generation_sufficiency_raw_results.json").exists() else []
    raw_retrieval_path = root / "reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY/raw_results.json"
    contexts = build_generation_contexts(root, "V2_COHERE_ONLY") if rows and raw_retrieval_path.exists() else []
    context_by_id = {item["question_id"]: item for item in contexts}
    for row in rows:
        qid = row.get("question_id", "unknown")
        if row.get("split") != "dev" or str(qid).startswith("test_"):
            issues.append(f"{qid}:heldout_or_non_dev_case")
        if row.get("prompt_version") != PROMPT_VERSION:
            issues.append(f"{qid}:wrong_prompt_version")
        if row.get("retrieval_experiment_id") != "V2_COHERE_ONLY":
            issues.append(f"{qid}:wrong_retrieval_experiment")
        supplied_chunk_ids = set(context_by_id.get(qid, {}).get("selected_chunk_ids", []) or row.get("selected_chunk_ids", []))
        for citation in row.get("citations", []):
            if citation.get("chunk_id") not in supplied_chunk_ids:
                issues.append(f"{qid}:citation_not_in_supplied_context:{citation.get('chunk_id')}")
        if row.get("generation_success") is False and (not row.get("generation_error_type") or row.get("generation_error_message") is None):
            issues.append(f"{qid}:failed_generation_missing_error_fields")
    eval_rows = load_json_file(out / "dev_sufficiency_eval_raw_results.json") if (out / "dev_sufficiency_eval_raw_results.json").exists() else []
    for row in eval_rows:
        for metric_name, metric in (row.get("metrics") or {}).items():
            if not metric.get("success", True) and metric.get("score") is not None:
                issues.append(f"{row.get('question_id')}:failed_metric_has_non_null_score:{metric_name}")
    all_artifacts = {}
    if out.exists():
        for path in out.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv", ".yaml", ".yml", ".log"}:
                all_artifacts[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
    if actual_key_values_serialized(all_artifacts):
        issues.append("api_key_value_serialized")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(set(issues)),
        "issues": sorted(set(issues)),
        "heldout_generation_run": False,
        "heldout_retrieval_run": False,
        "old_v2_generation_artifacts_overwritten": any(issue.startswith("checksum_changed:reports\\v2_generation") for issue in issues),
    }
    write_json(out / "v2_sufficiency_integrity.json", payload)
    lines = ["# V2 Sufficiency Integrity", "", f"Status: {payload['status']}", f"Issues: {payload['issue_count']}"]
    if payload["issues"]:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    (out / "v2_sufficiency_integrity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def generate_v2_sufficiency_report(root: Path = ROOT) -> dict[str, str]:
    out = root / V2_SUFF_OUT
    def load(name: str, default: Any) -> Any:
        path = out / name
        return load_json_file(path) if path.exists() else default
    classification = load("dev_sufficiency_classification.json", [])
    gen_summary = load("dev_generation_sufficiency_summary.json", {})
    eval_summary = load("dev_sufficiency_eval_summary.json", {"metrics": {}})
    comparison = load("old_vs_sufficiency_generation_comparison.json", {"summary": {}, "rows": []})
    integrity = load("v2_sufficiency_integrity.json", {})
    status = load("v2_sufficiency_status.json", {})
    status_counts = Counter(row.get("sufficiency_status") for row in classification)
    behavior_counts = Counter(row.get("required_generation_behavior") for row in classification)
    metrics = eval_summary.get("metrics", {})
    comp = comparison.get("summary", {})
    rows = comparison.get("rows", [])
    safer = [row for row in rows if not row.get("old_abstained") and row.get("new_abstained")][:3]
    strong = [
        row for row in rows
        if row.get("retrieval_complete_evidence_recall") is True
        and isinstance(row.get("new_factual_correctness"), (int, float))
        and row["new_factual_correctness"] >= 0.7
    ][:3]
    presentation = [
        "# Sufficiency Results for Presentation",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        "## Why sufficiency gating was added",
        "",
        "The previous V2 dev generation answered even when retrieval evidence was incomplete, so safety needed to move from retrieval tuning to evidence sufficiency detection.",
        "",
        "## Previous weakness",
        "",
        f"Old incomplete-retrieval abstention rate: {comp.get('old_incomplete_retrieval_abstention_rate')}",
        "",
        "## New behaviour",
        "",
        f"Sufficiency statuses: {dict(status_counts)}",
        f"Required generation behaviours: {dict(behavior_counts)}",
        "",
        "## Metric changes",
        "",
        "| Metric | Old | New | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in comp.get("metric_deltas", {}).items():
        presentation.append(f"| {metric} | {values.get('old_mean')} | {values.get('new_mean')} | {values.get('delta')} |")
    presentation += [
        "",
        "## Examples of safer abstention",
        "",
    ]
    presentation.extend(f"- `{row['question_id']}` changed from answered to abstained." for row in safer)
    presentation += ["", "## Examples where answer quality remained strong", ""]
    presentation.extend(f"- `{row['question_id']}` retained/new factual score {row.get('new_factual_correctness')}." for row in strong)
    presentation += [
        "",
        "## Trade-off",
        "",
        "The gate improves safety by reducing unsupported answers, but answer coverage can drop because insufficient cases abstain.",
        "",
        "## Scientific caveat",
        "",
        "This is development-only. Held-out retrieval and held-out generation were not run because the prior held-out set has already been used.",
        "",
        "The system is not production-ready.",
    ]
    presentation_path = out / "sufficiency_results_for_presentation.md"
    presentation_path.write_text("\n".join(presentation) + "\n", encoding="utf-8")
    report = [
        "# V2 Sufficiency Gate Report",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        f"Status: `{status.get('status')}`",
        f"Integrity: `{integrity.get('status')}`",
        "",
        "## Classifier results",
        "",
        f"Sufficiency statuses: {dict(status_counts)}",
        f"Required generation behaviours: {dict(behavior_counts)}",
        "",
        "## Generation results",
        "",
        f"Rows: {gen_summary.get('row_count')}; successes: {gen_summary.get('generation_success_count')}; failures: {gen_summary.get('generation_failure_count')}.",
        f"Prompt version: `{PROMPT_VERSION}`.",
        "",
        "## Evaluation metrics",
        "",
    ]
    for metric, values in metrics.items():
        report.append(f"- {metric}: mean={values.get('mean_score')}, n={values.get('successful_count')}")
    report += [
        "",
        "## Old vs new comparison",
        "",
        f"Old incomplete-retrieval abstention rate: {comp.get('old_incomplete_retrieval_abstention_rate')}",
        f"New incomplete-retrieval abstention rate: {comp.get('new_incomplete_retrieval_abstention_rate')}",
        f"Old unsupported-answer rate when retrieval incomplete: {comp.get('old_incomplete_unsupported_answer_rate')}",
        f"New unsupported-answer rate when retrieval incomplete: {comp.get('new_incomplete_unsupported_answer_rate')}",
        f"Old complete-retrieval factual correctness: {comp.get('old_complete_retrieval_factual')}",
        f"New complete-retrieval factual correctness: {comp.get('new_complete_retrieval_factual')}",
        "",
        "## Limitations",
        "",
        "- Sufficiency classification is deterministic and label-assisted on development data only.",
        "- Insufficient cases use a hard abstention gate, so answer coverage can drop.",
        "- Metrics remain deterministic heuristics, not human judgement.",
        "- Held-out data was not used.",
        "",
        "## Exact next phase",
        "",
        "Create a fresh V2 evaluation set, run final retrieval plus generation on that fresh set, then build the Streamlit interface.",
    ]
    report_path = out / "v2_sufficiency_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"presentation_summary_path": str(presentation_path), "final_report_path": str(report_path)}


def run_v2_sufficiency_generation(
    root: Path = ROOT,
    *,
    split: str = "dev",
    retrieval_experiment: str = "V2_COHERE_ONLY",
    config_path: Path = Path("configs/v2_selected_retrieval.yaml"),
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    generator: Any | None = None,
) -> dict[str, Any]:
    if split != "dev":
        raise RuntimeError("sufficiency generation only supports development split")
    ensure_dirs(root)
    load_project_dotenv(root)
    write_pre_sufficiency_checksums(root)
    readiness = write_environment_readiness(root)
    validation = write_input_validation(root, retrieval_experiment, config_path)
    if validation["status"] != "passed":
        raise RuntimeError("; ".join(validation["issues"]))
    retrieval_rows = load_json_file(root / "reports/v2_unstructured_cohere/experiments" / retrieval_experiment / "raw_results.json")
    contexts = build_generation_contexts(root, retrieval_experiment)
    cases = expected_case_lookup(root, split)
    classifications = classify_all(retrieval_rows, contexts, cases)
    write_sufficiency_classification(root, V2_SUFF_OUT, classifications)
    if not readiness["generation_may_proceed"]:
        raise RuntimeError("GROQ_API_KEY is unavailable; sufficiency generation not run")
    generator = generator or GroqGenerator(model_name, temperature)
    rows = run_sufficiency_generation_cases(
        contexts,
        cases,
        classifications,
        generator=generator,
        model_name=model_name,
        temperature=temperature,
        checkpoint_path=root / V2_SUFF_OUT / "dev_generation_sufficiency_raw_results.json",
    )
    gen_summary = write_generation_outputs(root, rows)
    eval_rows, eval_summary, coverage, failures = evaluate_generation_rows(rows, retrieval_rows, contexts)
    write_eval_outputs(root, eval_rows, eval_summary, coverage, failures)
    old_gen = load_json_file(root / "reports/v2_generation/dev_generation_raw_results.json")
    old_eval = load_json_file(root / "reports/v2_generation/dev_answer_eval_raw_results.json")
    comp_rows, comp_summary = comparison_outputs(retrieval_rows, old_gen, old_eval, rows, eval_rows)
    write_comparison_outputs(root, comp_rows, comp_summary)
    status = write_status(root, gen_summary)
    reports = generate_v2_sufficiency_report(root)
    integrity = validate_v2_sufficiency_artifacts(root)
    payload = {
        "status": status["status"],
        "generated_rows": len(rows),
        "generation_success_count": gen_summary["generation_success_count"],
        "integrity_status": integrity["status"],
        "heldout_generation_run": False,
        "heldout_retrieval_run": False,
        "presentation_summary_path": reports["presentation_summary_path"],
        "final_report_path": reports["final_report_path"],
    }
    if actual_key_values_serialized(payload):
        raise RuntimeError("sufficiency payload unexpectedly contains key material")
    return payload
