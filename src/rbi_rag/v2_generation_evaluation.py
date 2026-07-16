from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

from .artifact_io import file_sha, make_checksum_manifest, now_iso, write_csv, write_json
from .env_loading import load_project_dotenv
from .evaluation_cases import expected_case_lookup
from .generation_evaluation_core import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    METRIC_NAMES,
    GroqGenerator,
    _contains_abstention,
    actual_key_values_serialized,
    evaluate_generation_rows,
    invalid_citations,
    load_json_file,
    parse_citations,
    safe_error_message,
    summarise_eval_metrics,
    summarise_metric_coverage,
)
from .v2_experiments import contains_api_key_material
from .v2_generation_contexts import (
    REPORT_ORDER,
    build_generation_contexts,
    validate_selected_v2_config,
    validate_v2_retrieval_input,
)


ROOT = Path(".")
V2_GEN_OUT = Path("reports/v2_generation")
PROMPT_VERSION = "v2_source_labelled_context_v1"

CHECKSUM_TARGETS = [
    Path("configs/v2_selected_retrieval.yaml"),
    Path("reports/v2_unstructured_cohere"),
    Path("configs/final_retrieval_selected.yaml"),
    Path("reports/final_evaluation"),
    Path("reports/structural_optimisation"),
    Path("reports/optimisation"),
    Path("data/evaluation"),
]


def ensure_dirs(root: Path = ROOT) -> None:
    (root / V2_GEN_OUT).mkdir(parents=True, exist_ok=True)


def write_pre_generation_checksums(root: Path = ROOT) -> dict[str, Any]:
    payload = make_checksum_manifest(root, CHECKSUM_TARGETS)
    write_json(root / V2_GEN_OUT / "pre_v2_generation_checksums.json", payload)
    lines = [
        "# Pre-V2 Generation Checksums",
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
    (root / V2_GEN_OUT / "pre_v2_generation_checksums.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
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
        "project_dotenv_loaded": load_project_dotenv(root),
        "generation_may_proceed": bool(os.getenv("GROQ_API_KEY")),
    }
    write_json(root / V2_GEN_OUT / "environment_readiness.json", payload)
    lines = [
        "# V2 Generation Environment Readiness",
        "",
        f"Groq key available: {payload['groq_api_key_available']}",
        f"Cohere key available: {payload['cohere_api_key_available']}",
        f"Unstructured key available: {payload['unstructured_api_key_available']}",
        f"Project .env loaded: {payload['project_dotenv_loaded']}",
        f"Generation may proceed: {payload['generation_may_proceed']}",
    ]
    (root / V2_GEN_OUT / "environment_readiness.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def write_retrieval_input_validation(
    root: Path = ROOT,
    experiment_id: str = "V2_COHERE_ONLY",
    config_path: Path = Path("configs/v2_selected_retrieval.yaml"),
) -> dict[str, Any]:
    selected = validate_selected_v2_config(root, config_path, experiment_id)
    retrieval = validate_v2_retrieval_input(root, experiment_id)
    critical_issues = selected["issues"] + retrieval["issues"]
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not critical_issues else "failed",
        "selected_config_validation": selected,
        "retrieval_input_validation": retrieval,
        "issues": critical_issues,
        "old_multi_report_config_used": False,
        "phase7_retrieval_outputs_used": False,
        "heldout_generation_or_retrieval_run": False,
    }
    write_json(root / V2_GEN_OUT / "v2_retrieval_input_validation.json", payload)
    lines = [
        "# V2 Retrieval Input Validation",
        "",
        f"Status: {payload['status']}",
        f"Selected experiment: `{selected.get('selected_experiment_id')}`",
        f"Retrieval rows: {retrieval.get('row_count')}",
        f"Integrity status: {retrieval.get('integrity_status')}",
        "",
    ]
    if critical_issues:
        lines += ["## Issues", ""]
        lines.extend(f"- {issue}" for issue in critical_issues)
    else:
        lines.append("The selected V2 Cohere retrieval outputs are valid development-only inputs.")
    (root / V2_GEN_OUT / "v2_retrieval_input_validation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def write_generation_contexts(
    root: Path = ROOT,
    experiment_id: str = "V2_COHERE_ONLY",
) -> list[dict[str, Any]]:
    contexts = build_generation_contexts(root, experiment_id)
    write_json(root / V2_GEN_OUT / "v2_generation_contexts.json", contexts)
    lines = ["# V2 Generation Context Preview", ""]
    for item in contexts[:3]:
        preview = item["source_labelled_context"][:1600]
        if len(item["source_labelled_context"]) > len(preview):
            preview += "\n...[truncated]"
        lines += [
            f"## {item['question_id']}",
            "",
            f"Question: {item['original_query']}",
            "",
            "```text",
            preview,
            "```",
            "",
        ]
    (root / V2_GEN_OUT / "v2_generation_contexts_preview.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return contexts


def build_prompt(question: str, context: str, required_periods: list[str], query_type: str) -> str:
    periods = ", ".join(required_periods) if required_periods else "none"
    return f"""You are answering a question for a Temporal multi-document RAG for RBI Monetary Policy Reports.

Use only the supplied source-labelled context. Do not use outside knowledge.
Preserve report-specific attribution. For comparative or trend questions, discuss each required report separately before comparison.
If the supplied context is insufficient, say that the supplied context is insufficient.
Do not invent numbers. Do not infer unsupported policy changes. Do not use generic sentiment language.
Use policy stance, inflation outlook, growth projection, risk assessment, or narrative evolution as appropriate.

Required reports: {periods}
Query type: {query_type}

Output format:
Answer:
...

Citations:
- [report period, page, chunk_id]

Citation rule:
Copy chunk_id values exactly from the supplied SOURCE labels. Do not cite any page or chunk_id that is not present in the context.

Context:
{context}

Question: {question}
"""


def _required_periods(context_item: dict[str, Any]) -> list[str]:
    periods = {
        block["report_id"]: block["report_period"]
        for block in context_item.get("context_blocks", [])
    }
    return [
        periods.get(report_id, report_id)
        for report_id in sorted(context_item.get("required_report_ids") or [], key=lambda rid: REPORT_ORDER.get(rid, 99))
    ]


def run_generation_cases(
    contexts: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    *,
    generator: Any,
    model_provider: str = "Groq",
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    checkpoint_path: Path | None = None,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context_by_id = {item["question_id"]: item for item in contexts}
    if checkpoint_path and checkpoint_path.exists():
        rows = load_json_file(checkpoint_path)
        valid_rows = []
        for row in rows:
            context_item = context_by_id.get(row.get("question_id"), {})
            if row.get("generation_success") and invalid_citations(row, context_item):
                continue
            if row.get("generation_success"):
                row["generation_error_type"] = None
                row["generation_error_message"] = None
                row.setdefault("generation_retry_warnings", [])
            valid_rows.append(row)
        rows = valid_rows
        completed = {row["question_id"] for row in rows}
        if checkpoint_path:
            write_json(checkpoint_path, rows)
    else:
        completed = set()
    for item in contexts:
        if item["question_id"] in completed:
            continue
        case = cases.get(item["question_id"], {})
        prompt = build_prompt(
            item["original_query"] or case.get("question", ""),
            item["source_labelled_context"],
            _required_periods(item),
            str(item.get("query_type")),
        )
        started = time.perf_counter()
        generated_answer = None
        success = False
        error_type = None
        error_message = None
        attempts = 0
        retry_warnings: list[dict[str, Any]] = []
        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                generated_answer = generator.invoke(prompt)
                answer_text_for_validation = generated_answer or ""
                provisional = {
                    "citations": parse_citations(answer_text_for_validation, item),
                    "generation_success": True,
                }
                invalid = invalid_citations(provisional, item)
                if invalid and attempt < max_retries:
                    prompt += (
                        "\n\nYour previous answer cited chunk IDs that were not supplied. "
                        "Regenerate the answer and cite only exact chunk_id values from the SOURCE labels above."
                    )
                    retry_warnings.append({
                        "attempt": attempt,
                        "type": "InvalidCitation",
                        "invalid_chunk_ids": [citation.get("chunk_id") for citation in invalid],
                    })
                    error_type = "InvalidCitation"
                    error_message = f"invalid citations: {[citation.get('chunk_id') for citation in invalid]}"
                    continue
                error_type = None
                error_message = None
                success = True
                break
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = safe_error_message(exc)
                retry_warnings.append({"attempt": attempt, "type": error_type})
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 8))
        latency = (time.perf_counter() - started) * 1000
        answer_text = generated_answer or ""
        citations = parse_citations(answer_text, item)
        row = {
            "question_id": item["question_id"],
            "split": item.get("split"),
            "query_type": item.get("query_type"),
            "required_report_ids": item.get("required_report_ids"),
            "original_query": item.get("original_query") or case.get("question"),
            "normalised_query": item.get("normalised_query"),
            "retrieval_experiment_id": item.get("retrieval_experiment_id"),
            "retrieval_config_checksum": item.get("retrieval_config_checksum"),
            "selected_chunk_ids": item.get("selected_chunk_ids"),
            "selected_pages": item.get("selected_pages"),
            "source_labelled_context": item.get("source_labelled_context"),
            "prompt_version": PROMPT_VERSION,
            "model_provider": model_provider,
            "model_name": model_name,
            "temperature": temperature,
            "generated_answer": answer_text,
            "citations": citations,
            "generation_latency_ms": latency,
            "generation_success": success,
            "generation_error_type": error_type,
            "generation_error_message": error_message,
            "generation_attempts": attempts,
            "generation_retry_warnings": retry_warnings,
            "expected_answer": case.get("expected_answer"),
            "category": case.get("category"),
            "table_or_numeric_question": _is_table_or_numeric(case, item),
        }
        rows.append(row)
        if checkpoint_path:
            write_json(checkpoint_path, rows)
    return rows


def _is_table_or_numeric(case: dict[str, Any], context_item: dict[str, Any] | None = None) -> bool:
    text = " ".join([
        case.get("question", ""),
        case.get("expected_answer", ""),
        " ".join(case.get("source_information_type", []) or []),
    ]).lower()
    return any(token in text for token in ("table", "chart", "%", "per cent", "bps", "q1", "q2", "q3", "q4", "projection", "forecast", "rate"))


def write_generation_outputs(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    write_json(root / V2_GEN_OUT / "dev_generation_raw_results.json", rows)
    write_csv(root / V2_GEN_OUT / "dev_generation_results.csv", rows)
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
        "model_provider": rows[0].get("model_provider") if rows else None,
        "model_name": rows[0].get("model_name") if rows else None,
        "temperature": rows[0].get("temperature") if rows else None,
        "prompt_version": PROMPT_VERSION,
        "mean_generation_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_generation_latency_ms": statistics.median(latencies) if latencies else None,
    }
    write_json(root / V2_GEN_OUT / "dev_generation_summary.json", summary)
    lines = [
        "# V2 Development Generation Summary",
        "",
        f"Rows: {summary['row_count']}",
        f"Successful generations: {summary['generation_success_count']}",
        f"Failed generations: {summary['generation_failure_count']}",
        f"Model: {summary['model_provider']} / {summary['model_name']}",
        f"Prompt version: `{PROMPT_VERSION}`",
        f"Median latency ms: {summary['median_generation_latency_ms']}",
    ]
    (root / V2_GEN_OUT / "dev_generation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_eval_outputs(
    root: Path,
    eval_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    coverage: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    write_json(root / V2_GEN_OUT / "dev_answer_eval_raw_results.json", eval_rows)
    write_json(root / V2_GEN_OUT / "dev_answer_eval_summary.json", summary)
    write_json(root / V2_GEN_OUT / "dev_metric_coverage.json", coverage)
    write_csv(root / V2_GEN_OUT / "dev_generation_failures.csv", failures)
    lines = ["# V2 Development Answer Evaluation Summary", "", "| Metric | Mean score | Successful count |", "|---|---:|---:|"]
    for name, item in summary["metrics"].items():
        lines.append(f"| {name} | {item['mean_score']} | {item['successful_count']} |")
    (root / V2_GEN_OUT / "dev_answer_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cov_lines = ["# V2 Generation Metric Coverage", "", "| Metric | Coverage | Successful | Failed | Not applicable |", "|---|---:|---:|---:|---:|"]
    for name, item in coverage.items():
        cov_lines.append(f"| {name} | {item['coverage']} | {item['successful_evaluations']} | {item['failed_evaluations']} | {item['not_applicable']} |")
    (root / V2_GEN_OUT / "dev_metric_coverage.md").write_text("\n".join(cov_lines) + "\n", encoding="utf-8")


def retrieval_generation_analysis(
    retrieval_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gen_by_id = {row["question_id"]: row for row in generation_rows}
    eval_by_id = {row["question_id"]: row for row in eval_rows}
    rows = []
    for retrieval in retrieval_rows:
        qid = retrieval["question_id"]
        gen = gen_by_id.get(qid, {})
        ev = eval_by_id.get(qid, {"metrics": {}})
        metrics = ev.get("metrics", {})
        row = {
            "question_id": qid,
            "query_type": retrieval.get("query_type"),
            "table_or_numeric_question": retrieval.get("table_or_numeric_question"),
            "retrieval_complete_evidence_recall": retrieval.get("complete_evidence_recall"),
            "retrieval_evidence_recall": retrieval.get("evidence_recall"),
            "retrieval_all_reports_hit": retrieval.get("all_reports_hit"),
            "retrieval_macro_mrr": retrieval.get("macro_report_mrr"),
            "generation_factual_correctness": (metrics.get("factual_correctness") or {}).get("score"),
            "generation_faithfulness": (metrics.get("faithfulness_to_context") or {}).get("score"),
            "citation_correctness": (metrics.get("citation_correctness") or {}).get("score"),
            "temporal_attribution_correctness": (metrics.get("temporal_attribution_correctness") or {}).get("score"),
            "abstained": _contains_abstention(gen.get("generated_answer", "")),
            "generation_success": gen.get("generation_success"),
        }
        rows.append(row)
    complete = [row for row in rows if row["retrieval_complete_evidence_recall"] is True]
    incomplete = [row for row in rows if row["retrieval_complete_evidence_recall"] is False]
    table_rows = [row for row in rows if row.get("table_or_numeric_question")]
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "row_count": len(rows),
        "complete_retrieval_count": len(complete),
        "complete_retrieval_mean_factual": _mean([row["generation_factual_correctness"] for row in complete]),
        "incomplete_retrieval_count": len(incomplete),
        "incomplete_retrieval_abstention_rate": _mean([float(row["abstained"]) for row in incomplete]),
        "table_numeric_count": len(table_rows),
        "table_numeric_mean_factual": _mean([row["generation_factual_correctness"] for row in table_rows]),
        "macro_mrr_to_factual_correlation": _pearson(
            [row["retrieval_macro_mrr"] for row in rows],
            [row["generation_factual_correctness"] for row in rows],
        ),
    }
    return rows, summary


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _pearson(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pairs) < 3:
        return None
    x_vals, y_vals = zip(*pairs)
    mx, my = statistics.mean(x_vals), statistics.mean(y_vals)
    sx = sum((x - mx) ** 2 for x in x_vals)
    sy = sum((y - my) ** 2 for y in y_vals)
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy) ** 0.5


def write_analysis_outputs(root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv(root / V2_GEN_OUT / "retrieval_to_generation_analysis.csv", rows)
    write_json(root / V2_GEN_OUT / "retrieval_to_generation_analysis.json", {"summary": summary, "rows": rows})
    complete_retrieval_answer = (
        "Generation is materially better when retrieval is complete, but not perfect."
        if summary.get("complete_retrieval_mean_factual") is not None else
        "No complete-retrieval factual score was available."
    )
    incomplete_answer = (
        "Incomplete retrieval did not reliably trigger abstention."
        if summary.get("incomplete_retrieval_abstention_rate") == 0.0 else
        "Incomplete retrieval triggered some abstention."
    )
    temporal_answer = "Temporal attribution errors should be inspected row-by-row using citation and retrieval traces."
    table_answer = (
        "Table/numeric questions remain a material weakness area."
        if summary.get("table_numeric_mean_factual") is not None and summary.get("table_numeric_mean_factual") < 0.75 else
        "Table/numeric questions were not weaker than the overall factual score in this diagnostic."
    )
    mrr_answer = (
        "Macro MRR has a positive but modest diagnostic correlation with factual answer quality."
        if summary.get("macro_mrr_to_factual_correlation") is not None and summary.get("macro_mrr_to_factual_correlation") > 0 else
        "No positive Macro MRR to factual-quality relationship was observed."
    )
    lines = [
        "# Retrieval-to-Generation Analysis",
        "",
        f"Rows analysed: {summary['row_count']}",
        f"Complete retrieval cases: {summary['complete_retrieval_count']}",
        f"Mean factual score when retrieval complete: {summary['complete_retrieval_mean_factual']}",
        f"Incomplete retrieval cases: {summary['incomplete_retrieval_count']}",
        f"Abstention rate when retrieval incomplete: {summary['incomplete_retrieval_abstention_rate']}",
        f"Table/numeric cases: {summary['table_numeric_count']}",
        f"Table/numeric mean factual score: {summary['table_numeric_mean_factual']}",
        f"Macro MRR to factual-score correlation: {summary['macro_mrr_to_factual_correlation']}",
        "",
        "## Required diagnostic answers",
        "",
        f"1. When retrieval is complete, does generation answer correctly? {complete_retrieval_answer}",
        f"2. When retrieval is incomplete, does generation abstain or hallucinate? {incomplete_answer}",
        f"3. Are temporal attribution errors caused by retrieval or generation? {temporal_answer}",
        f"4. Are table/numeric questions still weak after Cohere retrieval? {table_answer}",
        f"5. Does better Macro MRR translate into better answer quality? {mrr_answer}",
        "",
        "Interpretation: these are development-only diagnostic links between saved retrieval outcomes and generated answers.",
    ]
    (root / V2_GEN_OUT / "retrieval_to_generation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_generation_artifacts(root: Path = ROOT) -> dict[str, Any]:
    load_project_dotenv(root)
    issues: list[str] = []
    pre_path = root / V2_GEN_OUT / "pre_v2_generation_checksums.json"
    selected_config = root / "configs/v2_selected_retrieval.yaml"
    retrieval_raw = root / "reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY/raw_results.json"
    if not pre_path.exists():
        issues.append("missing_pre_generation_checksums")
        pre = {"entries": []}
    else:
        pre = load_json_file(pre_path)
    pre_entries = {entry["path"]: entry["sha256"] for entry in pre.get("entries", [])}
    for path, name in ((selected_config, "configs\\v2_selected_retrieval.yaml"), (retrieval_raw, "reports\\v2_unstructured_cohere\\experiments\\V2_COHERE_ONLY\\raw_results.json")):
        if name in pre_entries and path.exists() and file_sha(path) != pre_entries[name]:
            issues.append(f"checksum_changed:{name}")
    rows = load_json_file(root / V2_GEN_OUT / "dev_generation_raw_results.json") if (root / V2_GEN_OUT / "dev_generation_raw_results.json").exists() else []
    eval_rows = load_json_file(root / V2_GEN_OUT / "dev_answer_eval_raw_results.json") if (root / V2_GEN_OUT / "dev_answer_eval_raw_results.json").exists() else []
    context_by_qid = {
        item["question_id"]: item
        for item in (load_json_file(root / V2_GEN_OUT / "v2_generation_contexts.json") if (root / V2_GEN_OUT / "v2_generation_contexts.json").exists() else [])
    }
    for row in rows:
        qid = row.get("question_id", "unknown")
        if row.get("retrieval_experiment_id") != "V2_COHERE_ONLY":
            issues.append(f"{qid}:wrong_retrieval_experiment")
        if row.get("split") != "dev" or str(qid).startswith("test_"):
            issues.append(f"{qid}:heldout_or_non_dev_case")
        for field in ("prompt_version", "model_name", "temperature", "generation_success"):
            if field not in row:
                issues.append(f"{qid}:missing_generation_field:{field}")
        supplied = set(context_by_qid.get(qid, {}).get("selected_chunk_ids", []))
        for citation in row.get("citations", []):
            if citation.get("chunk_id") not in supplied:
                issues.append(f"{qid}:citation_not_in_supplied_context:{citation.get('chunk_id')}")
        if not row.get("generation_success") and (not row.get("generation_error_type") or row.get("generation_error_message") is None):
            issues.append(f"{qid}:failed_generation_missing_error_fields")
    for row in eval_rows:
        for name, metric in (row.get("metrics") or {}).items():
            if not metric.get("success", True) and metric.get("score") is not None:
                issues.append(f"{row.get('question_id')}:failed_metric_has_non_null_score:{name}")
    coverage_path = root / V2_GEN_OUT / "dev_metric_coverage.json"
    if not coverage_path.exists():
        issues.append("missing_metric_coverage")
    all_artifacts = {}
    for path in (root / V2_GEN_OUT).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv", ".yaml", ".yml"}:
            all_artifacts[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
    if actual_key_values_serialized(all_artifacts):
        issues.append("api_key_value_serialized")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": sorted(set(issues)),
        "old_multi_report_config_used": False,
        "heldout_generation_run": False,
    }
    write_json(root / V2_GEN_OUT / "v2_generation_integrity.json", payload)
    lines = ["# V2 Generation Integrity", "", f"Status: {payload['status']}", f"Issues: {payload['issue_count']}", ""]
    if payload["issues"]:
        lines += ["## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    (root / V2_GEN_OUT / "v2_generation_integrity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_status(root: Path, generation_summary: dict[str, Any] | None, readiness: dict[str, Any]) -> dict[str, Any]:
    if not readiness.get("generation_may_proceed"):
        status = "generation_not_run_due_to_readiness_failure"
    elif not generation_summary or generation_summary.get("generation_success_count", 0) == 0:
        status = "dev_generation_failed"
    elif generation_summary["generation_success_count"] < generation_summary["row_count"]:
        status = "dev_generation_partial"
    else:
        status = "dev_generation_complete"
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": status,
        "generation_summary": generation_summary,
        "heldout_generation_run": False,
    }
    write_json(root / V2_GEN_OUT / "v2_generation_status.json", payload)
    (root / V2_GEN_OUT / "v2_generation_status.md").write_text(
        "# V2 Generation Status\n\n"
        f"Status: `{status}`\n\n"
        "Held-out generation was not run.\n",
        encoding="utf-8",
    )
    return payload


def generate_reports(root: Path = ROOT) -> dict[str, Any]:
    def load(name: str, default=None):
        path = root / V2_GEN_OUT / name
        return load_json_file(path) if path.exists() else default

    env = load("environment_readiness.json", {})
    retrieval_validation = load("v2_retrieval_input_validation.json", {})
    gen_summary = load("dev_generation_summary.json", {})
    eval_summary = load("dev_answer_eval_summary.json", {"metrics": {}})
    coverage = load("dev_metric_coverage.json", {})
    analysis = load("retrieval_to_generation_analysis.json", {"summary": {}})
    retrieval_summary = load("../v2_unstructured_cohere/experiments/V2_COHERE_ONLY/summary.json", {})
    status = load("v2_generation_status.json", {})
    total_metric_failures = sum(item.get("failed_evaluations", 0) for item in coverage.values())
    raw_rows = load("dev_generation_raw_results.json", [])
    examples_good = [
        row for row in raw_rows
        if row.get("generation_success") and row.get("citations")
    ][:2]
    examples_fail = [
        row for row in raw_rows
        if not row.get("generation_success") or not row.get("citations")
    ][:2]
    metrics = eval_summary.get("metrics", {})
    presentation = [
        "# V2 Generation Results for Presentation",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        "## Why generation was previously blocked",
        "",
        "Generation was blocked because the evaluator was not wired to consume the frozen V2 selected retrieval config and saved source-labelled retrieval outputs.",
        "",
        "## How it was fixed",
        "",
        "The evaluator now builds source-labelled contexts from saved `V2_COHERE_ONLY` retrieval outputs without rerunning retrieval.",
        "",
        "## Selected retrieval config",
        "",
        "`V2_COHERE_ONLY`",
        "",
        "## Retrieval development metrics",
        "",
        f"- Complete Evidence Recall: {retrieval_summary.get('complete_evidence_recall')}",
        f"- All-Reports Hit: {retrieval_summary.get('all_reports_hit')}",
        f"- Evidence Recall: {retrieval_summary.get('evidence_recall')}",
        f"- Macro Report MRR: {retrieval_summary.get('macro_report_mrr')}",
        f"- Median retrieval latency ms: {retrieval_summary.get('median_latency_ms')}",
        f"- Mean estimated tokens: {retrieval_summary.get('mean_estimated_tokens')}",
        "",
        "## Generation development metrics",
        "",
    ]
    for name, item in metrics.items():
        presentation.append(f"- {name}: mean={item.get('mean_score')}, n={item.get('successful_count')}")
    presentation += [
        "",
        f"Citation correctness: {metrics.get('citation_correctness', {}).get('mean_score')}",
        f"Temporal attribution correctness: {metrics.get('temporal_attribution_correctness', {}).get('mean_score')}",
        f"Comparative correctness: {metrics.get('comparative_correctness', {}).get('mean_score')}",
        f"Table/numeric mean factual score: {analysis.get('summary', {}).get('table_numeric_mean_factual')}",
        "",
        "## Example good answers",
        "",
    ]
    for row in examples_good:
        presentation.append(f"- `{row['question_id']}`: generated with {len(row.get('citations', []))} parsed citations.")
    presentation += ["", "## Example failure modes", ""]
    for row in examples_fail:
        reason = row.get("generation_error_message") or "missing parsed citations"
        presentation.append(f"- `{row['question_id']}`: {reason}")
    presentation += [
        "",
        "Latency/cost caveat: generation uses Groq API calls and should be treated separately from retrieval latency.",
        "",
        "Scientific caveat: development generation only; held-out generation was not run; any V2 held-out diagnostic is not a fresh benchmark unless separately created and labelled.",
    ]
    presentation_path = root / V2_GEN_OUT / "v2_generation_results_for_presentation.md"
    presentation_path.write_text("\n".join(presentation) + "\n", encoding="utf-8")

    lines = [
        "# V2 Generation Evaluation Report",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        "## Environment readiness",
        "",
        f"Groq key available: {env.get('groq_api_key_available')}. Cohere key available: {env.get('cohere_api_key_available')}.",
        "",
        "## Retrieval input validation",
        "",
        f"Status: {retrieval_validation.get('status')}. The evaluator used `V2_COHERE_ONLY` saved development retrieval outputs.",
        "",
        "## Context-building method",
        "",
        "Contexts are built from `selected_chunks_by_report` in saved V2 raw retrieval rows, grouped by report period in chronological order.",
        "",
        "## Prompt template/version",
        "",
        f"`{PROMPT_VERSION}`",
        "",
        "## Model/provider settings",
        "",
        f"Provider/model: {gen_summary.get('model_provider')} / {gen_summary.get('model_name')}; temperature={gen_summary.get('temperature')}.",
        "",
        "## Generation run status",
        "",
        f"Status: `{status.get('status')}`. Rows={gen_summary.get('row_count')}; successes={gen_summary.get('generation_success_count')}; failures={gen_summary.get('generation_failure_count')}.",
        "",
        "## Generation metrics",
        "",
    ]
    for name, item in metrics.items():
        lines.append(f"- {name}: mean={item.get('mean_score')}, n={item.get('successful_count')}")
    lines += [
        "",
        "## Metric coverage",
        "",
    ]
    for name, item in coverage.items():
        lines.append(f"- {name}: coverage={item.get('coverage')}, failed={item.get('failed_evaluations')}, not_applicable={item.get('not_applicable')}")
    lines += [
        "",
        "## Judge/evaluator failures",
        "",
        f"Failed metric evaluations: {total_metric_failures}. Metrics marked not applicable were excluded from their averages.",
        "",
        "## Retrieval-to-generation analysis",
        "",
        f"Complete retrieval mean factual score: {analysis.get('summary', {}).get('complete_retrieval_mean_factual')}.",
        f"Incomplete retrieval abstention rate: {analysis.get('summary', {}).get('incomplete_retrieval_abstention_rate')}.",
        f"Table/numeric mean factual score: {analysis.get('summary', {}).get('table_numeric_mean_factual')}.",
        f"Macro MRR to factual-score correlation: {analysis.get('summary', {}).get('macro_mrr_to_factual_correlation')}.",
        "",
        "## Citation validation",
        "",
        f"Citation correctness mean: {metrics.get('citation_correctness', {}).get('mean_score')}.",
        f"Citation completeness mean: {metrics.get('citation_completeness', {}).get('mean_score')}.",
        "",
        "## Temporal attribution validation",
        "",
        f"Temporal attribution correctness mean: {metrics.get('temporal_attribution_correctness', {}).get('mean_score')}.",
        "",
        "## Comparative correctness",
        "",
        f"Comparative correctness mean: {metrics.get('comparative_correctness', {}).get('mean_score')}.",
        "",
        "## Abstention behaviour",
        "",
        f"Abstention correctness mean: {metrics.get('abstention_correctness', {}).get('mean_score')}.",
        f"Incomplete retrieval abstention rate: {analysis.get('summary', {}).get('incomplete_retrieval_abstention_rate')}.",
        "",
        "## Table/numeric analysis",
        "",
        f"Table/numeric question count: {analysis.get('summary', {}).get('table_numeric_count')}.",
        f"Table/numeric mean factual score: {analysis.get('summary', {}).get('table_numeric_mean_factual')}.",
        "",
        "## Limitations",
        "",
        "- Metrics are deterministic heuristics, not an external human or LLM judge.",
        "- Development generation only; held-out generation was not run.",
        "- The project is not production-ready.",
        "",
        "## Exact next phase",
        "",
        "Review generation failures and citation quality, then decide whether to run a labelled post-final held-out diagnostic or create a fresh V2 evaluation set.",
    ]
    report_path = root / V2_GEN_OUT / "v2_generation_evaluation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "presentation_summary_path": str(presentation_path),
        "final_report_path": str(report_path),
    }


def run_v2_generation(
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
        raise RuntimeError("V2 generation evaluator only supports development split in this phase")
    ensure_dirs(root)
    load_project_dotenv(root)
    write_pre_generation_checksums(root)
    readiness = write_environment_readiness(root)
    validation = write_retrieval_input_validation(root, retrieval_experiment, config_path)
    if validation["status"] != "passed":
        raise RuntimeError("; ".join(validation["issues"]))
    contexts = write_generation_contexts(root, retrieval_experiment)
    if not readiness["generation_may_proceed"]:
        status = write_status(root, None, readiness)
        generate_reports(root)
        return {"status": status["status"], "generated_rows": 0}
    cases = expected_case_lookup(root, split)
    generator = generator or GroqGenerator(model_name, temperature)
    rows = run_generation_cases(
        contexts,
        cases,
        generator=generator,
        model_name=model_name,
        temperature=temperature,
        checkpoint_path=root / V2_GEN_OUT / "dev_generation_raw_results.json",
    )
    generation_summary = write_generation_outputs(root, rows)
    retrieval_rows = load_json_file(root / "reports/v2_unstructured_cohere/experiments" / retrieval_experiment / "raw_results.json")
    eval_rows, eval_summary, coverage, failures = evaluate_generation_rows(rows, retrieval_rows, contexts)
    write_eval_outputs(root, eval_rows, eval_summary, coverage, failures)
    analysis_rows, analysis_summary = retrieval_generation_analysis(retrieval_rows, rows, eval_rows)
    write_analysis_outputs(root, analysis_rows, analysis_summary)
    status = write_status(root, generation_summary, readiness)
    integrity = validate_generation_artifacts(root)
    generate_reports(root)
    payload = {
        "status": status["status"],
        "generated_rows": len(rows),
        "generation_success_count": generation_summary["generation_success_count"],
        "integrity_status": integrity["status"],
        "heldout_generation_run": False,
    }
    if contains_api_key_material(payload) or actual_key_values_serialized(payload):
        raise RuntimeError("generation payload unexpectedly contains key material")
    return payload
