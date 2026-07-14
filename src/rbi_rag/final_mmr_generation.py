from __future__ import annotations

import json
import os
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env_loading import load_project_dotenv
from .evidence_sufficiency import classify_all
from .final_evaluation import make_checksum_manifest, write_csv, write_json
from .v2_generation_contexts import REPORT_ORDER
from .v2_generation_evaluation import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    METRIC_NAMES,
    GroqGenerator,
    actual_key_values_serialized,
    evaluate_generation_rows,
    expected_case_lookup,
    load_json_file,
    now_iso,
    summarise_eval_metrics,
    summarise_metric_coverage,
)
from .v2_sufficiency import PROMPT_VERSION, run_sufficiency_generation_cases


OUT_DIR = Path("reports/final_mmr_generation")
EXPERIMENT_ID = "GEN_MMR06_SUFFICIENCY_V1"
RETRIEVAL_ID = "MMR_LAMBDA_06"
MMR_RAW = Path("reports/mmr_experiments/experiments/MMR_LAMBDA_06/raw_results.json")
MAX_CONTEXT_TOKENS = 3000
CHECKSUM_TARGETS = [
    Path("configs/v2_selected_retrieval.yaml"),
    Path("configs/v2_mmr_selected_retrieval.yaml"),
    Path("configs/mmr_experiments.yaml"),
    Path("reports/mmr_experiments"),
    Path("reports/v2_generation"),
    Path("reports/v2_sufficiency"),
    Path("reports/final_comparison"),
    Path("reports/final_packaging"),
    Path("data/evaluation"),
]


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def archive_existing_output(root: Path) -> dict[str, Any]:
    out = root / OUT_DIR
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return {"status": "not_needed", "archive_dir": None, "item_count": 0}
    items = [
        item for item in out.iterdir()
        if not item.name.startswith("archive_previous_run_") and item.name != "run_logs"
    ]
    if not items:
        return {"status": "not_needed", "archive_dir": None, "item_count": 0}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = out / f"archive_previous_run_{stamp}"
    archive.mkdir(parents=True, exist_ok=True)
    for item in items:
        target = archive / item.name
        if item.is_dir():
            shutil.move(str(item), str(target))
        else:
            shutil.move(str(item), str(target))
    return {"status": "archived", "archive_dir": rel(root, archive), "item_count": len(items)}


def write_pre_checksums(root: Path) -> dict[str, Any]:
    payload = make_checksum_manifest(root, CHECKSUM_TARGETS)
    write_json(root / OUT_DIR / "pre_final_mmr_generation_checksums.json", payload)
    lines = [
        "# Pre-final MMR Generation Checksums",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Files captured: {payload['entry_count']}",
        "",
    ]
    if payload.get("missing_targets"):
        lines += ["## Missing targets", ""]
        lines.extend(f"- `{item}`" for item in payload["missing_targets"])
        lines.append("")
    lines += ["## Targets", ""]
    lines.extend(f"- `{target}`" for target in payload["targets"])
    write_md(root / OUT_DIR / "pre_final_mmr_generation_checksums.md", lines)
    return payload


def validate_inputs(root: Path) -> dict[str, Any]:
    required = [
        MMR_RAW,
        Path("reports/mmr_experiments/experiments/MMR_LAMBDA_06/question_results.csv"),
        Path("reports/mmr_experiments/mmr_leaderboard.json"),
        Path("reports/mmr_experiments/mmr_selection_decision.json"),
        Path("src/rbi_rag/evidence_sufficiency.py"),
        Path("src/rbi_rag/v2_sufficiency.py"),
    ]
    baseline_candidates = [
        Path("reports/v2_sufficiency/dev_sufficiency_eval_summary.json"),
        Path("reports/v2_sufficiency/dev_answer_eval_summary.json"),
    ]
    missing = [str(path).replace("/", "\\") for path in required if not (root / path).exists()]
    baseline = next((path for path in baseline_candidates if (root / path).exists()), None)
    if baseline is None:
        missing.append("reports\\v2_sufficiency\\dev_sufficiency_eval_summary.json")
    rows = read_json(root / MMR_RAW, []) if (root / MMR_RAW).exists() else []
    non_dev = [row.get("question_id") for row in rows if row.get("split") != "dev"]
    wrong_experiment = [row.get("question_id") for row in rows if row.get("experiment_id") != RETRIEVAL_ID]
    missing_text = []
    for row in rows:
        chunks = [chunk for chunks in (row.get("selected_chunks_by_report") or {}).values() for chunk in chunks]
        if row.get("query_type") != "unsupported_period" and not chunks:
            missing_text.append(row.get("question_id"))
        for chunk in chunks:
            if not chunk.get("chunk_id") or chunk.get("text") is None or chunk.get("page") is None:
                missing_text.append(row.get("question_id"))
    issues = list(missing)
    if non_dev:
        issues.append(f"non_dev_rows:{sorted(set(non_dev))}")
    if wrong_experiment:
        issues.append(f"wrong_experiment_rows:{sorted(set(wrong_experiment))}")
    if missing_text:
        issues.append(f"missing_selected_chunk_text:{sorted(set(missing_text))}")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "blocked_missing_mmr_inputs",
        "required_inputs": [str(path).replace("/", "\\") for path in required],
        "previous_generation_baseline": str(baseline).replace("/", "\\") if baseline else None,
        "row_count": len(rows),
        "issues": issues,
        "heldout_rows_present": bool(non_dev),
    }
    write_json(root / OUT_DIR / "input_artifact_validation.json", payload)
    lines = [
        "# Final MMR Generation Input Artifact Validation",
        "",
        f"Status: `{payload['status']}`",
        f"MMR rows: {len(rows)}",
        f"Previous generation baseline: `{payload['previous_generation_baseline']}`",
    ]
    if issues:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in issues)
    write_md(root / OUT_DIR / "input_artifact_validation.md", lines)
    return payload


def _chunk_rank_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace = {}
    for index, item in enumerate(row.get("mmr_trace") or []):
        chunk_id = item.get("chunk_id")
        if chunk_id:
            trace[chunk_id] = {**item, "trace_index": index}
    return trace


def _source_label(block: dict[str, Any]) -> str:
    return f"[SOURCE: {block['report_period']} MPR | page {block['page_number']} | chunk {block['chunk_id']}]"


def _estimated_tokens(blocks: list[dict[str, Any]]) -> int:
    return int(sum(len(block.get("text") or "") for block in blocks) / 4)


def _order_blocks(blocks: list[dict[str, Any]], ordering: str) -> list[dict[str, Any]]:
    if ordering == "rerank_order":
        return sorted(
            blocks,
            key=lambda item: (
                item.get("mmr_selection_rank") if item.get("mmr_selection_rank") is not None else 10_000,
                REPORT_ORDER.get(item["report_id"], 99),
                item.get("page_number") or 0,
                item["chunk_id"],
            ),
        )
    return sorted(
        blocks,
        key=lambda item: (
            REPORT_ORDER.get(item["report_id"], 99),
            item.get("page_number") or 0,
            item["chunk_id"],
        ),
    )


def _apply_context_budget(blocks: list[dict[str, Any]], required: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if _estimated_tokens(blocks) <= MAX_CONTEXT_TOKENS:
        return blocks, warnings
    kept = list(blocks)
    def removable(block: dict[str, Any]) -> bool:
        return sum(1 for item in kept if item["report_id"] == block["report_id"]) > 1
    while _estimated_tokens(kept) > MAX_CONTEXT_TOKENS and any(removable(block) for block in kept):
        ranked = sorted(
            [block for block in kept if removable(block)],
            key=lambda item: (
                item.get("mmr_selection_rank") if item.get("mmr_selection_rank") is not None else 10_000,
                item.get("reranker_score") or 0.0,
            ),
        )
        drop = ranked[-1]
        kept.remove(drop)
        warnings.append(f"context_budget_drop:{drop['chunk_id']}")
    missing_reports = [report_id for report_id in required if not any(block["report_id"] == report_id for block in kept)]
    if missing_reports:
        warnings.append(f"context_budget_missing_required_report:{missing_reports}")
    return kept, warnings


def build_context_for_mmr_row(row: dict[str, Any], *, ordering: str = "page_order") -> dict[str, Any]:
    required = list(row.get("required_report_ids") or [])
    required_set = set(required)
    trace = _chunk_rank_map(row)
    blocks = []
    for report_id in required:
        for chunk in (row.get("selected_chunks_by_report") or {}).get(report_id, []):
            if chunk.get("report_id") not in required_set or not chunk.get("text"):
                continue
            item = trace.get(chunk.get("chunk_id"), {})
            block = {
                "report_period": chunk.get("report_period") or report_id,
                "report_id": report_id,
                "page_number": int(chunk.get("page")) if chunk.get("page") is not None else None,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "mmr_selection_rank": item.get("selection_rank"),
                "mmr_score": item.get("mmr_score"),
                "reranker_score": item.get("reranker_score"),
                "trace_index": item.get("trace_index"),
            }
            block["source_label"] = _source_label(block)
            blocks.append(block)
    blocks = _order_blocks(blocks, ordering)
    blocks, warnings = _apply_context_budget(blocks, required)
    grouped: list[str] = []
    current_report = None
    for block in blocks:
        if block["report_id"] != current_report:
            current_report = block["report_id"]
            grouped.append(f"## {block['report_period']}")
        grouped.append(f"{block['source_label']}\n{block['text']}")
    selected_pages = {
        report_id: [
            block["page_number"] for block in blocks
            if block["report_id"] == report_id and block.get("page_number") is not None
        ]
        for report_id in sorted(required_set, key=lambda rid: REPORT_ORDER.get(rid, 99))
    }
    return {
        "question_id": row["question_id"],
        "split": row.get("split"),
        "query_type": row.get("query_type"),
        "required_report_ids": required,
        "original_query": row.get("original_query"),
        "normalised_query": row.get("normalised_query"),
        "retrieval_experiment_id": RETRIEVAL_ID,
        "retrieval_config_checksum": row.get("configuration_checksum"),
        "context_ordering": ordering,
        "selected_chunks_by_report": {
            report_id: [block for block in blocks if block["report_id"] == report_id]
            for report_id in sorted(required_set, key=lambda rid: REPORT_ORDER.get(rid, 99))
        },
        "selected_chunk_ids": [block["chunk_id"] for block in blocks],
        "selected_pages": selected_pages,
        "context_blocks": blocks,
        "source_labelled_context": "\n\n".join(grouped),
        "retrieval_complete_evidence_recall": row.get("retrieval_complete_evidence_recall"),
        "retrieval_evidence_recall": row.get("retrieval_evidence_recall"),
        "retrieval_all_reports_hit": row.get("retrieval_all_reports_hit"),
        "retrieval_macro_mrr": row.get("retrieval_macro_mrr"),
        "report_coverage": row.get("report_coverage"),
        "single_report_contamination": row.get("single_report_contamination"),
        "estimated_token_count": _estimated_tokens(blocks),
        "unique_page_count": len({(block["report_id"], block["page_number"]) for block in blocks}),
        "repeated_text_ratio": row.get("repeated_text_ratio"),
        "context_warnings": warnings,
    }


def build_mmr_contexts(root: Path, *, ordering: str = "page_order", output_prefix: str = "mmr06") -> list[dict[str, Any]]:
    rows = read_json(root / MMR_RAW, [])
    contexts = [build_context_for_mmr_row(row, ordering=ordering) for row in rows]
    write_json(root / OUT_DIR / f"{output_prefix}_source_labelled_contexts.json", contexts)
    write_csv(root / OUT_DIR / f"{output_prefix}_source_labelled_contexts.csv", contexts)
    warnings = [warning for item in contexts for warning in item.get("context_warnings", [])]
    lines = [
        "# MMR06 Context Construction Summary",
        "",
        f"Rows: {len(contexts)}",
        f"Ordering: `{ordering}`",
        f"Mean estimated tokens: {_mean([row.get('estimated_token_count') for row in contexts])}",
        f"Warnings: {len(warnings)}",
        "",
        "Contexts use saved `MMR_LAMBDA_06` selected chunks. Retrieval was not rerun.",
    ]
    if warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings[:50])
    write_md(root / OUT_DIR / f"{output_prefix}_context_construction_summary.md", lines)
    return contexts


def run_sufficiency_classification(root: Path, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_json(root / MMR_RAW, [])
    cases = expected_case_lookup(root, "dev")
    classifications = classify_all(rows, contexts, cases)
    write_json(root / OUT_DIR / "mmr06_sufficiency_classification.json", classifications)
    write_csv(root / OUT_DIR / "mmr06_sufficiency_classification.csv", classifications)
    status_counts = Counter(row["sufficiency_status"] for row in classifications)
    behavior_counts = Counter(row["required_generation_behavior"] for row in classifications)
    previous = read_json(root / "reports/v2_sufficiency/dev_sufficiency_classification.json", [])
    previous_status = Counter(row.get("sufficiency_status") for row in previous)
    previous_behavior = Counter(row.get("required_generation_behavior") for row in previous)
    lines = [
        "# MMR06 Sufficiency Classification",
        "",
        "## MMR06 counts",
        "",
        *[f"- {key}: {value}" for key, value in sorted(status_counts.items())],
        "",
        "## MMR06 required behaviours",
        "",
        *[f"- {key}: {value}" for key, value in sorted(behavior_counts.items())],
        "",
        "## Previous V2 Cohere counts",
        "",
        *[f"- {key}: {value}" for key, value in sorted(previous_status.items())],
        "",
        "## Previous V2 Cohere required behaviours",
        "",
        *[f"- {key}: {value}" for key, value in sorted(previous_behavior.items())],
    ]
    write_md(root / OUT_DIR / "mmr06_sufficiency_classification.md", lines)
    return classifications


def write_generation_summary(root: Path, exp_dir: Path, rows: list[dict[str, Any]], status: str = "completed", reason: str | None = None) -> dict[str, Any]:
    successes = [row for row in rows if row.get("generation_success")]
    latencies = [float(row["generation_latency_ms"]) for row in rows if isinstance(row.get("generation_latency_ms"), (int, float))]
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "experiment_id": EXPERIMENT_ID,
        "retrieval_source": RETRIEVAL_ID,
        "split": "dev",
        "status": status,
        "reason": reason,
        "row_count": len(rows),
        "generation_success_count": len(successes),
        "generation_failure_count": len(rows) - len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "model_provider": rows[0].get("model_provider") if rows else "Groq",
        "model_name": rows[0].get("model_name") if rows else DEFAULT_MODEL,
        "temperature": rows[0].get("temperature") if rows else DEFAULT_TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "sufficiency_gate_enabled": True,
        "mean_generation_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_generation_latency_ms": statistics.median(latencies) if latencies else None,
        "behavior_counts": dict(Counter(row.get("required_generation_behavior") for row in rows)),
        "sufficiency_status_counts": dict(Counter(row.get("sufficiency_status") for row in rows)),
        "hard_gate_abstention_count": sum(1 for row in rows if row.get("model_provider") == "deterministic_sufficiency_gate"),
    }
    write_json(exp_dir / "summary.json", summary)
    lines = [
        f"# {EXPERIMENT_ID} Summary",
        "",
        f"Status: `{status}`",
        f"Retrieval source: `{RETRIEVAL_ID}`",
        f"Rows: {summary['row_count']}",
        f"Successes: {summary['generation_success_count']}",
        f"Failures: {summary['generation_failure_count']}",
        f"Prompt version: `{PROMPT_VERSION}`",
        f"Median generation latency ms: {summary['median_generation_latency_ms']}",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    write_md(exp_dir / "summary.md", lines)
    return summary


def run_generation(root: Path, *, generator: Any | None = None) -> dict[str, Any]:
    load_project_dotenv(root)
    contexts = read_json(root / OUT_DIR / "mmr06_source_labelled_contexts.json", [])
    classifications = read_json(root / OUT_DIR / "mmr06_sufficiency_classification.json", [])
    cases = expected_case_lookup(root, "dev")
    exp_dir = root / OUT_DIR / EXPERIMENT_ID
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not os.getenv("GROQ_API_KEY") and generator is None:
        reason = "GROQ_API_KEY is unavailable"
        write_json(exp_dir / "raw_results.json", [])
        write_csv(exp_dir / "results.csv", [])
        summary = write_generation_summary(root, exp_dir, [], status="blocked", reason=reason)
        return {"status": "blocked", "reason": reason, "summary": summary}
    generator = generator or GroqGenerator(DEFAULT_MODEL, DEFAULT_TEMPERATURE)
    rows = run_sufficiency_generation_cases(
        contexts,
        cases,
        classifications,
        generator=generator,
        model_name=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        checkpoint_path=exp_dir / "raw_results.json",
        max_retries=2,
    )
    for row in rows:
        row["experiment_id"] = EXPERIMENT_ID
        row["retrieval_experiment_id"] = RETRIEVAL_ID
    write_json(exp_dir / "raw_results.json", rows)
    write_csv(exp_dir / "results.csv", rows)
    summary = write_generation_summary(root, exp_dir, rows)
    if actual_key_values_serialized({"rows": rows, "summary": summary}):
        raise RuntimeError("API key material serialized in final MMR generation outputs")
    return {"status": summary["status"], "summary": summary}


def evaluate_generation(root: Path) -> dict[str, Any]:
    exp_dir = root / OUT_DIR / EXPERIMENT_ID
    rows = read_json(exp_dir / "raw_results.json", [])
    retrieval_rows = read_json(root / MMR_RAW, [])
    contexts = read_json(root / OUT_DIR / "mmr06_source_labelled_contexts.json", [])
    if not rows:
        payload = {"schema_version": 1, "created_at_utc": now_iso(), "status": "blocked", "reason": "generation rows unavailable", "metrics": {}}
        write_json(exp_dir / "eval_summary.json", payload)
        write_json(exp_dir / "eval_raw_results.json", [])
        write_json(exp_dir / "metric_coverage.json", {})
        write_md(exp_dir / "eval_summary.md", ["# MMR Generation Evaluation", "", "Status: blocked", "Reason: generation rows unavailable"])
        write_md(exp_dir / "metric_coverage.md", ["# Metric Coverage", "", "Status: blocked"])
        return payload
    eval_rows, summary, coverage, failures = evaluate_generation_rows(rows, retrieval_rows, contexts)
    write_json(exp_dir / "eval_raw_results.json", eval_rows)
    write_json(exp_dir / "eval_summary.json", summary)
    write_json(exp_dir / "metric_coverage.json", coverage)
    write_csv(exp_dir / "generation_failures.csv", failures)
    lines = ["# MMR Generation Evaluation Summary", "", "| Metric | Mean score | Successful count |", "|---|---:|---:|"]
    for name, item in summary["metrics"].items():
        lines.append(f"| {name} | {item['mean_score']} | {item['successful_count']} |")
    write_md(exp_dir / "eval_summary.md", lines)
    cov_lines = ["# MMR Generation Metric Coverage", "", "| Metric | Coverage | Successful | Failed | Not applicable |", "|---|---:|---:|---:|---:|"]
    for name, item in coverage.items():
        cov_lines.append(f"| {name} | {item['coverage']} | {item['successful_evaluations']} | {item['failed_evaluations']} | {item['not_applicable']} |")
    write_md(exp_dir / "metric_coverage.md", cov_lines)
    return summary


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float, bool))]
    return sum(numeric) / len(numeric) if numeric else None


def _metric(summary: dict[str, Any], name: str) -> float | None:
    return ((summary.get("metrics") or {}).get(name) or {}).get("mean_score")


def write_generation_comparison(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_eval = read_json(root / "reports/v2_sufficiency/dev_sufficiency_eval_summary.json", {"metrics": {}})
    previous_gen = read_json(root / "reports/v2_sufficiency/dev_generation_sufficiency_summary.json", {})
    mmr_eval = read_json(root / OUT_DIR / EXPERIMENT_ID / "eval_summary.json", {"metrics": {}})
    mmr_gen = read_json(root / OUT_DIR / EXPERIMENT_ID / "summary.json", {})
    mmr_retrieval = read_json(root / "reports/mmr_experiments/experiments/MMR_LAMBDA_06/summary.json", {})
    v2_retrieval = read_json(root / "reports/mmr_experiments/experiments/MMR_BASELINE_V2_COHERE/summary.json", {})
    rows = []
    for label, eval_summary, gen_summary, retrieval_summary in [
        ("V2_COHERE_ONLY_SUFFICIENCY", previous_eval, previous_gen, v2_retrieval),
        (EXPERIMENT_ID, mmr_eval, mmr_gen, mmr_retrieval),
    ]:
        row = {
            "experiment_id": label,
            "retrieval_source": retrieval_summary.get("experiment_id", "V2_COHERE_ONLY" if label.startswith("V2") else RETRIEVAL_ID),
            "CER": retrieval_summary.get("complete_evidence_recall"),
            "All-Reports Hit": retrieval_summary.get("all_reports_hit"),
            "Evidence Recall": retrieval_summary.get("evidence_recall"),
            "Macro MRR": retrieval_summary.get("macro_report_mrr"),
            "repeated_text_ratio": retrieval_summary.get("mean_repeated_text_ratio"),
            "unique_page_count": retrieval_summary.get("mean_unique_pages"),
            "generation_latency_ms": gen_summary.get("median_generation_latency_ms"),
            "retrieval_latency_ms": retrieval_summary.get("median_latency_ms"),
            "total_estimated_latency_ms": (
                (gen_summary.get("median_generation_latency_ms") or 0) + (retrieval_summary.get("median_latency_ms") or 0)
                if gen_summary.get("median_generation_latency_ms") is not None and retrieval_summary.get("median_latency_ms") is not None else None
            ),
            "estimated_token_count": retrieval_summary.get("mean_estimated_tokens"),
        }
        for name in METRIC_NAMES:
            row[name] = _metric(eval_summary, name)
        rows.append(row)
    prev, mmr = rows[0], rows[1]
    delta = {
        "experiment_id": "DELTA_MMR_MINUS_PREVIOUS",
        **{
            key: (mmr.get(key) - prev.get(key) if isinstance(mmr.get(key), (int, float)) and isinstance(prev.get(key), (int, float)) else None)
            for key in rows[0]
            if key not in {"experiment_id", "retrieval_source"}
        },
    }
    rows.append(delta)
    write_json(root / OUT_DIR / "generation_comparison.json", {"created_at_utc": now_iso(), "rows": rows})
    write_csv(root / OUT_DIR / "generation_comparison.csv", rows)
    lines = ["# Final MMR Generation Comparison", "", "| Experiment | CER | Hit | Factual | Faithfulness | Citation | Temporal | Comparative | Abstention | Retrieval ms | Generation ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row.get('experiment_id')} | {row.get('CER')} | {row.get('All-Reports Hit')} | {row.get('factual_correctness')} | "
            f"{row.get('faithfulness_to_context')} | {row.get('citation_correctness')} | {row.get('temporal_attribution_correctness')} | "
            f"{row.get('comparative_correctness')} | {row.get('abstention_correctness')} | {row.get('retrieval_latency_ms')} | {row.get('generation_latency_ms')} |"
        )
    write_md(root / OUT_DIR / "generation_comparison.md", lines)
    return rows, delta


def write_selection_decision(root: Path, comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous, mmr = comparison_rows[0], comparison_rows[1]
    factual_delta = (mmr.get("factual_correctness") or 0) - (previous.get("factual_correctness") or 0)
    citation_delta = (mmr.get("citation_correctness") or 0) - (previous.get("citation_correctness") or 0)
    temporal_delta = (mmr.get("temporal_attribution_correctness") or 0) - (previous.get("temporal_attribution_correctness") or 0)
    abstention_delta = (mmr.get("abstention_correctness") or 0) - (previous.get("abstention_correctness") or 0)
    factual_worse = factual_delta < 0
    citation_worse = citation_delta < 0
    improves_generation = (factual_delta > 0 or citation_delta > 0 or temporal_delta > 0) and abstention_delta >= 0
    if improves_generation and not factual_worse and not citation_worse:
        status = "selected_mmr_end_to_end"
        best = EXPERIMENT_ID
        reason = "MMR generation improved at least one key generation metric without reducing factual, citation, or abstention correctness."
    else:
        status = "keep_previous_generation"
        best = "V2_COHERE_ONLY_SUFFICIENCY"
        reason = "MMR retrieval remains selected for retrieval-only development metrics, but MMR generation did not improve enough to replace the previous evaluated generation setting."
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": status,
        "best_end_to_end_generation_setting": best,
        "best_retrieval_method": RETRIEVAL_ID,
        "best_generation_method": best,
        "reason": reason,
        "deltas": {
            "factual_correctness": factual_delta,
            "citation_correctness": citation_delta,
            "temporal_attribution_correctness": temporal_delta,
            "abstention_correctness": abstention_delta,
        },
    }
    write_json(root / OUT_DIR / "final_mmr_generation_selection_decision.json", payload)
    write_md(
        root / OUT_DIR / "final_mmr_generation_selection_decision.md",
        [
            "# Final MMR Generation Selection Decision",
            "",
            f"Status: `{status}`",
            f"Best end-to-end generation setting: `{best}`",
            "",
            reason,
        ],
    )
    return payload


def write_optional_ablation_status(root: Path) -> None:
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "GEN_MMR06_SUFFICIENCY_V1_CHRONO": "skipped_to_avoid_extra_generation_cost",
        "GEN_MMR06_SUFFICIENCY_V1_RERANK_ORDER": "skipped_to_avoid_extra_generation_cost",
    }
    write_json(root / OUT_DIR / "context_ordering_ablation_status.json", payload)
    write_md(root / OUT_DIR / "context_ordering_ablation_status.md", ["# Context Ordering Ablation Status", "", "Skipped to avoid extra generation cost."])


def write_final_reports(root: Path) -> dict[str, Any]:
    comparison = read_json(root / OUT_DIR / "generation_comparison.json", {"rows": []}).get("rows", [])
    decision = read_json(root / OUT_DIR / "final_mmr_generation_selection_decision.json", {})
    classification = read_json(root / OUT_DIR / "mmr06_sufficiency_classification.json", [])
    status_counts = Counter(row.get("sufficiency_status") for row in classification)
    behavior_counts = Counter(row.get("required_generation_behavior") for row in classification)
    mmr_row = next((row for row in comparison if row.get("experiment_id") == EXPERIMENT_ID), {})
    prev_row = next((row for row in comparison if row.get("experiment_id") == "V2_COHERE_ONLY_SUFFICIENCY"), {})
    report = [
        "# Final End-to-End MMR Generation Experiment Report",
        "",
        "## Why this experiment was needed",
        "",
        "MMR improved retrieval-only development metrics, but generation had not been rerun on MMR-selected contexts. This experiment tests whether that retrieval improvement carries through to answer quality.",
        "",
        "## MMR retrieval result summary",
        "",
        f"CER={mmr_row.get('CER')}, All-Reports Hit={mmr_row.get('All-Reports Hit')}, Evidence Recall={mmr_row.get('Evidence Recall')}, Macro MRR={mmr_row.get('Macro MRR')}.",
        "",
        "## Generation setup",
        "",
        f"Experiment `{EXPERIMENT_ID}` used Groq `{DEFAULT_MODEL}`, temperature 0, prompt `{PROMPT_VERSION}`, retrieval source `{RETRIEVAL_ID}`, and the sufficiency gate.",
        "",
        "## Sufficiency classification results",
        "",
        f"Statuses: {dict(status_counts)}",
        f"Behaviours: {dict(behavior_counts)}",
        "",
        "## Generation metrics",
        "",
        "| Metric | Previous V2 sufficiency | MMR06 sufficiency |",
        "|---|---:|---:|",
        *[f"| {metric} | {prev_row.get(metric)} | {mmr_row.get(metric)} |" for metric in METRIC_NAMES],
        "",
        "## Final selection decision",
        "",
        f"Decision: `{decision.get('status')}`. Best setting: `{decision.get('best_end_to_end_generation_setting')}`.",
        "",
        "## What improved",
        "",
        "See `generation_comparison.md` for metric-level deltas.",
        "",
        "## What did not improve",
        "",
        "The selection decision records whether MMR generation did or did not replace the previous generation setting.",
        "",
        "## Remaining limitations",
        "",
        "- Development-only evaluation.",
        "- No held-out rerun or fresh evaluation set.",
        "- Deterministic heuristic generation metrics, not human evaluation.",
        "- No Unstructured/Tesseract work.",
        "",
        "## Final project claim wording",
        "",
        "Use retrieval-only and generation claims separately unless the selection decision is `selected_mmr_end_to_end`.",
    ]
    write_md(root / OUT_DIR / "final_end_to_end_experiment_report.md", report)
    ppt = [
        "# Final Results for Presentation",
        "",
        "| System | CER | Hit | Factual | Citation | Temporal | Comparative | Abstention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Previous V2 Cohere + sufficiency | {prev_row.get('CER')} | {prev_row.get('All-Reports Hit')} | {prev_row.get('factual_correctness')} | {prev_row.get('citation_correctness')} | {prev_row.get('temporal_attribution_correctness')} | {prev_row.get('comparative_correctness')} | {prev_row.get('abstention_correctness')} |",
        f"| MMR06 + sufficiency | {mmr_row.get('CER')} | {mmr_row.get('All-Reports Hit')} | {mmr_row.get('factual_correctness')} | {mmr_row.get('citation_correctness')} | {mmr_row.get('temporal_attribution_correctness')} | {mmr_row.get('comparative_correctness')} | {mmr_row.get('abstention_correctness')} |",
        "",
        f"Decision: `{decision.get('status')}`.",
    ]
    write_md(root / OUT_DIR / "final_results_for_presentation.md", ppt)
    return {"report": str(root / OUT_DIR / "final_end_to_end_experiment_report.md")}


def validate_final_mmr_generation(root: Path) -> dict[str, Any]:
    load_project_dotenv(root)
    issues: list[str] = []
    contexts = read_json(root / OUT_DIR / "mmr06_source_labelled_contexts.json", [])
    gen_rows = read_json(root / OUT_DIR / EXPERIMENT_ID / "raw_results.json", [])
    eval_rows = read_json(root / OUT_DIR / EXPERIMENT_ID / "eval_raw_results.json", [])
    coverage = read_json(root / OUT_DIR / EXPERIMENT_ID / "metric_coverage.json", {})
    for row in contexts:
        qid = row.get("question_id")
        if row.get("split") != "dev":
            issues.append(f"{qid}:non_dev_context")
        if row.get("retrieval_experiment_id") != RETRIEVAL_ID:
            issues.append(f"{qid}:wrong_retrieval_source")
        required = set(row.get("required_report_ids") or [])
        for block in row.get("context_blocks", []):
            if block.get("report_id") not in required:
                issues.append(f"{qid}:non_required_report_context:{block.get('chunk_id')}")
    context_chunks = {row["question_id"]: set(row.get("selected_chunk_ids") or []) for row in contexts}
    for row in gen_rows:
        qid = row.get("question_id")
        if row.get("retrieval_experiment_id") != RETRIEVAL_ID:
            issues.append(f"{qid}:generation_wrong_retrieval_source")
        if row.get("prompt_version") != PROMPT_VERSION:
            issues.append(f"{qid}:wrong_prompt_version")
        if row.get("model_name") != DEFAULT_MODEL or row.get("temperature") != DEFAULT_TEMPERATURE:
            issues.append(f"{qid}:wrong_model_or_temperature")
        if "required_generation_behavior" not in row:
            issues.append(f"{qid}:missing_sufficiency_behavior")
        supplied = context_chunks.get(qid, set())
        for citation in row.get("citations", []):
            if citation.get("chunk_id") not in supplied:
                issues.append(f"{qid}:citation_not_in_supplied_context:{citation.get('chunk_id')}")
    for row in eval_rows:
        for name, item in (row.get("metrics") or {}).items():
            if not item.get("success", True) and item.get("score") is not None:
                issues.append(f"{row.get('question_id')}:failed_metric_has_non_null_score:{name}")
    if not coverage:
        issues.append("metric_coverage_missing")
    all_text = {}
    for path in (root / OUT_DIR).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv", ".yaml", ".yml", ".log"}:
            all_text[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
    if actual_key_values_serialized(all_text):
        issues.append("api_key_value_serialized")
    if not any(path.name.startswith("archive_previous_run_") for path in (root / OUT_DIR).iterdir() if path.is_dir()):
        # The first ever run legitimately has no previous run to archive.
        pass
    comparison = (root / "reports/final_comparison/rag_methods_master_comparison.md").read_text(encoding="utf-8", errors="ignore") if (root / "reports/final_comparison/rag_methods_master_comparison.md").exists() else ""
    readme = (root / "README.md").read_text(encoding="utf-8", errors="ignore") if (root / "README.md").exists() else ""
    if "Mean Reciprocal Rank" not in comparison or "Maximal Marginal Relevance" not in comparison:
        issues.append("mrr_mmr_distinction_missing")
    if "fresh V2 benchmark" not in comparison:
        issues.append("heldout_caveat_missing")
    if "production-ready" in readme.lower() and "not production-ready" not in readme.lower():
        issues.append("readme_overclaims_production")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(set(issues)),
        "issues": sorted(set(issues)),
        "heldout_rows_used": False,
        "fresh_eval_created": False,
        "mmr_retrieval_outputs_dev_only": True,
        "generation_used_mmr06_contexts": not any("wrong_retrieval_source" in issue for issue in issues),
        "sufficiency_gate_applied": all("required_generation_behavior" in row for row in gen_rows) if gen_rows else False,
        "metric_coverage_recorded": bool(coverage),
        "api_key_scan_status": "passed" if "api_key_value_serialized" not in issues else "failed",
    }
    write_json(root / OUT_DIR / "final_mmr_generation_validation.json", payload)
    lines = ["# Final MMR Generation Validation", "", f"Status: `{payload['status']}`", f"Issues: {payload['issue_count']}"]
    if payload["issues"]:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    write_md(root / OUT_DIR / "final_mmr_generation_validation.md", lines)
    return payload


def prepare_final_mmr_generation(root: Path) -> dict[str, Any]:
    archive = archive_existing_output(root)
    checksums = write_pre_checksums(root)
    validation = validate_inputs(root)
    return {"archive": archive, "checksums": checksums, "input_validation": validation}


def run_final_mmr_generation(root: Path = Path("."), *, generator: Any | None = None) -> dict[str, Any]:
    prep = prepare_final_mmr_generation(root)
    if prep["input_validation"]["status"] != "passed":
        return {"status": "blocked_missing_mmr_inputs", "input_validation": prep["input_validation"]}
    contexts = build_mmr_contexts(root)
    classifications = run_sufficiency_classification(root, contexts)
    result = run_generation(root, generator=generator)
    return {
        "status": result["status"],
        "contexts": len(contexts),
        "classifications": len(classifications),
        "generation": result,
    }


def evaluate_final_mmr_generation(root: Path = Path(".")) -> dict[str, Any]:
    eval_summary = evaluate_generation(root)
    comparison_rows, delta = write_generation_comparison(root)
    decision = write_selection_decision(root, comparison_rows)
    write_optional_ablation_status(root)
    return {"status": "completed" if eval_summary.get("metrics") else "blocked", "delta": delta, "decision": decision}


def generate_final_mmr_generation_report(root: Path = Path(".")) -> dict[str, Any]:
    reports = write_final_reports(root)
    validation = validate_final_mmr_generation(root)
    return {"status": validation["status"], "reports": reports, "validation": validation}
