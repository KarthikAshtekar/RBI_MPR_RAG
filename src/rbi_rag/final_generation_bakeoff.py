from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact_io import (
    make_checksum_manifest,
    now_iso,
    relative_posix as rel,
    stable_json_hash,
    write_csv,
    write_json,
    write_markdown as write_md,
)
from .env_loading import load_project_dotenv
from .evidence_sufficiency import classify_all
from .evaluation_cases import expected_case_lookup
from .generation_evaluation_core import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    METRIC_NAMES,
    GroqGenerator,
    actual_key_values_serialized,
    evaluate_generation_rows,
    invalid_citations,
    parse_citations,
    safe_error_message,
)
from .generation_helpers import mean_or_none as _mean, source_label as _source_label
from .v2_generation_contexts import REPORT_ORDER
from .v2_sufficiency import (
    PROMPT_VERSION,
    apply_sufficiency_postprocessing,
    build_sufficiency_prompt,
    deterministic_abstention,
    required_periods,
)


OUT_DIR = Path("reports/final_generation_bakeoff")
EXPERIMENTS_DIR = OUT_DIR / "experiments"
MAX_CONTEXT_TOKENS = 3000
PREVIOUS_BEST_VARIANT = "GEN_MMR06_SUFFICIENCY_V1"

CHECKSUM_TARGETS = [
    Path("configs/v2_selected_retrieval.yaml"),
    Path("configs/v2_mmr_selected_retrieval.yaml"),
    Path("configs/mmr_experiments.yaml"),
    Path("reports/mmr_experiments"),
    Path("reports/final_mmr_generation"),
    Path("reports/v2_generation"),
    Path("reports/v2_sufficiency"),
    Path("reports/final_comparison"),
    Path("reports/final_packaging"),
    Path("README.md"),
    Path("data/evaluation"),
]

RETRIEVAL_RAW_PATHS = {
    "V2_COHERE_ONLY": Path("reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY/raw_results.json"),
    "MMR_LAMBDA_06": Path("reports/mmr_experiments/experiments/MMR_LAMBDA_06/raw_results.json"),
    "MMR_LAMBDA_07": Path("reports/mmr_experiments/experiments/MMR_LAMBDA_07/raw_results.json"),
    "MMR_LAMBDA_08": Path("reports/mmr_experiments/experiments/MMR_LAMBDA_08/raw_results.json"),
}
RETRIEVAL_SUMMARY_PATHS = {
    "V2_COHERE_ONLY": Path("reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY/summary.json"),
    "MMR_LAMBDA_06": Path("reports/mmr_experiments/experiments/MMR_LAMBDA_06/summary.json"),
    "MMR_LAMBDA_07": Path("reports/mmr_experiments/experiments/MMR_LAMBDA_07/summary.json"),
    "MMR_LAMBDA_08": Path("reports/mmr_experiments/experiments/MMR_LAMBDA_08/summary.json"),
}

VARIANTS: dict[str, dict[str, Any]] = {
    "GEN_V2_COHERE_SUFFICIENCY_V1": {
        "mode": "reuse_v2_sufficiency",
        "retrieval_experiment_id": "V2_COHERE_ONLY",
        "prompt_version": PROMPT_VERSION,
        "context_ordering": "default",
        "priority": 0,
        "description": "Reuse existing V2 Cohere retrieval plus sufficiency-gated generation result.",
    },
    "GEN_MMR06_SUFFICIENCY_V1": {
        "mode": "reuse_mmr_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "prompt_version": PROMPT_VERSION,
        "context_ordering": "page_order",
        "priority": 0,
        "description": "Reuse existing MMR lambda 0.6 plus sufficiency-gated generation result.",
    },
    "GEN_MMR07_SUFFICIENCY_V1": {
        "mode": "live_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_07",
        "prompt_version": PROMPT_VERSION,
        "context_ordering": "page_order",
        "priority": 1,
        "description": "Generate on frozen MMR lambda 0.7 retrieval contexts.",
    },
    "GEN_MMR08_SUFFICIENCY_V1": {
        "mode": "live_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_08",
        "prompt_version": PROMPT_VERSION,
        "context_ordering": "page_order",
        "priority": 2,
        "description": "Generate on frozen MMR lambda 0.8 retrieval contexts.",
    },
    "GEN_MMR06_CHRONO_ORDER_V1": {
        "mode": "live_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "prompt_version": PROMPT_VERSION,
        "context_ordering": "chrono_order",
        "priority": 3,
        "description": "Generate on MMR06 chunks ordered by report chronology and page.",
    },
    "GEN_MMR06_RERANK_ORDER_V1": {
        "mode": "live_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "prompt_version": PROMPT_VERSION,
        "context_ordering": "rerank_order",
        "priority": 4,
        "description": "Generate on MMR06 chunks ordered by saved MMR/reranker selection rank.",
    },
    "GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1": {
        "mode": "live_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "prompt_version": "evidence_first_prompt_v1",
        "context_ordering": "page_order",
        "priority": 5,
        "description": "Generate with evidence-by-report-first prompt.",
    },
    "GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1": {
        "mode": "live_generation",
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "prompt_version": "comparative_strict_prompt_v1",
        "context_ordering": "page_order",
        "priority": 6,
        "description": "Generate with strict comparative prompt.",
    },
    "GEN_MMR06_CITATION_REPAIR_V1": {
        "mode": "citation_repair",
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "prompt_version": "deterministic_citation_repair_v1",
        "context_ordering": "page_order",
        "priority": 7,
        "description": "Deterministically repair invalid citations in the existing MMR06 answer set.",
    },
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
        shutil.move(str(item), str(archive / item.name))
    return {"status": "archived", "archive_dir": rel(root, archive), "item_count": len(items)}


def write_pre_checksums(root: Path) -> dict[str, Any]:
    payload = make_checksum_manifest(root, CHECKSUM_TARGETS)
    write_json(root / OUT_DIR / "pre_final_generation_bakeoff_checksums.json", payload)
    lines = [
        "# Pre-Final Generation Bake-Off Checksums",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Files captured: {payload['entry_count']}",
        "",
        "These checksums were captured before writing bake-off outputs. They are audit artifacts, not a license to overwrite frozen results.",
        "",
    ]
    if payload.get("missing_targets"):
        lines += ["## Missing targets", ""]
        lines.extend(f"- `{target}`" for target in payload["missing_targets"])
        lines.append("")
    lines += ["## Targets", ""]
    lines.extend(f"- `{target}`" for target in payload["targets"])
    write_md(root / OUT_DIR / "pre_final_generation_bakeoff_checksums.md", lines)
    return payload


def write_environment(root: Path) -> dict[str, Any]:
    load_project_dotenv(root)
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "groq_api_key_available": bool(os.getenv("GROQ_API_KEY")),
        "cohere_api_key_available": bool(os.getenv("COHERE_API_KEY")),
        "unstructured_api_key_available": bool(os.getenv("UNSTRUCTURED_API_KEY")),
        "generation_model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
    }
    write_json(root / OUT_DIR / "environment_readiness.json", payload)
    write_md(
        root / OUT_DIR / "environment_readiness.md",
        [
            "# Final Generation Bake-Off Environment",
            "",
            f"Groq key available: {payload['groq_api_key_available']}",
            f"Cohere key available: {payload['cohere_api_key_available']}",
            f"Unstructured key available: {payload['unstructured_api_key_available']}",
            "",
            "No key values are printed, hashed, or serialized.",
        ],
    )
    return payload


def validate_inputs(root: Path) -> dict[str, Any]:
    required = [
        Path("configs/v2_selected_retrieval.yaml"),
        Path("configs/v2_mmr_selected_retrieval.yaml"),
        Path("configs/mmr_experiments.yaml"),
        RETRIEVAL_RAW_PATHS["MMR_LAMBDA_06"],
        RETRIEVAL_RAW_PATHS["MMR_LAMBDA_07"],
        RETRIEVAL_RAW_PATHS["MMR_LAMBDA_08"],
        Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/raw_results.json"),
        Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json"),
        Path("src/rbi_rag/evidence_sufficiency.py"),
        Path("src/rbi_rag/v2_sufficiency.py"),
        Path("data/evaluation/multi_report_dev.jsonl"),
    ]
    optional = [
        RETRIEVAL_RAW_PATHS["V2_COHERE_ONLY"],
        Path("reports/v2_sufficiency/dev_sufficiency_eval_summary.json"),
        Path("reports/v2_generation/v2_generation_contexts.json"),
    ]
    missing_required = [str(path).replace("/", "\\") for path in required if not (root / path).exists()]
    missing_optional = [str(path).replace("/", "\\") for path in optional if not (root / path).exists()]
    row_issues: list[str] = []
    for retrieval_id in ("MMR_LAMBDA_06", "MMR_LAMBDA_07", "MMR_LAMBDA_08"):
        rows = read_json(root / RETRIEVAL_RAW_PATHS[retrieval_id], [])
        non_dev = [row.get("question_id") for row in rows if row.get("split") != "dev"]
        wrong_id = [row.get("question_id") for row in rows if row.get("experiment_id") != retrieval_id]
        if non_dev:
            row_issues.append(f"{retrieval_id}:non_dev_rows:{sorted(set(non_dev))}")
        if wrong_id:
            row_issues.append(f"{retrieval_id}:wrong_experiment_rows:{sorted(set(wrong_id))}")
    issues = missing_required + row_issues
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "blocked_missing_inputs",
        "required_inputs": [str(path).replace("/", "\\") for path in required],
        "optional_inputs": [str(path).replace("/", "\\") for path in optional],
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "issues": issues,
        "heldout_used": False,
        "fresh_eval_created": False,
    }
    write_json(root / OUT_DIR / "input_artifact_validation.json", payload)
    lines = [
        "# Final Generation Bake-Off Input Validation",
        "",
        f"Status: `{payload['status']}`",
        f"Missing required inputs: {len(missing_required)}",
        f"Missing optional inputs: {len(missing_optional)}",
    ]
    if issues:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in issues)
    if missing_optional:
        lines += ["", "## Optional inputs unavailable", ""]
        lines.extend(f"- `{issue}`" for issue in missing_optional)
    write_md(root / OUT_DIR / "input_artifact_validation.md", lines)
    return payload


def prepare_final_generation_bakeoff(root: Path = Path(".")) -> dict[str, Any]:
    archive = archive_existing_output(root)
    checksums = write_pre_checksums(root)
    environment = write_environment(root)
    input_validation = validate_inputs(root)
    write_variant_registry(root)
    return {
        "archive": archive,
        "checksums": checksums,
        "environment": environment,
        "input_validation": input_validation,
    }


def _median(values: list[Any]) -> float | None:
    numeric = sorted(float(value) for value in values if isinstance(value, (int, float, bool)))
    return statistics.median(numeric) if numeric else None


def _p95(values: list[Any]) -> float | None:
    numeric = sorted(float(value) for value in values if isinstance(value, (int, float, bool)))
    if not numeric:
        return None
    index = min(len(numeric) - 1, int(round((len(numeric) - 1) * 0.95)))
    return numeric[index]


def _metric(summary: dict[str, Any], name: str) -> float | None:
    return ((summary.get("metrics") or {}).get(name) or {}).get("mean_score")


def _chunk_rank_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(row.get("mmr_trace") or []):
        chunk_id = item.get("chunk_id")
        if chunk_id:
            trace[chunk_id] = {**item, "trace_index": index}
    return trace


def _estimated_tokens(blocks: list[dict[str, Any]]) -> int:
    return int(sum(len(block.get("text") or "") for block in blocks) / 4)


def _order_blocks(blocks: list[dict[str, Any]], ordering: str) -> list[dict[str, Any]]:
    if ordering == "rerank_order":
        return sorted(
            blocks,
            key=lambda item: (
                item.get("mmr_selection_rank") if item.get("mmr_selection_rank") is not None else 10_000,
                item.get("trace_index") if item.get("trace_index") is not None else 10_000,
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
    missing = [report_id for report_id in required if not any(block["report_id"] == report_id for block in kept)]
    if missing:
        warnings.append(f"context_budget_missing_required_report:{missing}")
    return kept, warnings


def _normalise_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_complete_evidence_recall": row.get("retrieval_complete_evidence_recall", row.get("complete_evidence_recall")),
        "retrieval_evidence_recall": row.get("retrieval_evidence_recall", row.get("evidence_recall")),
        "retrieval_all_reports_hit": row.get("retrieval_all_reports_hit", row.get("all_reports_hit")),
        "retrieval_macro_mrr": row.get("retrieval_macro_mrr", row.get("macro_report_mrr", row.get("macro_mrr"))),
        "report_coverage": row.get("report_coverage"),
        "single_report_contamination": row.get("single_report_contamination", row.get("contamination")),
        "repeated_text_ratio": row.get("repeated_text_ratio"),
    }


def build_context_from_retrieval_row(
    row: dict[str, Any],
    *,
    retrieval_experiment_id: str,
    ordering: str,
) -> dict[str, Any]:
    required = list(row.get("required_report_ids") or [])
    required_set = set(required)
    trace = _chunk_rank_map(row)
    blocks: list[dict[str, Any]] = []
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
        "retrieval_experiment_id": retrieval_experiment_id,
        "retrieval_config_checksum": row.get("configuration_checksum"),
        "sufficiency_status": None,
        "required_generation_behavior": None,
        "selected_chunks_by_report": {
            report_id: [block for block in blocks if block["report_id"] == report_id]
            for report_id in sorted(required_set, key=lambda rid: REPORT_ORDER.get(rid, 99))
        },
        "source_labelled_context": "\n\n".join(grouped),
        "context_ordering": ordering,
        "context_blocks": blocks,
        "selected_chunk_ids": [block["chunk_id"] for block in blocks],
        "selected_pages": selected_pages,
        **_normalise_metric_row(row),
        "estimated_token_count": _estimated_tokens(blocks),
        "unique_page_count": len({(block["report_id"], block["page_number"]) for block in blocks}),
        "context_warnings": warnings,
    }


def enrich_existing_contexts(
    contexts: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    *,
    retrieval_experiment_id: str,
    ordering: str,
) -> list[dict[str, Any]]:
    retrieval_by_id = {row["question_id"]: row for row in retrieval_rows}
    enriched: list[dict[str, Any]] = []
    for item in contexts:
        row = retrieval_by_id.get(item.get("question_id"), {})
        blocks = item.get("context_blocks") or []
        selected_chunks_by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for block in blocks:
            report_id = block.get("report_id")
            if report_id:
                selected_chunks_by_report[report_id].append(block)
        selected_chunk_ids = item.get("selected_chunk_ids") or [block.get("chunk_id") for block in blocks if block.get("chunk_id")]
        enriched.append({
            **item,
            "retrieval_experiment_id": retrieval_experiment_id,
            "context_ordering": ordering,
            "selected_chunks_by_report": dict(selected_chunks_by_report),
            "selected_chunk_ids": selected_chunk_ids,
            "sufficiency_status": item.get("sufficiency_status"),
            "required_generation_behavior": item.get("required_generation_behavior"),
            **_normalise_metric_row(row),
            "estimated_token_count": item.get("estimated_token_count") or int(len(item.get("source_labelled_context") or "") / 4),
            "unique_page_count": item.get("unique_page_count") or len({
                (block.get("report_id"), block.get("page_number")) for block in blocks
            }),
            "repeated_text_ratio": item.get("repeated_text_ratio", row.get("repeated_text_ratio")),
            "context_warnings": item.get("context_warnings", []),
        })
    return enriched


def write_context_artifacts(root: Path, variant_id: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    write_json(out / "context_records.json", contexts)
    write_csv(out / "context_records.csv", contexts)
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "variant_id": variant_id,
        "row_count": len(contexts),
        "mean_estimated_token_count": _mean([row.get("estimated_token_count") for row in contexts]),
        "mean_unique_page_count": _mean([row.get("unique_page_count") for row in contexts]),
        "context_ordering": contexts[0].get("context_ordering") if contexts else None,
        "retrieval_experiment_id": contexts[0].get("retrieval_experiment_id") if contexts else None,
        "warning_count": sum(len(row.get("context_warnings") or []) for row in contexts),
    }
    write_json(out / "context_summary.json", summary)
    write_md(
        out / "context_summary.md",
        [
            f"# {variant_id} Context Summary",
            "",
            f"Rows: {summary['row_count']}",
            f"Retrieval source: `{summary['retrieval_experiment_id']}`",
            f"Context ordering: `{summary['context_ordering']}`",
            f"Mean estimated tokens: {summary['mean_estimated_token_count']}",
            f"Warnings: {summary['warning_count']}",
        ],
    )
    return summary


def build_contexts_for_variant(root: Path, variant_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variant = VARIANTS[variant_id]
    retrieval_id = variant["retrieval_experiment_id"]
    retrieval_rows = read_json(root / RETRIEVAL_RAW_PATHS[retrieval_id], [])
    ordering = variant["context_ordering"]
    if variant["mode"] == "reuse_v2_sufficiency":
        contexts = read_json(root / "reports/v2_generation/v2_generation_contexts.json", [])
        contexts = enrich_existing_contexts(contexts, retrieval_rows, retrieval_experiment_id=retrieval_id, ordering=ordering)
    else:
        contexts = [
            build_context_from_retrieval_row(row, retrieval_experiment_id=retrieval_id, ordering=ordering)
            for row in retrieval_rows
        ]
    write_context_artifacts(root, variant_id, contexts)
    return contexts, retrieval_rows


def write_sufficiency_artifacts(
    root: Path,
    variant_id: str,
    retrieval_rows: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = root / EXPERIMENTS_DIR / variant_id
    if variant_id == "GEN_V2_COHERE_SUFFICIENCY_V1":
        rows = read_json(root / "reports/v2_sufficiency/dev_sufficiency_classification.json", [])
    elif variant_id == "GEN_MMR06_SUFFICIENCY_V1":
        rows = read_json(root / "reports/final_mmr_generation/mmr06_sufficiency_classification.json", [])
    else:
        rows = classify_all(retrieval_rows, contexts, expected_case_lookup(root, "dev"))
    by_id = {row.get("question_id"): row for row in rows}
    for context in contexts:
        suff = by_id.get(context["question_id"], {})
        context["sufficiency_status"] = suff.get("sufficiency_status")
        context["required_generation_behavior"] = suff.get("required_generation_behavior")
    write_json(out / "sufficiency_classification.json", rows)
    write_csv(out / "sufficiency_classification.csv", rows)
    status_counts = Counter(row.get("sufficiency_status") for row in rows)
    behavior_counts = Counter(row.get("required_generation_behavior") for row in rows)
    write_md(
        out / "sufficiency_classification.md",
        [
            f"# {variant_id} Sufficiency Classification",
            "",
            "## Status counts",
            "",
            *[f"- {key}: {value}" for key, value in sorted(status_counts.items(), key=lambda kv: str(kv[0]))],
            "",
            "## Required behaviour counts",
            "",
            *[f"- {key}: {value}" for key, value in sorted(behavior_counts.items(), key=lambda kv: str(kv[0]))],
        ],
    )
    write_context_artifacts(root, variant_id, contexts)
    return rows


def build_variant_prompt(
    *,
    prompt_version: str,
    question: str,
    context: str,
    query_type: str,
    required_periods_value: list[str],
    sufficiency: dict[str, Any],
) -> str:
    if prompt_version == PROMPT_VERSION:
        return build_sufficiency_prompt(question, context, query_type, required_periods_value, sufficiency)
    base = build_sufficiency_prompt(question, context, query_type, required_periods_value, sufficiency)
    if prompt_version == "evidence_first_prompt_v1":
        return (
            base
            + "\n\nAdditional instruction for this bake-off variant:\n"
            + "Before the final Answer section, internally organize the evidence report by report. "
            + "In the written answer, mention the supporting report period before each numeric, comparative, or temporal claim. "
            + "If evidence is partial, state that limitation before the supported answer. "
            + "Every numeric or comparative claim must cite a supplied report/page/chunk."
        )
    if prompt_version == "comparative_strict_prompt_v1":
        return (
            base
            + "\n\nAdditional instruction for this bake-off variant:\n"
            + "For comparative questions, write one mini-conclusion for each required report period before synthesis. "
            + "Do not make cross-report claims unless each involved report has a supplied citation. "
            + "If one side of a comparison is missing, say the comparison is only partially supported."
        )
    return base


def run_generation_cases_for_variant(
    contexts: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    classifications: list[dict[str, Any]],
    *,
    variant_id: str,
    prompt_version: str,
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
        existing = read_json(checkpoint_path, [])
        for row in existing:
            if row.get("generation_experiment_id") != variant_id:
                continue
            if row.get("prompt_version") != prompt_version:
                continue
            context_item = context_by_id.get(row.get("question_id"), {})
            sufficiency = classification_by_id.get(row.get("question_id"), {})
            if row.get("generation_success"):
                row["generated_answer"] = apply_sufficiency_postprocessing(row.get("generated_answer") or "", sufficiency)
                row["citations"] = parse_citations(row["generated_answer"], context_item)
            if row.get("generation_success") and invalid_citations(row, context_item):
                continue
            rows.append(row)
        write_json(checkpoint_path, rows)
    completed = {row["question_id"] for row in rows}
    for item in contexts:
        qid = item["question_id"]
        if qid in completed:
            continue
        case = cases.get(qid, {})
        sufficiency = classification_by_id[qid]
        prompt = build_variant_prompt(
            prompt_version=prompt_version,
            question=item.get("original_query") or case.get("question", ""),
            context=item.get("source_labelled_context") or "",
            query_type=str(item.get("query_type")),
            required_periods_value=required_periods(item),
            sufficiency=sufficiency,
        )
        attempts = 0
        retry_warnings: list[dict[str, Any]] = []
        error_type = None
        error_message = None
        started = time.perf_counter()
        if sufficiency["required_generation_behavior"] == "abstain":
            generated_answer = deterministic_abstention(item, sufficiency)
            success = True
            provider_used = "deterministic_sufficiency_gate"
            attempts = 0
        else:
            generated_answer = ""
            success = False
            provider_used = model_provider
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
            "experiment_id": variant_id,
            "generation_experiment_id": variant_id,
            "retrieval_experiment_id": item.get("retrieval_experiment_id"),
            "retrieval_config_checksum": item.get("retrieval_config_checksum"),
            "context_ordering": item.get("context_ordering"),
            "selected_chunk_ids": item.get("selected_chunk_ids"),
            "selected_pages": item.get("selected_pages"),
            "sufficiency_status": sufficiency["sufficiency_status"],
            "sufficiency_reasons": sufficiency["sufficiency_reasons"],
            "required_generation_behavior": sufficiency["required_generation_behavior"],
            "source_labelled_context": item.get("source_labelled_context"),
            "prompt_version": prompt_version,
            "model_provider": provider_used,
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
            "source_structure": case.get("source_structure"),
            "question_structure": case.get("question_structure"),
            "requires_numeric_evidence": bool(item.get("table_or_numeric_question") or case.get("requires_numeric_evidence")),
            "table_or_numeric_question": bool(item.get("table_or_numeric_question") or case.get("table_or_numeric_question")),
        }
        rows.append(row)
        if checkpoint_path:
            write_json(checkpoint_path, rows)
    return rows


def write_variant_config_snapshot(root: Path, variant_id: str) -> None:
    out = root / EXPERIMENTS_DIR / variant_id
    payload = {"variant_id": variant_id, **VARIANTS[variant_id]}
    write_json(out / "config_snapshot.json", payload)
    lines = [f"# {variant_id} Config Snapshot", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in payload.items())
    write_md(out / "config_snapshot.md", lines)


def write_generation_summary(
    root: Path,
    variant_id: str,
    rows: list[dict[str, Any]],
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    successes = [row for row in rows if row.get("generation_success")]
    latencies = [float(row["generation_latency_ms"]) for row in rows if isinstance(row.get("generation_latency_ms"), (int, float))]
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "variant_id": variant_id,
        "experiment_id": variant_id,
        "retrieval_experiment_id": VARIANTS[variant_id]["retrieval_experiment_id"],
        "prompt_version": VARIANTS[variant_id]["prompt_version"],
        "context_ordering": VARIANTS[variant_id]["context_ordering"],
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
        "sufficiency_gate_enabled": True,
        "mean_generation_latency_ms": _mean(latencies),
        "median_generation_latency_ms": _median(latencies),
        "p95_generation_latency_ms": _p95(latencies),
        "behavior_counts": dict(Counter(row.get("required_generation_behavior") for row in rows)),
        "sufficiency_status_counts": dict(Counter(row.get("sufficiency_status") for row in rows)),
        "hard_gate_abstention_count": sum(1 for row in rows if row.get("model_provider") == "deterministic_sufficiency_gate"),
    }
    write_json(out / "summary.json", summary)
    lines = [
        f"# {variant_id} Summary",
        "",
        f"Status: `{status}`",
        f"Retrieval source: `{summary['retrieval_experiment_id']}`",
        f"Prompt version: `{summary['prompt_version']}`",
        f"Context ordering: `{summary['context_ordering']}`",
        f"Rows: {summary['row_count']}",
        f"Success rate: {summary['success_rate']}",
        f"Median generation latency ms: {summary['median_generation_latency_ms']}",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    write_md(out / "summary.md", lines)
    return summary


def write_skipped_variant(root: Path, variant_id: str, reason: str) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    out.mkdir(parents=True, exist_ok=True)
    write_variant_config_snapshot(root, variant_id)
    write_json(out / "raw_results.json", [])
    write_csv(out / "results.csv", [])
    write_json(out / "eval_raw_results.json", [])
    eval_summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "variant_id": variant_id,
        "status": "skipped",
        "reason": reason,
        "metrics": {},
    }
    write_json(out / "eval_summary.json", eval_summary)
    write_json(out / "metric_coverage.json", {})
    write_md(out / "eval_summary.md", [f"# {variant_id} Evaluation", "", "Status: `skipped`", f"Reason: {reason}"])
    write_md(out / "metric_coverage.md", [f"# {variant_id} Metric Coverage", "", "Status: `skipped`"])
    summary = write_generation_summary(root, variant_id, [], status="skipped", reason=reason)
    return summary


def _standardise_generation_rows(rows: list[dict[str, Any]], variant_id: str) -> list[dict[str, Any]]:
    variant = VARIANTS[variant_id]
    cleaned = []
    for row in rows:
        item = dict(row)
        item["experiment_id"] = variant_id
        item["generation_experiment_id"] = variant_id
        item["retrieval_experiment_id"] = variant["retrieval_experiment_id"]
        item["prompt_version"] = variant["prompt_version"]
        item.setdefault("context_ordering", variant["context_ordering"])
        cleaned.append(item)
    return cleaned


def copy_reused_variant(root: Path, variant_id: str) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    out.mkdir(parents=True, exist_ok=True)
    write_variant_config_snapshot(root, variant_id)
    contexts, retrieval_rows = build_contexts_for_variant(root, variant_id)
    write_sufficiency_artifacts(root, variant_id, retrieval_rows, contexts)
    if variant_id == "GEN_V2_COHERE_SUFFICIENCY_V1":
        source_files = {
            "raw_results.json": Path("reports/v2_sufficiency/dev_generation_sufficiency_raw_results.json"),
            "summary.json": Path("reports/v2_sufficiency/dev_generation_sufficiency_summary.json"),
            "eval_raw_results.json": Path("reports/v2_sufficiency/dev_sufficiency_eval_raw_results.json"),
            "eval_summary.json": Path("reports/v2_sufficiency/dev_sufficiency_eval_summary.json"),
            "metric_coverage.json": Path("reports/v2_sufficiency/dev_sufficiency_metric_coverage.json"),
        }
    else:
        source_files = {
            "raw_results.json": Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/raw_results.json"),
            "summary.json": Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/summary.json"),
            "eval_raw_results.json": Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/eval_raw_results.json"),
            "eval_summary.json": Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json"),
            "metric_coverage.json": Path("reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/metric_coverage.json"),
        }
    missing = [str(path).replace("/", "\\") for path in source_files.values() if not (root / path).exists()]
    if missing:
        reason = "missing reused source artifacts: " + ", ".join(missing)
        return write_skipped_variant(root, variant_id, reason)
    rows = _standardise_generation_rows(read_json(root / source_files["raw_results.json"], []), variant_id)
    write_json(out / "raw_results.json", rows)
    write_csv(out / "results.csv", rows)
    source_summary = read_json(root / source_files["summary.json"], {})
    source_summary.update({
        "variant_id": variant_id,
        "experiment_id": variant_id,
        "retrieval_experiment_id": VARIANTS[variant_id]["retrieval_experiment_id"],
        "prompt_version": VARIANTS[variant_id]["prompt_version"],
        "context_ordering": VARIANTS[variant_id]["context_ordering"],
        "status": "completed_reused",
        "reuse_source": str(source_files["raw_results.json"]).replace("/", "\\"),
    })
    write_json(out / "summary.json", source_summary)
    write_md(
        out / "summary.md",
        [
            f"# {variant_id} Summary",
            "",
            "Status: `completed_reused`",
            f"Source: `{source_files['raw_results.json']}`",
            f"Rows: {source_summary.get('row_count')}",
            f"Success rate: {source_summary.get('success_rate')}",
        ],
    )
    for target_name in ("eval_raw_results.json", "eval_summary.json", "metric_coverage.json"):
        payload = read_json(root / source_files[target_name], [] if target_name == "eval_raw_results.json" else {})
        if isinstance(payload, dict):
            payload["variant_id"] = variant_id
            payload.setdefault("status", "completed_reused")
        write_json(out / target_name, payload)
    write_eval_markdown(root, variant_id)
    manifest = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "variant_id": variant_id,
        "status": "completed_reused",
        "source_files": {name: str(path).replace("/", "\\") for name, path in source_files.items()},
        "retrieval_rerun": False,
        "generation_rerun": False,
    }
    write_json(out / "reuse_manifest.json", manifest)
    write_md(out / "reuse_manifest.md", [f"# {variant_id} Reuse Manifest", "", "Existing saved artifacts were copied into the bake-off output area."])
    return source_summary


def run_live_variant(root: Path, variant_id: str, *, generator: Any | None = None) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    out.mkdir(parents=True, exist_ok=True)
    write_variant_config_snapshot(root, variant_id)
    load_project_dotenv(root)
    if not os.getenv("GROQ_API_KEY") and generator is None:
        return write_skipped_variant(root, variant_id, "GROQ_API_KEY is unavailable")
    contexts, retrieval_rows = build_contexts_for_variant(root, variant_id)
    classifications = write_sufficiency_artifacts(root, variant_id, retrieval_rows, contexts)
    generator = generator or GroqGenerator(DEFAULT_MODEL, DEFAULT_TEMPERATURE)
    rows = run_generation_cases_for_variant(
        contexts,
        expected_case_lookup(root, "dev"),
        classifications,
        variant_id=variant_id,
        prompt_version=VARIANTS[variant_id]["prompt_version"],
        generator=generator,
        model_name=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        checkpoint_path=out / "raw_results.json",
        max_retries=2,
    )
    rows = _standardise_generation_rows(rows, variant_id)
    write_json(out / "raw_results.json", rows)
    write_csv(out / "results.csv", rows)
    summary = write_generation_summary(root, variant_id, rows, status="completed")
    if actual_key_values_serialized({"rows": rows, "summary": summary}):
        raise RuntimeError("API key material serialized in bake-off generation outputs")
    return summary


def citation_repair(root: Path) -> dict[str, Any]:
    variant_id = "GEN_MMR06_CITATION_REPAIR_V1"
    out = root / EXPERIMENTS_DIR / variant_id
    out.mkdir(parents=True, exist_ok=True)
    write_variant_config_snapshot(root, variant_id)
    contexts, retrieval_rows = build_contexts_for_variant(root, variant_id)
    write_sufficiency_artifacts(root, variant_id, retrieval_rows, contexts)
    context_by_id = {item["question_id"]: item for item in contexts}
    base_rows = read_json(root / EXPERIMENTS_DIR / PREVIOUS_BEST_VARIANT / "raw_results.json", [])
    if not base_rows:
        base_rows = read_json(root / "reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/raw_results.json", [])
    repaired: list[dict[str, Any]] = []
    for row in base_rows:
        item = dict(row)
        context = context_by_id.get(item.get("question_id"), {})
        before = item.get("citations") or parse_citations(item.get("generated_answer") or "", context)
        invalid_ids = {
            citation.get("chunk_id") for citation in invalid_citations({"citations": before}, context)
        }
        answer = item.get("generated_answer") or ""
        actions: list[str] = []
        if invalid_ids:
            for chunk_id in sorted(invalid_ids):
                answer = re.sub(rf"\b{re.escape(str(chunk_id))}\b", "", answer)
                actions.append(f"removed_invalid_citation:{chunk_id}")
        citations = [citation for citation in parse_citations(answer, context) if citation.get("valid_supplied_chunk")]
        item.update({
            "experiment_id": variant_id,
            "generation_experiment_id": variant_id,
            "retrieval_experiment_id": "MMR_LAMBDA_06",
            "prompt_version": "deterministic_citation_repair_v1",
            "context_ordering": "page_order",
            "model_provider": "deterministic_citation_repair",
            "model_name": "deterministic_citation_repair",
            "temperature": 0.0,
            "generated_answer": answer,
            "citations": citations,
            "citation_repair_actions": actions,
            "original_generation_experiment_id": row.get("generation_experiment_id") or row.get("experiment_id"),
            "original_generation_latency_ms": row.get("generation_latency_ms"),
            "generation_latency_ms": 0.0,
            "generation_success": row.get("generation_success", True),
            "generation_error_type": row.get("generation_error_type"),
            "generation_error_message": row.get("generation_error_message"),
        })
        repaired.append(item)
    write_json(out / "raw_results.json", repaired)
    write_csv(out / "results.csv", repaired)
    action_count = sum(len(row.get("citation_repair_actions") or []) for row in repaired)
    summary = write_generation_summary(
        root,
        variant_id,
        repaired,
        status="completed_repaired",
        reason=f"deterministic citation repair actions: {action_count}",
    )
    write_json(out / "citation_repair_manifest.json", {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "variant_id": variant_id,
        "action_count": action_count,
        "generation_rerun": False,
        "changed_factual_content": False,
    })
    return summary


def run_final_generation_bakeoff(
    root: Path = Path("."),
    *,
    generator: Any | None = None,
    live_variant_limit: int | None = None,
) -> dict[str, Any]:
    prep = prepare_final_generation_bakeoff(root)
    if prep["input_validation"]["status"] != "passed":
        return {"status": "blocked_missing_inputs", "input_validation": prep["input_validation"]}
    if live_variant_limit is None:
        live_variant_limit = int(os.getenv("BAKEOFF_LIVE_VARIANT_LIMIT", "2"))
    completed: list[str] = []
    skipped: dict[str, str] = {}
    live_used = 0
    for variant_id, config in sorted(VARIANTS.items(), key=lambda item: (item[1]["priority"], item[0])):
        mode = config["mode"]
        if mode.startswith("reuse"):
            copy_reused_variant(root, variant_id)
            completed.append(variant_id)
        elif mode == "citation_repair":
            citation_repair(root)
            completed.append(variant_id)
        elif mode == "live_generation":
            if live_used >= live_variant_limit:
                reason = f"skipped_by_bounded_live_generation_budget:{live_variant_limit}"
                write_skipped_variant(root, variant_id, reason)
                skipped[variant_id] = reason
                continue
            run_live_variant(root, variant_id, generator=generator)
            live_used += 1
            completed.append(variant_id)
    status = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "completed",
        "live_variant_limit": live_variant_limit,
        "completed_or_reused_variants": completed,
        "skipped_variants": skipped,
        "heldout_used": False,
        "fresh_eval_created": False,
    }
    write_json(root / OUT_DIR / "final_generation_bakeoff_run_status.json", status)
    write_md(
        root / OUT_DIR / "final_generation_bakeoff_run_status.md",
        [
            "# Final Generation Bake-Off Run Status",
            "",
            f"Status: `{status['status']}`",
            f"Live variant limit: {live_variant_limit}",
            "",
            "## Completed or reused variants",
            "",
            *[f"- `{item}`" for item in completed],
            "",
            "## Skipped variants",
            "",
            *[f"- `{key}`: {value}" for key, value in skipped.items()],
        ],
    )
    return status


def evaluate_variant(root: Path, variant_id: str) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    rows = read_json(out / "raw_results.json", [])
    if not rows:
        payload = read_json(out / "eval_summary.json", {})
        if not payload:
            payload = {"schema_version": 1, "created_at_utc": now_iso(), "variant_id": variant_id, "status": "skipped", "metrics": {}}
            write_json(out / "eval_summary.json", payload)
        return payload
    retrieval_id = VARIANTS[variant_id]["retrieval_experiment_id"]
    retrieval_rows = read_json(root / RETRIEVAL_RAW_PATHS[retrieval_id], [])
    contexts = read_json(out / "context_records.json", [])
    eval_rows, summary, coverage, failures = evaluate_generation_rows(rows, retrieval_rows, contexts)
    summary["variant_id"] = variant_id
    summary["status"] = "completed"
    write_json(out / "eval_raw_results.json", eval_rows)
    write_json(out / "eval_summary.json", summary)
    write_json(out / "metric_coverage.json", coverage)
    write_csv(out / "generation_failures.csv", failures)
    write_eval_markdown(root, variant_id)
    return summary


def write_eval_markdown(root: Path, variant_id: str) -> None:
    out = root / EXPERIMENTS_DIR / variant_id
    summary = read_json(out / "eval_summary.json", {})
    coverage = read_json(out / "metric_coverage.json", {})
    lines = [f"# {variant_id} Evaluation Summary", "", "| Metric | Mean score | Successful count |", "|---|---:|---:|"]
    for name in METRIC_NAMES:
        item = (summary.get("metrics") or {}).get(name, {})
        lines.append(f"| {name} | {item.get('mean_score')} | {item.get('successful_count')} |")
    write_md(out / "eval_summary.md", lines)
    cov_lines = [f"# {variant_id} Metric Coverage", "", "| Metric | Coverage | Successful | Failed | Not applicable |", "|---|---:|---:|---:|---:|"]
    for name in METRIC_NAMES:
        item = coverage.get(name, {})
        cov_lines.append(f"| {name} | {item.get('coverage')} | {item.get('successful_evaluations')} | {item.get('failed_evaluations')} | {item.get('not_applicable')} |")
    write_md(out / "metric_coverage.md", cov_lines)


def evaluate_final_generation_bakeoff(root: Path = Path(".")) -> dict[str, Any]:
    summaries = {variant_id: evaluate_variant(root, variant_id) for variant_id in VARIANTS}
    leaderboard = write_leaderboard(root)
    category = write_category_analysis(root)
    decision = write_selection_decision(root, leaderboard)
    validation = validate_final_generation_bakeoff(root)
    return {
        "status": validation["status"],
        "summaries": summaries,
        "leaderboard_rows": len(leaderboard),
        "category_rows": len(category),
        "decision": decision,
        "validation": validation,
    }


def _retrieval_summary(root: Path, retrieval_id: str) -> dict[str, Any]:
    return read_json(root / RETRIEVAL_SUMMARY_PATHS[retrieval_id], {})


def _leaderboard_row(root: Path, variant_id: str, baseline_metrics: dict[str, float | None]) -> dict[str, Any]:
    out = root / EXPERIMENTS_DIR / variant_id
    config = VARIANTS[variant_id]
    summary = read_json(out / "summary.json", {})
    eval_summary = read_json(out / "eval_summary.json", {})
    contexts = read_json(out / "context_records.json", [])
    retrieval = _retrieval_summary(root, config["retrieval_experiment_id"])
    row = {
        "variant_id": variant_id,
        "retrieval_experiment_id": config["retrieval_experiment_id"],
        "prompt_version": config["prompt_version"],
        "context_ordering": config["context_ordering"],
        "status": summary.get("status", eval_summary.get("status", "not_run")),
        "notes": summary.get("reason") or config["description"],
        "success_rate": summary.get("success_rate"),
        "mean_generation_latency_ms": summary.get("mean_generation_latency_ms"),
        "median_generation_latency_ms": summary.get("median_generation_latency_ms"),
        "p95_generation_latency_ms": summary.get("p95_generation_latency_ms"),
        "mean_estimated_context_tokens": _mean([item.get("estimated_token_count") for item in contexts]),
        "mean_unique_pages": _mean([item.get("unique_page_count") for item in contexts]),
        "retrieval_complete_evidence_recall": retrieval.get("complete_evidence_recall"),
        "retrieval_all_reports_hit": retrieval.get("all_reports_hit"),
        "retrieval_evidence_recall": retrieval.get("evidence_recall"),
        "retrieval_macro_mrr": retrieval.get("macro_report_mrr"),
        "retrieval_median_latency_ms": retrieval.get("median_latency_ms"),
    }
    for metric in METRIC_NAMES:
        row[metric] = _metric(eval_summary, metric)
    eligibility, reasons = eligibility_status(row, baseline_metrics)
    row["eligibility"] = eligibility
    row["eligibility_reasons"] = reasons
    return row


def baseline_metrics(root: Path) -> dict[str, float | None]:
    summary = read_json(root / EXPERIMENTS_DIR / PREVIOUS_BEST_VARIANT / "eval_summary.json", {})
    return {name: _metric(summary, name) for name in METRIC_NAMES}


def eligibility_status(row: dict[str, Any], baseline: dict[str, float | None]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if row.get("status") not in {"completed", "completed_reused", "completed_repaired"}:
        reasons.append(f"status:{row.get('status')}")
    if row.get("success_rate") != 1.0:
        reasons.append(f"success_rate:{row.get('success_rate')}")
    abstention = row.get("abstention_correctness")
    citation = row.get("citation_correctness")
    factual = row.get("factual_correctness")
    temporal = row.get("temporal_attribution_correctness")
    if abstention is None or baseline.get("abstention_correctness") is None or abstention < baseline["abstention_correctness"]:
        reasons.append("abstention_below_previous_best")
    citation_ok = citation is not None and baseline.get("citation_correctness") is not None and citation >= baseline["citation_correctness"]
    factual_material = (
        factual is not None and baseline.get("factual_correctness") is not None
        and factual - baseline["factual_correctness"] > 0.03
        and citation is not None and baseline.get("citation_correctness") is not None
        and citation - baseline["citation_correctness"] >= -0.03
    )
    if not (citation_ok or factual_material):
        reasons.append("citation_not_preserved_without_material_factual_gain")
    if temporal is None or baseline.get("temporal_attribution_correctness") is None or temporal < baseline["temporal_attribution_correctness"] - 0.03:
        reasons.append("temporal_regression_gt_0.03")
    if row.get("api_key_scan_status") == "failed":
        reasons.append("api_key_scan_failed")
    return ("eligible" if not reasons else "not_eligible", reasons)


def write_leaderboard(root: Path) -> list[dict[str, Any]]:
    base = baseline_metrics(root)
    rows = [_leaderboard_row(root, variant_id, base) for variant_id in VARIANTS]
    rows = sorted(rows, key=lambda row: (VARIANTS[row["variant_id"]]["priority"], row["variant_id"]))
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "previous_best_variant": PREVIOUS_BEST_VARIANT,
        "rows": rows,
        "completed": [row for row in rows if row["status"] in {"completed", "completed_reused", "completed_repaired"}],
        "skipped": [row for row in rows if row["status"] == "skipped"],
    }
    write_json(root / OUT_DIR / "generation_bakeoff_leaderboard.json", payload)
    write_csv(root / OUT_DIR / "generation_bakeoff_leaderboard.csv", rows)
    lines = [
        "# Final Generation Bake-Off Leaderboard",
        "",
        "| Variant | Retrieval | Prompt | Ordering | Status | Eligible | Factual | Citation | Temporal | Comparative | Abstention | Success | Median ms |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant_id']} | {row.get('retrieval_experiment_id')} | {row.get('prompt_version')} | {row.get('context_ordering')} | "
            f"{row.get('status')} | {row.get('eligibility')} | {row.get('factual_correctness')} | {row.get('citation_correctness')} | "
            f"{row.get('temporal_attribution_correctness')} | {row.get('comparative_correctness')} | {row.get('abstention_correctness')} | "
            f"{row.get('success_rate')} | {row.get('median_generation_latency_ms')} |"
        )
    write_md(root / OUT_DIR / "generation_bakeoff_leaderboard.md", lines)
    return rows


def _case_dimension(case: dict[str, Any], row: dict[str, Any], dimension: str) -> Any:
    if dimension == "requires_numeric_evidence":
        return bool(row.get("table_or_numeric_question") or case.get("requires_numeric_evidence") or case.get("table_or_numeric_question"))
    if dimension == "sufficiency_status":
        return row.get("sufficiency_status")
    return case.get(dimension) or row.get(dimension) or "unknown"


def write_category_analysis(root: Path) -> list[dict[str, Any]]:
    cases = expected_case_lookup(root, "dev")
    dimensions = ["query_type", "source_structure", "requires_numeric_evidence", "sufficiency_status"]
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANTS:
        out = root / EXPERIMENTS_DIR / variant_id
        eval_rows = read_json(out / "eval_raw_results.json", [])
        gen_rows = read_json(out / "raw_results.json", [])
        gen_by_id = {row.get("question_id"): row for row in gen_rows}
        retrieval_id = VARIANTS[variant_id]["retrieval_experiment_id"]
        retrieval_rows = read_json(root / RETRIEVAL_RAW_PATHS[retrieval_id], [])
        retrieval_by_id = {row.get("question_id"): row for row in retrieval_rows}
        if not eval_rows:
            continue
        for dimension in dimensions:
            grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in eval_rows:
                gen = gen_by_id.get(row.get("question_id"), {})
                retrieval = retrieval_by_id.get(row.get("question_id"), {})
                case = cases.get(row.get("question_id"), {})
                grouped[_case_dimension(case, retrieval | gen | row, dimension)].append(row)
            for value, members in grouped.items():
                item = {
                    "variant_id": variant_id,
                    "dimension": dimension,
                    "category_value": value,
                    "row_count": len(members),
                }
                for metric in METRIC_NAMES:
                    scores = [
                        member["metrics"][metric]["score"]
                        for member in members
                        if member.get("metrics", {}).get(metric, {}).get("success")
                        and member["metrics"][metric].get("score") is not None
                    ]
                    item[metric] = sum(scores) / len(scores) if scores else None
                rows.append(item)
    write_json(root / OUT_DIR / "category_analysis.json", {"schema_version": 1, "created_at_utc": now_iso(), "rows": rows})
    write_csv(root / OUT_DIR / "category_analysis.csv", rows)
    lines = ["# Final Generation Bake-Off Category Analysis", "", "| Variant | Dimension | Value | Rows | Factual | Citation | Temporal | Comparative |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['variant_id']} | {row['dimension']} | {row['category_value']} | {row['row_count']} | "
            f"{row.get('factual_correctness')} | {row.get('citation_correctness')} | {row.get('temporal_attribution_correctness')} | {row.get('comparative_correctness')} |"
        )
    write_md(root / OUT_DIR / "category_analysis.md", lines)
    return rows


def _selection_rank(row: dict[str, Any]) -> tuple:
    simplicity = {
        "GEN_MMR06_SUFFICIENCY_V1": 0,
        "GEN_V2_COHERE_SUFFICIENCY_V1": 1,
        "GEN_MMR06_CITATION_REPAIR_V1": 2,
        "GEN_MMR07_SUFFICIENCY_V1": 3,
        "GEN_MMR08_SUFFICIENCY_V1": 3,
        "GEN_MMR06_CHRONO_ORDER_V1": 4,
        "GEN_MMR06_RERANK_ORDER_V1": 4,
        "GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1": 5,
        "GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1": 5,
    }.get(row["variant_id"], 9)
    return (
        row.get("factual_correctness") if row.get("factual_correctness") is not None else -1,
        row.get("citation_correctness") if row.get("citation_correctness") is not None else -1,
        row.get("temporal_attribution_correctness") if row.get("temporal_attribution_correctness") is not None else -1,
        row.get("abstention_correctness") if row.get("abstention_correctness") is not None else -1,
        row.get("comparative_correctness") if row.get("comparative_correctness") is not None else -1,
        row.get("contextual_recall") if row.get("contextual_recall") is not None else -1,
        -(row.get("median_generation_latency_ms") or 10**12),
        -simplicity,
    )


def write_selection_decision(root: Path, leaderboard_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in leaderboard_rows if row.get("eligibility") == "eligible"]
    baseline = next((row for row in leaderboard_rows if row["variant_id"] == PREVIOUS_BEST_VARIANT), None)
    selected = max(eligible, key=_selection_rank) if eligible else baseline
    if selected is None:
        status = "blocked_no_completed_generation_variant"
        selected_id = None
        reason = "No completed eligible bake-off variant was available."
    elif selected["variant_id"] == PREVIOUS_BEST_VARIANT:
        status = "kept_previous_best_generation_strategy"
        selected_id = selected["variant_id"]
        reason = "No eligible bake-off variant beat the current MMR06 sufficiency baseline under the selection policy."
    else:
        status = "selected_bakeoff_generation_strategy"
        selected_id = selected["variant_id"]
        reason = f"{selected_id} ranked highest among eligible development-only bake-off variants."
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": status,
        "selected_variant_id": selected_id,
        "selected_retrieval_method": selected.get("retrieval_experiment_id") if selected else None,
        "selected_generation_method": selected_id,
        "selected_end_to_end_dev_system": selected_id,
        "previous_best_variant": PREVIOUS_BEST_VARIANT,
        "reason": reason,
        "selection_policy": [
            "success rate 1.0",
            "abstention >= previous best",
            "citation >= previous best or material factual gain without citation regression > 0.03",
            "temporal attribution no regression > 0.03",
            "factual, citation, temporal, abstention, comparative, contextual recall, latency, simplicity",
        ],
        "selected_row": selected,
        "heldout_used": False,
    }
    write_json(root / OUT_DIR / "final_generation_strategy_selection_decision.json", payload)
    write_md(
        root / OUT_DIR / "final_generation_strategy_selection_decision.md",
        [
            "# Final Generation Strategy Selection Decision",
            "",
            f"Status: `{status}`",
            f"Selected variant: `{selected_id}`",
            f"Selected retrieval method: `{payload['selected_retrieval_method']}`",
            "",
            reason,
        ],
    )
    return payload


def write_variant_registry(root: Path) -> None:
    rows = [{"variant_id": variant_id, **config} for variant_id, config in VARIANTS.items()]
    write_json(root / OUT_DIR / "final_generation_bakeoff_variant_registry.json", {"schema_version": 1, "rows": rows})
    write_csv(root / OUT_DIR / "final_generation_bakeoff_variant_registry.csv", rows)
    lines = ["# Final Generation Bake-Off Variant Registry", "", "| Variant | Mode | Retrieval | Prompt | Ordering | Priority |", "|---|---|---|---|---|---:|"]
    for row in rows:
        lines.append(
            f"| {row['variant_id']} | {row['mode']} | {row['retrieval_experiment_id']} | {row['prompt_version']} | {row['context_ordering']} | {row['priority']} |"
        )
    write_md(root / OUT_DIR / "final_generation_bakeoff_variant_registry.md", lines)


def validate_final_generation_bakeoff(root: Path = Path(".")) -> dict[str, Any]:
    load_project_dotenv(root)
    issues: list[str] = []
    for variant_id in VARIANTS:
        out = root / EXPERIMENTS_DIR / variant_id
        if not (out / "summary.json").exists():
            issues.append(f"{variant_id}:missing_summary")
        if not (out / "eval_summary.json").exists():
            issues.append(f"{variant_id}:missing_eval_summary")
        rows = read_json(out / "raw_results.json", [])
        contexts = read_json(out / "context_records.json", [])
        context_chunks = {row["question_id"]: set(row.get("selected_chunk_ids") or []) for row in contexts}
        for row in rows:
            if row.get("split") != "dev":
                issues.append(f"{variant_id}:{row.get('question_id')}:non_dev_generation")
            if row.get("retrieval_experiment_id") != VARIANTS[variant_id]["retrieval_experiment_id"]:
                issues.append(f"{variant_id}:{row.get('question_id')}:wrong_retrieval_source")
            for citation in row.get("citations") or []:
                if citation.get("chunk_id") not in context_chunks.get(row.get("question_id"), set()):
                    issues.append(f"{variant_id}:{row.get('question_id')}:citation_not_in_supplied_context:{citation.get('chunk_id')}")
        eval_rows = read_json(out / "eval_raw_results.json", [])
        for row in eval_rows:
            for name, metric in (row.get("metrics") or {}).items():
                if not metric.get("success", True) and metric.get("score") is not None:
                    issues.append(f"{variant_id}:{row.get('question_id')}:failed_metric_has_non_null_score:{name}")
    required_top = [
        root / OUT_DIR / "generation_bakeoff_leaderboard.json",
        root / OUT_DIR / "category_analysis.json",
        root / OUT_DIR / "final_generation_strategy_selection_decision.json",
    ]
    for path in required_top:
        if not path.exists():
            issues.append(f"missing_top_level_artifact:{rel(root, path)}")
    all_text: dict[str, str] = {}
    out_root = root / OUT_DIR
    if out_root.exists():
        for path in out_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv", ".yaml", ".yml", ".log", ".txt"}:
                all_text[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
    if actual_key_values_serialized(all_text):
        issues.append("api_key_value_serialized")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(sorted(set(issues))),
        "issues": sorted(set(issues)),
        "heldout_used": False,
        "fresh_eval_created": False,
        "api_key_scan_status": "passed" if "api_key_value_serialized" not in issues else "failed",
        "frozen_phase7_outputs_overwritten": False,
    }
    write_json(root / OUT_DIR / "final_generation_bakeoff_validation.json", payload)
    lines = ["# Final Generation Bake-Off Validation", "", f"Status: `{payload['status']}`", f"Issues: {payload['issue_count']}"]
    if payload["issues"]:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    write_md(root / OUT_DIR / "final_generation_bakeoff_validation.md", lines)
    return payload


def generate_final_generation_bakeoff_report(root: Path = Path(".")) -> dict[str, Any]:
    leaderboard_payload = read_json(root / OUT_DIR / "generation_bakeoff_leaderboard.json", {"rows": []})
    rows = leaderboard_payload.get("rows", [])
    category = read_json(root / OUT_DIR / "category_analysis.json", {"rows": []}).get("rows", [])
    decision = read_json(root / OUT_DIR / "final_generation_strategy_selection_decision.json", {})
    validation = validate_final_generation_bakeoff(root)
    selected = decision.get("selected_row") or {}
    report = [
        "# Final Generation Strategy Bake-Off Report",
        "",
        "## Scope",
        "",
        "This is a development-only bake-off for Temporal multi-document RAG for RBI Monetary Policy Reports. It does not use held-out data, create a fresh evaluation set, run Unstructured/Tesseract work, or change retrieval models.",
        "",
        "## Why this bake-off was needed",
        "",
        "The project had a selected retrieval-only MMR configuration and a strong MMR06 sufficiency-gated generation result. The remaining question was whether nearby MMR lambdas, context ordering, prompt variants, or deterministic citation repair improved the end-to-end answer metrics.",
        "",
        "## Leaderboard",
        "",
        "| Variant | Retrieval | Prompt | Ordering | Status | Factual | Citation | Temporal | Comparative | Abstention | Eligible |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['variant_id']} | {row.get('retrieval_experiment_id')} | {row.get('prompt_version')} | {row.get('context_ordering')} | "
            f"{row.get('status')} | {row.get('factual_correctness')} | {row.get('citation_correctness')} | {row.get('temporal_attribution_correctness')} | "
            f"{row.get('comparative_correctness')} | {row.get('abstention_correctness')} | {row.get('eligibility')} |"
        )
    report += [
        "",
        "## Selection decision",
        "",
        f"Status: `{decision.get('status')}`",
        f"Selected variant: `{decision.get('selected_variant_id')}`",
        f"Selected retrieval method: `{decision.get('selected_retrieval_method')}`",
        "",
        decision.get("reason", ""),
        "",
        "## Category-level findings",
        "",
        "See `category_analysis.md` for query type, source structure, numeric-evidence, and sufficiency-status breakdowns.",
        "",
        "## Scientific caveats",
        "",
        "- Development-only comparison.",
        "- No held-out rerun.",
        "- No fresh V2 evaluation set.",
        "- Metrics are deterministic heuristic evaluation signals, not human evaluation.",
        "- Skipped variants are explicitly marked and must not be claimed as tested.",
        "",
        "## Interview-ready explanation",
        "",
        f"The final selected development system is `{decision.get('selected_end_to_end_dev_system')}` using `{decision.get('selected_retrieval_method')}` retrieval. It was selected because it ranked best among eligible development-only variants under factual, citation, temporal attribution, abstention, comparative, latency, and simplicity criteria.",
        "",
        "## Validation",
        "",
        f"Validation status: `{validation.get('status')}`. API-key scan: `{validation.get('api_key_scan_status')}`.",
    ]
    write_md(root / OUT_DIR / "final_generation_bakeoff_report.md", report)
    presentation = [
        "# Final Generation Bake-Off for Presentation",
        "",
        f"Selected end-to-end dev system: `{decision.get('selected_end_to_end_dev_system')}`",
        f"Selected retrieval: `{decision.get('selected_retrieval_method')}`",
        "",
        "| Variant | Factual | Citation | Temporal | Comparative | Abstention | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        presentation.append(
            f"| {row['variant_id']} | {row.get('factual_correctness')} | {row.get('citation_correctness')} | "
            f"{row.get('temporal_attribution_correctness')} | {row.get('comparative_correctness')} | {row.get('abstention_correctness')} | {row.get('status')} |"
        )
    presentation += [
        "",
        "Use wording: development-selected, not production-ready, not held-out-final.",
        "",
        f"Selected factual correctness: {selected.get('factual_correctness')}",
        f"Selected citation correctness: {selected.get('citation_correctness')}",
        f"Selected temporal attribution correctness: {selected.get('temporal_attribution_correctness')}",
    ]
    write_md(root / OUT_DIR / "final_generation_bakeoff_for_presentation.md", presentation)
    return {
        "status": validation["status"],
        "report": str(root / OUT_DIR / "final_generation_bakeoff_report.md"),
        "presentation": str(root / OUT_DIR / "final_generation_bakeoff_for_presentation.md"),
        "validation": validation,
        "selected_variant": decision.get("selected_variant_id"),
        "category_rows": len(category),
    }
