from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .final_evaluation import write_csv, write_json
from .v2_generation_evaluation import now_iso


SUFFICIENCY_STATUSES = {"sufficient", "partially_sufficient", "insufficient"}
REQUIRED_GENERATION_BEHAVIORS = {"answer", "answer_with_caveat", "abstain"}
NUMERIC_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|bps|basis points|per cent|percent|%)", re.IGNORECASE)


def selected_chunks_by_required_report(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    required = list(row.get("required_report_ids") or [])
    chunks_by_report = row.get("selected_chunks_by_report") or {}
    return {
        report_id: [
            chunk for chunk in chunks_by_report.get(report_id, [])
            if chunk.get("text")
        ]
        for report_id in required
    }


def context_text(context_item: dict[str, Any] | None, row: dict[str, Any]) -> str:
    if context_item and context_item.get("source_labelled_context"):
        return str(context_item["source_labelled_context"])
    chunks = [
        chunk
        for chunks in (row.get("selected_chunks_by_report") or {}).values()
        for chunk in chunks
    ]
    return "\n".join(str(chunk.get("text") or "") for chunk in chunks)


def selected_chunk_text(row: dict[str, Any]) -> str:
    chunks = [
        chunk
        for chunks in (row.get("selected_chunks_by_report") or {}).values()
        for chunk in chunks
    ]
    return "\n".join(str(chunk.get("text") or "") for chunk in chunks)


def is_table_or_numeric(row: dict[str, Any], case: dict[str, Any] | None = None) -> bool:
    if row.get("table_or_numeric_question"):
        return True
    case = case or {}
    text = " ".join([
        str(row.get("original_query") or ""),
        str(row.get("source_structure") or ""),
        " ".join(row.get("source_information_type") or []),
        str(case.get("question") or ""),
        str(case.get("expected_answer") or ""),
        " ".join(case.get("source_information_type") or []),
    ]).lower()
    return any(token in text for token in ("table", "chart", "projection", "forecast", "rate", "per cent", "%", "bps", "inflation", "growth"))


def has_numeric_evidence(text: str) -> bool:
    return bool(NUMERIC_RE.search(text or ""))


def classify_evidence_sufficiency(
    row: dict[str, Any],
    context_item: dict[str, Any] | None = None,
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_type = row.get("query_type")
    required_reports = list(row.get("required_report_ids") or [])
    chunks_by_required = selected_chunks_by_required_report(row)
    selected_text = context_text(context_item, row)
    numeric_text = selected_chunk_text(row)
    reasons: list[str] = []

    if query_type == "unsupported_period":
        if not selected_text.strip() or not required_reports:
            reasons.extend(["unsupported_period", "context_empty"])
        return {
            "question_id": row.get("question_id"),
            "query_type": query_type,
            "required_report_ids": required_reports,
            "sufficiency_status": "insufficient",
            "sufficiency_reasons": sorted(set(reasons or ["unsupported_period"])),
            "required_generation_behavior": "abstain",
            "retrieval_complete_evidence_recall": row.get("complete_evidence_recall"),
            "retrieval_evidence_recall": row.get("evidence_recall"),
            "retrieval_all_reports_hit": row.get("all_reports_hit"),
            "report_coverage": row.get("report_coverage"),
            "table_or_numeric_question": is_table_or_numeric(row, case),
            "numeric_evidence_present": has_numeric_evidence(numeric_text),
            "selected_chunk_count": sum(len(chunks) for chunks in chunks_by_required.values()),
            "missing_required_reports": [],
        }

    if not selected_text.strip():
        reasons.append("context_empty")

    report_coverage = row.get("report_coverage")
    if isinstance(report_coverage, (int, float)) and float(report_coverage) < 1.0:
        reasons.append("missing_required_report")

    missing_reports = [
        report_id for report_id, chunks in chunks_by_required.items()
        if not chunks
    ]
    if missing_reports:
        reasons.append("missing_required_report")

    if row.get("complete_evidence_recall") is True and not reasons:
        status = "sufficient"
        behavior = "answer"
    else:
        evidence_recall = row.get("evidence_recall")
        if row.get("complete_evidence_recall") is not True:
            reasons.append("incomplete_evidence")
        if isinstance(evidence_recall, (int, float)) and float(evidence_recall) < 0.5:
            reasons.append("low_evidence_recall")
        if query_type in {"pairwise_comparison", "trend_all_reports"} and missing_reports:
            reasons.append("missing_comparative_counterpart")
            status = "insufficient"
            behavior = "abstain"
        elif "context_empty" in reasons or "missing_required_report" in reasons:
            status = "insufficient"
            behavior = "abstain"
        else:
            status = "partially_sufficient"
            behavior = "answer_with_caveat"

    table_numeric = is_table_or_numeric(row, case)
    numeric_present = has_numeric_evidence(numeric_text)
    if status == "sufficient":
        return {
            "question_id": row.get("question_id"),
            "query_type": query_type,
            "required_report_ids": required_reports,
            "sufficiency_status": status,
            "sufficiency_reasons": [],
            "required_generation_behavior": behavior,
            "retrieval_complete_evidence_recall": row.get("complete_evidence_recall"),
            "retrieval_evidence_recall": row.get("evidence_recall"),
            "retrieval_all_reports_hit": row.get("all_reports_hit"),
            "report_coverage": row.get("report_coverage"),
            "table_or_numeric_question": table_numeric,
            "numeric_evidence_present": numeric_present,
            "selected_chunk_count": sum(len(chunks) for chunks in chunks_by_required.values()),
            "missing_required_reports": missing_reports,
        }
    if table_numeric and not numeric_present:
        reasons.append("missing_numeric_evidence")
        if status == "sufficient":
            status = "partially_sufficient"
            behavior = "answer_with_caveat"
        elif status == "partially_sufficient" and query_type in {"pairwise_comparison", "trend_all_reports"}:
            status = "insufficient"
            behavior = "abstain"
    source_info = " ".join(row.get("source_information_type") or []) + " " + str(row.get("source_structure") or "")
    if "table" in source_info.lower() and "table" not in selected_text.lower():
        reasons.append("missing_table_evidence")
        if status == "sufficient":
            status = "partially_sufficient"
            behavior = "answer_with_caveat"

    if status == "sufficient":
        reasons = []

    return {
        "question_id": row.get("question_id"),
        "query_type": query_type,
        "required_report_ids": required_reports,
        "sufficiency_status": status,
        "sufficiency_reasons": sorted(set(reasons or ["other"] if status != "sufficient" else [])),
        "required_generation_behavior": behavior,
        "retrieval_complete_evidence_recall": row.get("complete_evidence_recall"),
        "retrieval_evidence_recall": row.get("evidence_recall"),
        "retrieval_all_reports_hit": row.get("all_reports_hit"),
        "report_coverage": row.get("report_coverage"),
        "table_or_numeric_question": table_numeric,
        "numeric_evidence_present": numeric_present,
        "selected_chunk_count": sum(len(chunks) for chunks in chunks_by_required.values()),
        "missing_required_reports": missing_reports,
    }


def classify_all(
    retrieval_rows: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    context_by_id = {item["question_id"]: item for item in contexts}
    cases = cases or {}
    return [
        classify_evidence_sufficiency(row, context_by_id.get(row["question_id"]), cases.get(row["question_id"]))
        for row in retrieval_rows
    ]


def write_sufficiency_classification(root: Path, out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = root / out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dev_sufficiency_classification.json", rows)
    write_csv(out / "dev_sufficiency_classification.csv", rows)
    status_counts = Counter(row["sufficiency_status"] for row in rows)
    behavior_counts = Counter(row["required_generation_behavior"] for row in rows)
    reason_counts = Counter(reason for row in rows for reason in row.get("sufficiency_reasons", []))
    summary = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "behavior_counts": dict(behavior_counts),
        "reason_counts": dict(reason_counts),
    }
    lines = [
        "# Development Evidence Sufficiency Classification",
        "",
        f"Rows: {len(rows)}",
        "",
        "## Status counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(status_counts.items()))
    lines += ["", "## Required generation behaviour", ""]
    lines.extend(f"- {key}: {value}" for key, value in sorted(behavior_counts.items()))
    lines += ["", "## Reason counts", ""]
    lines.extend(f"- {key}: {value}" for key, value in sorted(reason_counts.items()))
    (out / "dev_sufficiency_classification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
