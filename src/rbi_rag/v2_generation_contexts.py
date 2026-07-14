from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REPORT_ORDER = {
    "rbi_mpr_2025_04": 0,
    "rbi_mpr_2025_10": 1,
    "rbi_mpr_2026_04": 2,
}


REQUIRED_EXPERIMENT_FILES = {
    "config_snapshot.yaml",
    "environment.json",
    "index_manifest.json",
    "raw_results.json",
    "question_results.csv",
    "report_level_results.csv",
    "summary.json",
    "summary.md",
    "stage_diagnostics.csv",
    "integrity.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_selected_v2_config(
    root: Path,
    config_path: Path = Path("configs/v2_selected_retrieval.yaml"),
    expected_experiment_id: str = "V2_COHERE_ONLY",
) -> dict[str, Any]:
    path = root / config_path
    issues: list[str] = []
    if not path.exists():
        return {"status": "failed", "issues": [f"missing_config:{config_path}"]}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("id") != expected_experiment_id:
        issues.append(f"selected_config_id_is_not_{expected_experiment_id}:{raw.get('id')}")
    if raw.get("reranker", {}).get("provider") != "cohere":
        issues.append("selected_config_reranker_is_not_cohere")
    if raw.get("parser", {}).get("provider") != "pypdfloader":
        issues.append("selected_config_parser_is_not_pypdfloader")
    return {
        "status": "passed" if not issues else "failed",
        "path": str(config_path).replace("/", "\\"),
        "selected_experiment_id": raw.get("id"),
        "issues": issues,
        "config": raw,
    }


def experiment_dir(root: Path, experiment_id: str) -> Path:
    return root / "reports/v2_unstructured_cohere/experiments" / experiment_id


def validate_v2_retrieval_input(root: Path, experiment_id: str = "V2_COHERE_ONLY") -> dict[str, Any]:
    path = experiment_dir(root, experiment_id)
    issues: list[str] = []
    if not path.exists():
        return {"status": "failed", "issues": [f"missing_experiment_dir:{path}"]}
    files = {child.name for child in path.iterdir() if child.is_file()}
    for name in sorted(REQUIRED_EXPERIMENT_FILES - files):
        issues.append(f"missing_required_file:{name}")
    integrity = load_json(path / "integrity.json") if (path / "integrity.json").exists() else {}
    summary = load_json(path / "summary.json") if (path / "summary.json").exists() else {}
    rows = load_json(path / "raw_results.json") if (path / "raw_results.json").exists() else []
    if integrity.get("status") != "valid":
        issues.append(f"integrity_not_valid:{integrity.get('status')}")
    if summary.get("experiment_id") != experiment_id:
        issues.append(f"summary_experiment_id_mismatch:{summary.get('experiment_id')}")
    heldout_rows = []
    missing_context_rows = []
    missing_source_rows = []
    for row in rows:
        qid = row.get("question_id", "unknown")
        if row.get("split") != "dev" or str(qid).startswith("test_"):
            heldout_rows.append(qid)
        chunks = [
            chunk
            for chunks in (row.get("selected_chunks_by_report") or {}).values()
            for chunk in chunks
        ] or list(row.get("all_selected_chunks") or [])
        if row.get("query_type") != "unsupported_period" and not chunks:
            missing_context_rows.append(qid)
        for chunk in chunks:
            if not chunk.get("report_id") or chunk.get("page") is None or not chunk.get("chunk_id"):
                missing_source_rows.append(qid)
            if chunk.get("text") is None:
                missing_context_rows.append(qid)
    if heldout_rows:
        issues.append(f"heldout_or_non_dev_rows_present:{sorted(set(heldout_rows))}")
    if missing_context_rows:
        issues.append(f"missing_selected_chunk_text:{sorted(set(missing_context_rows))}")
    if missing_source_rows:
        issues.append(f"missing_source_metadata:{sorted(set(missing_source_rows))}")
    return {
        "status": "passed" if not issues else "failed",
        "experiment_id": experiment_id,
        "experiment_dir": str(path),
        "row_count": len(rows),
        "integrity_status": integrity.get("status"),
        "summary_experiment_id": summary.get("experiment_id"),
        "split": "dev",
        "heldout_rows_present": bool(heldout_rows),
        "selected_chunk_text_present": not missing_context_rows,
        "source_metadata_present": not missing_source_rows,
        "issues": issues,
    }


def _retriever_source(row: dict[str, Any], report_id: str, chunk_id: str) -> str:
    dense_ids = {
        item.get("chunk_id")
        for item in (row.get("dense_candidates_by_report") or {}).get(report_id, [])
    }
    bm25_ids = {
        item.get("chunk_id")
        for item in (row.get("bm25_candidates_by_report") or {}).get(report_id, [])
    }
    sources = []
    if chunk_id in dense_ids:
        sources.append("dense")
    if chunk_id in bm25_ids:
        sources.append("bm25")
    if not sources:
        sources.append("selected_after_rerank")
    return "+".join(sources)


def _selected_chunks(row: dict[str, Any]) -> list[dict[str, Any]]:
    chunks_by_report = row.get("selected_chunks_by_report") or {}
    chunks = [dict(chunk) for _, values in chunks_by_report.items() for chunk in values]
    if chunks:
        return chunks
    by_id = {chunk.get("chunk_id"): dict(chunk) for chunk in row.get("all_selected_chunks") or []}
    output: list[dict[str, Any]] = []
    for report_id, trace in (row.get("per_report") or {}).items():
        for chunk_id in trace.get("selected_chunk_ids_after_dedup", []):
            if chunk_id in by_id:
                output.append(by_id[chunk_id])
            else:
                output.append({"chunk_id": chunk_id, "report_id": report_id, "text": None})
    return output


def _source_label(block: dict[str, Any]) -> str:
    return (
        f"[SOURCE: {block['report_period']} MPR | page {block['page_number']} | "
        f"chunk {block['chunk_id']}]"
    )


def build_context_for_row(row: dict[str, Any]) -> dict[str, Any]:
    required = set(row.get("required_report_ids") or [])
    chunks = [
        chunk for chunk in _selected_chunks(row)
        if chunk.get("report_id") in required and chunk.get("text")
    ]
    blocks: list[dict[str, Any]] = []
    for chunk in chunks:
        report_id = str(chunk["report_id"])
        page_number = chunk.get("page", chunk.get("page_number"))
        block = {
            "report_period": chunk.get("report_period") or report_id,
            "report_id": report_id,
            "page_number": int(page_number) if page_number is not None else None,
            "chunk_id": chunk["chunk_id"],
            "retriever_source": _retriever_source(row, report_id, chunk["chunk_id"]),
            "reranker_provider": row.get("reranker_provider"),
            "reranker_model": row.get("reranker_model"),
            "text": chunk["text"],
        }
        block["source_label"] = _source_label(block)
        blocks.append(block)
    blocks.sort(key=lambda item: (
        REPORT_ORDER.get(item["report_id"], 99),
        item["page_number"] or 0,
        item["chunk_id"],
    ))
    grouped_lines: list[str] = []
    current_report: str | None = None
    for block in blocks:
        if block["report_id"] != current_report:
            current_report = block["report_id"]
            grouped_lines.append(f"## {block['report_period']}")
        grouped_lines.append(f"{block['source_label']}\n{block['text']}")
    return {
        "question_id": row["question_id"],
        "split": row.get("split"),
        "query_type": row.get("query_type"),
        "required_report_ids": list(row.get("required_report_ids") or []),
        "original_query": row.get("original_query"),
        "normalised_query": row.get("normalised_query"),
        "retrieval_experiment_id": row.get("experiment_id"),
        "retrieval_config_checksum": row.get("configuration_checksum"),
        "reranker_provider": row.get("reranker_provider"),
        "reranker_model": row.get("reranker_model"),
        "selected_chunk_ids": [block["chunk_id"] for block in blocks],
        "selected_pages": {
            report_id: [
                block["page_number"] for block in blocks
                if block["report_id"] == report_id and block["page_number"] is not None
            ]
            for report_id in sorted(required, key=lambda rid: REPORT_ORDER.get(rid, 99))
        },
        "context_blocks": blocks,
        "source_labelled_context": "\n\n".join(grouped_lines),
    }


def build_generation_contexts(
    root: Path,
    experiment_id: str = "V2_COHERE_ONLY",
) -> list[dict[str, Any]]:
    rows = load_json(experiment_dir(root, experiment_id) / "raw_results.json")
    return [build_context_for_row(row) for row in rows]
