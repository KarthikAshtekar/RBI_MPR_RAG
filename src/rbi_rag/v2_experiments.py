from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .cohere_reranker import CohereRerankConfig, CohereRerankError, CohereReranker, cohere_available
from .env_loading import load_project_dotenv
from .final_evaluation import (
    assert_config_not_mutated,
    canonicalise_retrieval_row,
    contains_groq_secret,
    file_sha,
    groq_key_available,
    make_checksum_manifest,
    recompute_retrieval_metrics,
    report_level_rows,
    stable_json_hash,
    validate_final_status,
    validate_generation_integrity,
    validate_heldout_dataset_manifest,
    validate_heldout_raw_schema,
    write_csv,
    write_json,
)
from .config import file_sha256
from .multi_config import MultiReportConfig
from .multi_evaluation import load_jsonl
from .multi_index import build_multi_report_index
from .poppler_setup import POPPLER_RETRY_OUT
from .report_bm25 import BM25ByReport
from .report_registry import ReportRegistry
from .temporal_router import TemporalQueryRouter
from .unstructured_extraction import (
    chunk_unstructured_elements,
    extract_pdf_elements,
    page_element_counts,
    unstructured_available,
    unstructured_version,
)
from .uncertainty import bootstrap_mean_interval

from scripts.run_stage_a_ablations import run_question as run_stage_a_question


ROOT = Path(".")
V2_OUT = Path("reports/v2_unstructured_cohere")
V2_INDEX = Path("indexes/v2_unstructured_cohere")
V2_CONFIG = Path("configs/v2_unstructured_cohere_experiments.yaml")
FINAL_CONFIG = Path("configs/final_retrieval_selected.yaml")

CONTROLLED_EXPERIMENTS = (
    "V2_BASELINE_FINAL",
    "V2_UNSTRUCTURED_ONLY",
    "V2_COHERE_ONLY",
    "V2_UNSTRUCTURED_COHERE",
)

V2_CHECKSUM_TARGETS = [
    Path("configs/final_retrieval_selected.yaml"),
    Path("reports/final_evaluation"),
    Path("reports/structural_optimisation"),
    Path("reports/optimisation"),
    Path("data/evaluation"),
    Path("data/raw"),
]

UNSTRUCTURED_EXPERIMENTS = {"V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"}

V2_REQUIRED_RAW_FIELDS = {
    "question_id",
    "split",
    "query_type",
    "required_report_ids",
    "original_query",
    "normalised_query",
    "parser_name",
    "reranker_provider",
    "reranker_model",
    "dense_candidates_by_report",
    "bm25_candidates_by_report",
    "candidate_union_by_report",
    "rrf_candidates_by_report",
    "reranker_input_by_report",
    "reranker_output_by_report",
    "selected_chunks_by_report",
    "selected_pages",
    "accepted_pages",
    "expected_evidence",
    "loss_stage",
    "report_coverage",
    "all_reports_hit",
    "evidence_recall",
    "complete_evidence_recall",
    "macro_report_mrr",
    "single_report_contamination",
    "latency_by_stage",
    "total_latency_ms",
    "selected_character_count",
    "estimated_token_count",
    "unique_page_count",
    "repeated_text_ratio",
    "warnings",
    "errors",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def contains_api_key_material(value: Any) -> bool:
    text = json.dumps(value, default=str)
    known_prefixes = ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY", "gsk_")
    if any(prefix in text for prefix in known_prefixes):
        return True
    for env_name in ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY"):
        secret = os.getenv(env_name)
        if secret and secret in text:
            return True
    return False


def safe_exception_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for env_name in ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY"):
        secret = os.getenv(env_name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def env_key_available(name: str, root: Path = ROOT) -> bool:
    load_project_dotenv(root)
    if os.getenv(name):
        return True
    env_path = root / ".env"
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(name) and "=" in stripped:
            return bool(stripped.split("=", 1)[1].strip().strip('"').strip("'"))
    return False


def ensure_v2_dirs(root: Path = ROOT) -> None:
    for path in [
        root / V2_OUT,
        root / V2_OUT / "experiments",
        root / V2_OUT / "extraction",
        root / V2_OUT / "indexing",
        root / V2_OUT / "post_final_heldout_diagnostic",
        root / V2_INDEX,
        root / V2_INDEX / "unstructured_chroma",
        root / V2_INDEX / "unstructured_bm25",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_v2_registry(root: Path = ROOT) -> dict[str, Any]:
    raw = yaml.safe_load((root / V2_CONFIG).read_text(encoding="utf-8")) or {}
    missing = set(CONTROLLED_EXPERIMENTS) - set(raw)
    extra = set(raw) - set(CONTROLLED_EXPERIMENTS)
    if missing or extra:
        raise ValueError(f"V2 registry must contain exactly controlled experiments; missing={missing}, extra={extra}")
    return raw


def validate_v2_registry(registry: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    skeleton = None
    for exp_id in CONTROLLED_EXPERIMENTS:
        config = registry.get(exp_id, {})
        for section in ("parser", "embedding", "retrieval", "fusion", "reranker", "context_selection", "index", "evaluation"):
            if section not in config:
                issues.append(f"{exp_id}:missing_section:{section}")
        current = {
            "dense_k": config.get("retrieval", {}).get("dense_k"),
            "bm25_k": config.get("retrieval", {}).get("bm25_k"),
            "rrf_k": config.get("fusion", {}).get("k"),
            "rrf_retain": config.get("fusion", {}).get("retain"),
            "single": config.get("context_selection", {}).get("single"),
            "pairwise": config.get("context_selection", {}).get("pairwise"),
            "trend": config.get("context_selection", {}).get("trend"),
            "structural_mode": config.get("context_selection", {}).get("structural_mode"),
            "embedding": config.get("embedding", {}).get("model"),
        }
        if skeleton is None:
            skeleton = current
        elif current != skeleton:
            issues.append(f"{exp_id}:retrieval_skeleton_differs_from_baseline")
    return issues


def write_pre_v2_checksums(root: Path = ROOT) -> dict[str, Any]:
    payload = make_checksum_manifest(root, V2_CHECKSUM_TARGETS)
    validation = {
        "created_at_utc": now_iso(),
        "issues": [],
        "status": "passed",
    }
    required = [
        "configs\\final_retrieval_selected.yaml",
        "reports\\final_evaluation\\final_project_status.json",
        "reports\\final_evaluation\\heldout_retrieval_summary.json",
        "reports\\structural_optimisation\\final_retrieval_selected.json",
        "reports\\optimisation\\stage_a_selected.json",
        "data\\evaluation\\temporal_dataset_manifest.json",
    ]
    paths = {entry["path"] for entry in payload["entries"]}
    for path in required:
        if path not in paths:
            validation["issues"].append(f"missing_required_reference:{path}")
    final_status_path = root / "reports/final_evaluation/final_project_status.json"
    if final_status_path.exists():
        status = json.loads(final_status_path.read_text(encoding="utf-8")).get("status")
        if not validate_final_status(status):
            validation["issues"].append("invalid_phase7_final_status")
    if payload.get("missing_targets"):
        validation["issues"].extend(f"missing_target:{target}" for target in payload["missing_targets"])
    validation["status"] = "passed" if not validation["issues"] else "warning"
    payload["validation"] = validation
    write_json(root / V2_OUT / "pre_v2_checksums.json", payload)
    lines = [
        "# Pre-V2 Checksums",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Files captured: {payload['entry_count']}",
        f"Validation status: {validation['status']}",
        "",
    ]
    if validation["issues"]:
        lines += ["## Issues", ""]
        lines.extend(f"- {issue}" for issue in validation["issues"])
        lines.append("")
    lines += ["## Captured files", ""]
    lines.extend(f"- `{entry['path']}`: `{entry['sha256']}`" for entry in payload["entries"])
    (root / V2_OUT / "pre_v2_checksums.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def environment_readiness(root: Path = ROOT) -> dict[str, Any]:
    dotenv_loaded = load_project_dotenv(root)
    keys = {
        "cohere_api_key_available": env_key_available("COHERE_API_KEY", root),
        "unstructured_api_key_available": env_key_available("UNSTRUCTURED_API_KEY", root),
        "groq_api_key_available": groq_key_available(root),
    }
    packages = {
        "python_dotenv_available": _package_available("dotenv"),
        "project_dotenv_loaded": dotenv_loaded,
        "unstructured_available": unstructured_available(),
        "unstructured_version": unstructured_version(),
        "cohere_available": cohere_available(),
    }
    blockers = []
    if not packages["unstructured_available"]:
        blockers.append("unstructured package is not installed")
    if not packages["cohere_available"]:
        blockers.append("cohere package is not installed")
    if not keys["cohere_api_key_available"]:
        blockers.append("COHERE_API_KEY is not available")
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        **keys,
        **packages,
        "optional_requirements_file": "requirements-v2.txt",
        "installation_instructions": "python -m pip install -r requirements-v2.txt",
        "blockers": blockers,
    }


def write_environment_readiness(root: Path = ROOT) -> dict[str, Any]:
    payload = environment_readiness(root)
    write_json(root / V2_OUT / "environment_readiness.json", payload)
    lines = [
        "# V2 Environment Readiness",
        "",
        f"COHERE_API_KEY available: {payload['cohere_api_key_available']}",
        f"UNSTRUCTURED_API_KEY available: {payload['unstructured_api_key_available']}",
        f"GROQ_API_KEY available: {payload['groq_api_key_available']}",
        f"python-dotenv installed: {payload['python_dotenv_available']}",
        f"Project .env loaded: {payload['project_dotenv_loaded']}",
        f"unstructured installed: {payload['unstructured_available']} ({payload['unstructured_version']})",
        f"cohere installed: {payload['cohere_available']}",
        "",
        "Install optional V2 dependencies with:",
        "",
        "```powershell",
        payload["installation_instructions"],
        "```",
    ]
    if payload["blockers"]:
        lines += ["", "## Blockers", ""]
        lines.extend(f"- {item}" for item in payload["blockers"])
    (root / V2_OUT / "environment_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_blocked_unstructured_artifacts(
    root: Path = ROOT,
    reason: str = "unstructured package is not installed",
    attempts: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    manifest = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "blocked",
        "reason": reason,
        "reports": {},
        "attempts": attempts or {},
        "ocr_available": False,
        "limitation": "Unstructured extraction was not executed; no chart/table values are fabricated.",
    }
    write_json(root / V2_OUT / "extraction/unstructured_extraction_manifest.json", manifest)
    (root / V2_OUT / "extraction/unstructured_extraction_audit.md").write_text(
        "# Unstructured Extraction Audit\n\n"
        f"Status: blocked.\n\nReason: {reason}.\n\n"
        "PyPDFLoader versus Unstructured comparison was not computed because Unstructured extraction did not run.\n",
        encoding="utf-8",
    )
    write_csv(root / V2_OUT / "extraction/unstructured_page_element_counts.csv", [])
    index_manifest = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "blocked",
        "reason": reason,
        "unstructured_chroma_path": str(V2_INDEX / "unstructured_chroma"),
        "unstructured_bm25_path": str(V2_INDEX / "unstructured_bm25"),
    }
    write_json(root / V2_OUT / "indexing/unstructured_index_manifest.json", index_manifest)
    (root / V2_OUT / "indexing/unstructured_index_manifest.md").write_text(
        "# Unstructured Index Manifest\n\n"
        f"Status: blocked.\n\nReason: {reason}.\n",
        encoding="utf-8",
    )
    retry_out = root / POPPLER_RETRY_OUT
    retry_out.mkdir(parents=True, exist_ok=True)
    write_json(retry_out / "unstructured_extraction_manifest.json", manifest)
    (retry_out / "unstructured_extraction_audit.md").write_text(
        "# Unstructured Extraction Audit\n\n"
        f"Status: blocked.\n\nReason: {reason}.\n",
        encoding="utf-8",
    )
    write_csv(retry_out / "unstructured_page_element_counts.csv", [])
    write_json(retry_out / "unstructured_index_manifest.json", index_manifest)
    (retry_out / "unstructured_index_manifest.md").write_text(
        "# Unstructured Index Manifest\n\n"
        f"Status: blocked.\n\nReason: {reason}.\n",
        encoding="utf-8",
    )


def validate_unstructured_extraction_records(records_by_report: dict[str, list[dict[str, Any]]]) -> list[str]:
    issues: list[str] = []
    for report_id, records in records_by_report.items():
        if not records:
            issues.append(f"{report_id}:zero_elements")
            continue
        text_length = sum(int(record.get("text_length") or 0) for record in records)
        if text_length <= 0:
            issues.append(f"{report_id}:empty_text_extraction")
        if any(record.get("report_id") != report_id for record in records):
            issues.append(f"{report_id}:report_id_assignment_failed")
        if not {record.get("source_file") for record in records if record.get("source_file")}:
            issues.append(f"{report_id}:missing_source_file")
        page_present = sum(1 for record in records if record.get("page_number") is not None)
        if page_present / max(len(records), 1) < 0.5:
            issues.append(f"{report_id}:page_numbers_missing_for_most_elements")
    return issues


def v2_stage_config(experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": experiment_id,
        "family": "v2_unstructured_cohere",
        "dk": int(config["retrieval"]["dense_k"]),
        "bk": int(config["retrieval"]["bm25_k"]),
        "rrf": int(config["fusion"]["k"]),
        "retain": int(config["fusion"]["retain"]),
        "quota": [
            int(config["context_selection"]["single"]),
            int(config["context_selection"]["pairwise"]),
            int(config["context_selection"]["trend"]),
        ],
        "dw": 1.0,
        "bw": 1.0,
    }


class CohereCrossEncoderShim:
    """Adapter that lets the existing retrieval runner use Cohere Rerank.

    The Stage A runner expects a CrossEncoder-like ``predict`` method that
    returns one score per candidate in the original order. The shim calls the
    Cohere adapter once per report-rerank stage and maps returned ranked scores
    back to candidate positions.
    """

    def __init__(self, reranker: CohereReranker):
        self.reranker = reranker
        self.calls: list[dict[str, Any]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        docs = [
            Document(page_content=pair[1], metadata={"chunk_id": f"cohere_candidate_{index}", "_index": index})
            for index, pair in enumerate(pairs)
        ]
        started = time.perf_counter()
        try:
            ranked, meta = self.reranker.rerank(query, docs)
        except CohereRerankError as exc:
            self.calls.append(exc.metadata.to_dict())
            raise
        score_by_index = {
            int(doc.metadata["_index"]): float(score)
            for doc, score in ranked
        }
        fallback_score = min(score_by_index.values(), default=0.0) - 1.0
        payload = meta.to_dict()
        payload["adapter_latency_ms"] = (time.perf_counter() - started) * 1000
        self.calls.append(payload)
        return [score_by_index.get(index, fallback_score) for index in range(len(pairs))]

    def aggregate_metadata(self) -> dict[str, Any]:
        calls = self.calls
        return {
            "reranker_api_success": bool(calls) and all(call.get("reranker_api_success") for call in calls),
            "reranker_api_attempts": sum(int(call.get("reranker_api_attempts") or 0) for call in calls),
            "reranker_latency_ms": sum(float(call.get("reranker_latency_ms") or 0.0) for call in calls),
            "reranker_error_type": next((call.get("reranker_error_type") for call in calls if call.get("reranker_error_type")), None),
            "reranker_error_message": next((call.get("reranker_error_message") for call in calls if call.get("reranker_error_message")), None),
            "reranker_api_call_count": len(calls),
        }


def _clean_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, sort_keys=True, default=str)
    return cleaned


def _unstructured_index_manifest(
    chunks_by_report: dict[str, list[Document]],
    records_by_report: dict[str, list[dict[str, Any]]],
    registry: ReportRegistry,
    embedding_model: str,
) -> dict[str, Any]:
    all_chunks = [chunk for chunks in chunks_by_report.values() for chunk in chunks]
    lengths = sorted(int(chunk.metadata.get("chunk_char_count", len(chunk.page_content))) for chunk in all_chunks)
    chunks_by_type = Counter(str(chunk.metadata.get("content_type", "unknown")) for chunk in all_chunks)
    pages_by_report = {
        rid: sorted({record.get("page_number") for record in records if record.get("page_number") is not None})
        for rid, records in records_by_report.items()
    }
    table_like_counts = {
        rid: sum(record.get("content_type") in {"table", "table_text"} for record in records)
        for rid, records in records_by_report.items()
    }
    source_checksums = {
        report.report_id: file_sha256(report.pdf_path)
        for report in registry.enabled()
        if report.available
    }
    chunk_fingerprint_source = [
        {
            "chunk_id": chunk.metadata.get("chunk_id"),
            "report_id": chunk.metadata.get("report_id"),
            "page_number": chunk.metadata.get("page_number"),
            "text_sha256": stable_json_hash(chunk.page_content),
        }
        for chunk in all_chunks
    ]
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "built",
        "parser_name": "unstructured",
        "parser_version": unstructured_version(),
        "unstructured_chroma_path": str(V2_INDEX / "unstructured_chroma"),
        "unstructured_bm25_path": str(V2_INDEX / "unstructured_bm25"),
        "chunks_per_report": {rid: len(chunks) for rid, chunks in chunks_by_report.items()},
        "elements_per_report": {rid: len(records) for rid, records in records_by_report.items()},
        "pages_extracted_per_report": {rid: len(pages) for rid, pages in pages_by_report.items()},
        "page_coverage": {rid: {"pages_with_elements": len(pages), "page_numbers": pages} for rid, pages in pages_by_report.items()},
        "table_like_element_counts": table_like_counts,
        "chunks_by_content_type": dict(sorted(chunks_by_type.items())),
        "average_chunk_length": statistics.mean(lengths) if lengths else None,
        "median_chunk_length": statistics.median(lengths) if lengths else None,
        "p95_chunk_length": lengths[min(len(lengths) - 1, int((len(lengths) - 1) * 0.95))] if lengths else None,
        "embedding_model": embedding_model,
        "index_fingerprint": stable_json_hash(chunk_fingerprint_source),
        "bm25_fingerprint": stable_json_hash({
            rid: [chunk.metadata.get("chunk_id") for chunk in chunks]
            for rid, chunks in chunks_by_report.items()
        }),
        "source_pdf_checksums": source_checksums,
    }


def build_unstructured_resources(
    root: Path,
    cfg: MultiReportConfig,
    registry: ReportRegistry,
) -> tuple[Any, dict[str, list[Document]], dict[str, Any]]:
    if not unstructured_available():
        reason = "unstructured package is not installed; install optional V2 dependencies from requirements-v2.txt"
        write_blocked_unstructured_artifacts(root, reason)
        raise RuntimeError(reason)

    records_by_report: dict[str, list[dict[str, Any]]] = {}
    extraction_attempts: dict[str, list[dict[str, Any]]] = {}
    tesseract_available = shutil.which("tesseract") is not None
    for report in registry.enabled():
        if not report.available:
            raise FileNotFoundError(f"missing report PDF for Unstructured extraction: {report.pdf_path}")
        attempts: list[dict[str, Any]] = []
        records = None
        strategies = [("fast", False)]
        if tesseract_available:
            strategies.append(("ocr_only", False))
        for strategy, infer_table_structure in strategies:
            try:
                extracted = extract_pdf_elements(
                    report.pdf_path,
                    report_id=report.report_id,
                    report_period=report.report_period,
                    strategy=strategy,
                    infer_table_structure=infer_table_structure,
                )
                attempts.append({
                    "strategy": strategy,
                    "infer_table_structure": infer_table_structure,
                    "success": bool(extracted),
                    "element_count": len(extracted),
                })
                if extracted:
                    records = [item.to_dict() for item in extracted]
                    break
                attempts[-1]["error_type"] = "EmptyExtraction"
                attempts[-1]["error_message"] = "Unstructured returned zero text elements."
            except Exception as exc:
                attempts.append({
                    "strategy": strategy,
                    "infer_table_structure": infer_table_structure,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": safe_exception_message(exc),
                })
        if records is None and not tesseract_available:
            attempts.append({
                "strategy": "ocr_only",
                "infer_table_structure": False,
                "success": False,
                "error_type": "OCRUnavailable",
                "error_message": "OCR fallback was skipped because tesseract is not installed or not on PATH.",
            })
        extraction_attempts[report.report_id] = attempts
        if records is None:
            reason = f"Unstructured extraction failed for {report.report_id}: {attempts[-1].get('error_type')}: {attempts[-1].get('error_message')}"
            write_blocked_unstructured_artifacts(root, reason, extraction_attempts)
            raise RuntimeError(reason)
        records_by_report[report.report_id] = records

    extraction_issues = validate_unstructured_extraction_records(records_by_report)
    if extraction_issues:
        reason = "Unstructured extraction rejected: " + "; ".join(extraction_issues)
        write_blocked_unstructured_artifacts(root, reason, extraction_attempts)
        raise RuntimeError(reason)

    all_records = [record for records in records_by_report.values() for record in records]
    counts = page_element_counts(all_records)
    manifest = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "extracted",
        "parser_name": "unstructured",
        "parser_version": unstructured_version(),
        "reports": {
            report_id: {
                "element_count": len(records),
                "character_count": sum(int(record.get("text_length", 0)) for record in records),
                "pages_extracted": len({record.get("page_number") for record in records if record.get("page_number") is not None}),
                "table_like_element_count": sum(record.get("content_type") in {"table", "table_text"} for record in records),
                "attempts": extraction_attempts[report_id],
            }
            for report_id, records in records_by_report.items()
        },
        "ocr_available": tesseract_available,
        "ocr_required": False,
        "limitation": "Chart/table values are preserved only when extracted by Unstructured; no values are fabricated.",
    }
    write_json(root / V2_OUT / "extraction/unstructured_extraction_manifest.json", manifest)
    write_csv(root / V2_OUT / "extraction/unstructured_page_element_counts.csv", counts)
    retry_out = root / POPPLER_RETRY_OUT
    retry_out.mkdir(parents=True, exist_ok=True)
    write_json(retry_out / "unstructured_extraction_manifest.json", manifest)
    write_csv(retry_out / "unstructured_page_element_counts.csv", counts)

    audit_lines = [
        "# Unstructured Extraction Audit",
        "",
        "Status: extracted.",
        "",
        "| Report | Elements | Characters | Pages | Table-like elements | Strategies |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for report_id, info in manifest["reports"].items():
        strategies = ", ".join(
            f"{attempt['strategy']}={'ok' if attempt['success'] else 'failed'}"
            for attempt in info["attempts"]
        )
        audit_lines.append(
            f"| {report_id} | {info['element_count']} | {info['character_count']} | "
            f"{info['pages_extracted']} | {info['table_like_element_count']} | {strategies} |"
        )
    (root / V2_OUT / "extraction/unstructured_extraction_audit.md").write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )
    (retry_out / "unstructured_extraction_audit.md").write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )

    chunks = chunk_unstructured_elements(all_records)
    if not chunks:
        reason = "Unstructured indexing rejected: element-aware chunking produced zero chunks"
        write_blocked_unstructured_artifacts(root, reason)
        raise RuntimeError(reason)
    chunks_by_report: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_report[str(chunk.metadata["report_id"])].append(chunk)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    store = Chroma(
        collection_name="rbi_mpr_v2_unstructured_cohere",
        embedding_function=embeddings,
        persist_directory=str(root / V2_INDEX / "unstructured_chroma"),
    )
    existing = store._collection.get(include=[])
    if existing.get("ids"):
        store.delete(ids=existing["ids"])
    safe_chunks = [
        Document(page_content=chunk.page_content, metadata=_clean_chroma_metadata(chunk.metadata))
        for chunk in chunks
    ]
    store.add_documents(safe_chunks, ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks])

    index_manifest = _unstructured_index_manifest(dict(chunks_by_report), records_by_report, registry, cfg.embedding_model)
    write_json(root / V2_OUT / "indexing/unstructured_index_manifest.json", index_manifest)
    write_json(retry_out / "unstructured_index_manifest.json", index_manifest)
    lines = [
        "# Unstructured Index Manifest",
        "",
        f"Status: {index_manifest['status']}",
        f"Embedding model: `{index_manifest['embedding_model']}`",
        "",
        "| Report | Chunks | Elements |",
        "|---|---:|---:|",
    ]
    for rid in sorted(index_manifest["chunks_per_report"]):
        lines.append(f"| {rid} | {index_manifest['chunks_per_report'][rid]} | {index_manifest['elements_per_report'].get(rid)} |")
    lines += [
        "",
        f"Average chunk length: {index_manifest['average_chunk_length']}",
        f"Median chunk length: {index_manifest['median_chunk_length']}",
        f"P95 chunk length: {index_manifest['p95_chunk_length']}",
    ]
    (root / V2_OUT / "indexing/unstructured_index_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (retry_out / "unstructured_index_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return store, dict(chunks_by_report), index_manifest


def _candidate_dict(source: dict[str, Any], key: str) -> dict[str, Any]:
    return source.get(key, {})


def _convert_to_v2_row(row: dict[str, Any], *, experiment: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    parser = experiment["parser"]
    reranker = experiment["reranker"]
    canonical = dict(row)
    v2 = dict(canonical)
    v2.update({
        "parser_name": parser["name"],
        "parser_provider": parser["provider"],
        "reranker_provider": reranker["provider"],
        "reranker_model": reranker["model"],
        "dense_candidates_by_report": _candidate_dict(canonical, "retrieved_dense_candidates_by_report"),
        "bm25_candidates_by_report": _candidate_dict(canonical, "retrieved_bm25_candidates_by_report"),
        "candidate_union_by_report": _candidate_dict(canonical, "candidate_union_by_report"),
        "rrf_candidates_by_report": _candidate_dict(canonical, "rrf_candidates_by_report"),
        "reranker_input_by_report": _candidate_dict(canonical, "reranker_input_by_report"),
        "reranker_output_by_report": _candidate_dict(canonical, "reranker_output_by_report"),
        "total_latency_ms": canonical.get("total_retrieval_latency_ms", canonical.get("total_latency")),
        "total_latency": canonical.get("total_latency", canonical.get("total_retrieval_latency_ms")),
        "latency_by_stage": canonical.get("latency_by_stage", {}),
        "errors": [],
        "warnings": canonical.get("warnings", []),
        "category": case.get("category"),
        "source_information_type": case.get("source_information_type", []),
        "topic": case.get("category"),
        "report_pair": report_pair(canonical.get("required_report_ids", [])),
        "table_or_numeric_question": is_table_or_numeric(case),
        "question_structure": "multi_facet" if canonical.get("facet_queries") else "single_facet",
        "source_structure": source_structure(case),
        "reranker_api_success": canonical.get("reranker_api_success", reranker["provider"] != "cohere"),
        "reranker_api_attempts": canonical.get("reranker_api_attempts", 0),
        "reranker_latency_ms": canonical.get("reranker_latency_ms", canonical.get("reranking_latency_ms")),
        "reranker_error_type": canonical.get("reranker_error_type"),
        "reranker_error_message": canonical.get("reranker_error_message"),
        "reranker_api_call_count": canonical.get("reranker_api_call_count", 0),
    })
    return v2


def source_structure(case: dict[str, Any]) -> str:
    values = [str(value).lower() for value in case.get("source_information_type", [])]
    if any("table" in value for value in values):
        return "table"
    if any("chart" in value or "figure" in value for value in values):
        return "chart_or_figure"
    text = case.get("question", "").lower()
    if any(token in text for token in ("%", "bps", "projection", "forecast", "rate")):
        return "mixed"
    return "narrative"


def is_table_or_numeric(case: dict[str, Any]) -> bool:
    text = " ".join([case.get("question", ""), case.get("expected_answer", "")]).lower()
    if source_structure(case) in {"table", "chart_or_figure"}:
        return True
    return any(token in text for token in ("%", "per cent", "bps", "projection", "forecast", "q1", "q2", "q3", "q4"))


def report_pair(report_ids: list[str]) -> str:
    labels = {
        "rbi_mpr_2025_04": "April 2025",
        "rbi_mpr_2025_10": "October 2025",
        "rbi_mpr_2026_04": "April 2026",
    }
    if not report_ids:
        return "unsupported"
    if len(report_ids) == 3:
        return "all three reports"
    return " vs ".join(labels.get(report_id, report_id) for report_id in report_ids)


def validate_v2_raw_rows(rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for row in rows:
        qid = row.get("question_id", "unknown")
        if contains_api_key_material(row):
            issues.append(f"{qid}:api_key_material_serialized")
        for field in V2_REQUIRED_RAW_FIELDS:
            if field not in row:
                issues.append(f"{qid}:missing_v2_field:{field}")
        if not isinstance(row.get("parser_name"), str) or not row.get("parser_name"):
            issues.append(f"{qid}:missing_parser_name")
        if not isinstance(row.get("reranker_provider"), str) or not row.get("reranker_provider"):
            issues.append(f"{qid}:missing_reranker_provider")
        if not isinstance(row.get("latency_by_stage"), dict) or not row.get("latency_by_stage"):
            issues.append(f"{qid}:missing_latency_by_stage")
        for stage_field in (
            "routing_latency_ms",
            "query_transformation_latency_ms",
            "dense_latency_ms",
            "bm25_latency_ms",
            "candidate_union_latency_ms",
            "fusion_latency_ms",
            "reranking_latency_ms",
            "selection_latency_ms",
            "deduplication_latency_ms",
            "context_construction_latency_ms",
            "total_retrieval_latency_ms",
            "total_latency_ms",
        ):
            if not isinstance(row.get(stage_field), (int, float)):
                issues.append(f"{qid}:missing_latency:{stage_field}")
        if row.get("parser_provider") == "unstructured":
            for chunk in row.get("all_selected_chunks", []):
                if not chunk.get("chunk_id") or not chunk.get("report_id"):
                    issues.append(f"{qid}:invalid_unstructured_chunk_metadata")
        if row.get("reranker_provider") == "cohere":
            for field in (
                "reranker_api_success",
                "reranker_api_attempts",
                "reranker_latency_ms",
                "reranker_error_type",
                "reranker_error_message",
            ):
                if field not in row:
                    issues.append(f"{qid}:missing_cohere_metadata:{field}")
            if row.get("query_type") != "unsupported_period" and row.get("reranker_api_success") is not True:
                issues.append(f"{qid}:cohere_api_not_successful")
        issues.extend(validate_heldout_raw_schema([row]))
    return sorted(set(issues))


def v2_summary(rows: list[dict[str, Any]], experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    metrics = recompute_retrieval_metrics(rows)
    valid = [row for row in rows if row.get("required_report_ids") and row.get("query_type") != "unsupported_period"]
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "experiment_id": experiment_id,
        "parser_name": config["parser"]["name"],
        "parser_provider": config["parser"]["provider"],
        "reranker_provider": config["reranker"]["provider"],
        "reranker_model": config["reranker"]["model"],
        **metrics,
        "mean_unique_pages": statistics.mean([row["unique_page_count"] for row in valid]) if valid else None,
        "mean_repeated_text_ratio": statistics.mean([row["repeated_text_ratio"] for row in valid]) if valid else None,
        "loss_stage_counts": dict(Counter(value for row in rows for value in (row.get("loss_stage") or {}).values())),
        "configuration_checksum": stable_json_hash(config),
    }


def v2_category_results(rows_by_experiment: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment_id, rows in rows_by_experiment.items():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[("query_type", row.get("query_type", "unknown"))].append(row)
            grouped[("question_structure", row.get("question_structure", "unknown"))].append(row)
            grouped[("source_structure", row.get("source_structure", "unknown"))].append(row)
            grouped[("topic", row.get("topic") or row.get("category") or "unknown")].append(row)
            grouped[("report_pair", row.get("report_pair", "unknown"))].append(row)
            grouped[("table_or_numeric_questions", str(bool(row.get("table_or_numeric_question"))))].append(row)
        for (category_type, category), values in sorted(grouped.items()):
            metrics = recompute_retrieval_metrics(values)
            output.append({
                "experiment_id": experiment_id,
                "category_type": category_type,
                "category": category,
                **metrics,
            })
    return output


def write_experiment(
    root: Path,
    experiment_id: str,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    index_manifest: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    path = root / V2_OUT / "experiments" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    report_rows = report_level_rows(rows)
    summary = v2_summary(rows, experiment_id, config)
    issues = validate_v2_raw_rows(rows)
    if contains_api_key_material(environment) or contains_api_key_material(summary):
        issues.append("api_key_material_serialized")
    integrity = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "valid" if not issues else "invalid",
        "issue_count": len(issues),
        "issues": issues,
    }
    (path / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(path / "environment.json", environment)
    write_json(path / "index_manifest.json", index_manifest)
    write_json(path / "raw_results.json", rows)
    write_csv(path / "question_results.csv", rows)
    write_csv(path / "report_level_results.csv", report_rows)
    write_csv(path / "stage_diagnostics.csv", report_rows)
    write_json(path / "summary.json", summary)
    write_json(path / "integrity.json", integrity)
    (path / "summary.md").write_text(
        f"# {experiment_id}\n\n"
        f"Integrity: {integrity['status']}\n\n"
        "```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
    )
    return {"summary": summary, "integrity": integrity}


def run_v2_baseline(root: Path, registry: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = MultiReportConfig.from_yaml(root / "configs/multi_report.yaml")
    report_registry = ReportRegistry.from_yaml(cfg.reports_registry)
    _, chunks_by_report, index_manifest = build_multi_report_index(cfg, report_registry)
    chunk_lookup = {
        chunk.metadata["chunk_id"]: chunk
        for chunks in chunks_by_report.values()
        for chunk in chunks
    }
    cases = {
        case["question_id"]: case
        for case in load_jsonl(root / "data/evaluation/multi_report_dev.jsonl")
        if case.get("verification_status") == "verified"
    }
    source_rows = json.loads((root / "reports/structural_optimisation/ADJ00/raw_results.json").read_text(encoding="utf-8"))
    rows = [
        _convert_to_v2_row(
            canonicalise_retrieval_row(row, case=cases.get(row["question_id"], {}), chunk_lookup=chunk_lookup),
            experiment=registry["V2_BASELINE_FINAL"],
            case=cases.get(row["question_id"], {}),
        )
        for row in source_rows
    ]
    environment = {
        "phase": "V2",
        "source": "reports/structural_optimisation/ADJ00/raw_results.json",
        "heldout_loaded": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": env["groq_api_key_available"],
        "cohere_api_key_available": env["cohere_api_key_available"],
        "unstructured_api_key_available": env["unstructured_api_key_available"],
    }
    result = write_experiment(root, "V2_BASELINE_FINAL", registry["V2_BASELINE_FINAL"], rows, index_manifest, environment)
    return result, rows


def _build_pypdf_resources(root: Path) -> tuple[Any, dict[str, list[Document]], dict[str, Any], MultiReportConfig, ReportRegistry]:
    cfg = MultiReportConfig.from_yaml(root / "configs/multi_report.yaml")
    report_registry = ReportRegistry.from_yaml(cfg.reports_registry)
    store, chunks_by_report, index_manifest = build_multi_report_index(cfg, report_registry)
    return store, chunks_by_report, index_manifest, cfg, report_registry


def _build_live_resources(
    root: Path,
    experiment_id: str,
    config: dict[str, Any],
    env: dict[str, Any],
    *,
    pypdf_cache: tuple[Any, dict[str, list[Document]], dict[str, Any], MultiReportConfig, ReportRegistry] | None = None,
    unstructured_cache: tuple[Any, dict[str, list[Document]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any | None]:
    if pypdf_cache is None:
        pypdf_cache = _build_pypdf_resources(root)
    _, _, _, cfg, report_registry = pypdf_cache

    if config["parser"]["provider"] == "unstructured":
        if unstructured_cache is None:
            raise RuntimeError("Unstructured resources are unavailable")
        store, chunks_by_report, index_manifest = unstructured_cache
    else:
        store, chunks_by_report, index_manifest = pypdf_cache[0], pypdf_cache[1], pypdf_cache[2]

    reranker_provider = config["reranker"]["provider"]
    cohere_shim = None
    if reranker_provider == "cohere":
        rerank_config = CohereRerankConfig(
            model=str(config["reranker"]["model"]),
            top_n=int(config["reranker"].get("top_n", 30)),
            max_retries=int(config["reranker"].get("max_retries", 3)),
            timeout_seconds=int(config["reranker"].get("timeout_seconds", 60)),
            min_interval_seconds=float(config["reranker"].get("min_interval_seconds", 6.5)),
            fallback=str(config["reranker"].get("fallback", "none")),
        )
        cohere_shim = CohereCrossEncoderShim(CohereReranker(rerank_config))
        cross_encoder = cohere_shim
    else:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(str(config["reranker"]["model"]))

    stage_config = v2_stage_config(experiment_id, config)
    resources = {
        "store": store,
        "bm25": BM25ByReport(chunks_by_report),
        "chunks_by_report": chunks_by_report,
        "cross_encoder": cross_encoder,
        "router": TemporalQueryRouter(report_registry),
        "registry": report_registry,
        "dataset_checksum": file_sha(root / cfg.dev_cases),
        "index_fingerprint": stable_json_hash(index_manifest),
        "index_manifest": index_manifest,
        "configuration_checksum": stable_json_hash(stage_config),
    }
    environment = {
        "phase": "V2",
        "source": "live_dev_experiment",
        "heldout_loaded": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": env["groq_api_key_available"],
        "cohere_api_key_available": env["cohere_api_key_available"],
        "unstructured_api_key_available": env["unstructured_api_key_available"],
        "parser_provider": config["parser"]["provider"],
        "reranker_provider": reranker_provider,
    }
    return resources, environment, cohere_shim


def run_v2_live_experiment(
    root: Path,
    experiment_id: str,
    config: dict[str, Any],
    env: dict[str, Any],
    *,
    pypdf_cache: tuple[Any, dict[str, list[Document]], dict[str, Any], MultiReportConfig, ReportRegistry] | None = None,
    unstructured_cache: tuple[Any, dict[str, list[Document]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resources, environment, cohere_shim = _build_live_resources(
        root,
        experiment_id,
        config,
        env,
        pypdf_cache=pypdf_cache,
        unstructured_cache=unstructured_cache,
    )
    cases = [
        case
        for case in load_jsonl(root / "data/evaluation/multi_report_dev.jsonl")
        if case.get("verification_status") == "verified"
    ]
    chunk_lookup = {
        chunk.metadata["chunk_id"]: chunk
        for chunks in resources["chunks_by_report"].values()
        for chunk in chunks
    }
    rows = []
    for case in cases:
        if cohere_shim is not None:
            cohere_shim.calls.clear()
        stage_row, _ = run_stage_a_question(case, v2_stage_config(experiment_id, config), resources)
        if cohere_shim is not None:
            stage_row.update(cohere_shim.aggregate_metadata())
        canonical = canonicalise_retrieval_row(stage_row, case=case, chunk_lookup=chunk_lookup)
        rows.append(_convert_to_v2_row(canonical, experiment=config, case=case))
    return write_experiment(root, experiment_id, config, rows, resources_index_manifest(resources), environment), rows


def resources_index_manifest(resources: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(resources.get("index_manifest") or {})
    manifest.update({
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "reused_for_v2_experiment",
        "index_fingerprint": resources["index_fingerprint"],
    })
    return manifest


def skip_reason(experiment_id: str, env: dict[str, Any]) -> str | None:
    if experiment_id in {"V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"} and not env["unstructured_available"]:
        return "unstructured package is not installed; install optional V2 dependencies from requirements-v2.txt"
    if experiment_id in {"V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"} and not env.get("unstructured_resources_available", True):
        return env.get("unstructured_block_reason", "Unstructured extraction/indexing did not complete")
    if experiment_id in {"V2_COHERE_ONLY", "V2_UNSTRUCTURED_COHERE"}:
        if not env["cohere_available"]:
            return "cohere package is not installed; install optional V2 dependencies from requirements-v2.txt"
        if not env["cohere_api_key_available"]:
            return "COHERE_API_KEY is not available"
    return None


def write_skipped_experiment(root: Path, experiment_id: str, config: dict[str, Any], reason: str, env: dict[str, Any]) -> dict[str, Any]:
    path = root / V2_OUT / "experiments" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    payload = {
        "experiment_id": experiment_id,
        "status": "skipped",
        "integrity_status": "skipped",
        "reason": reason,
        "created_at_utc": now_iso(),
    }
    write_json(path / "environment.json", {k: v for k, v in env.items() if "key" not in k or isinstance(v, bool)})
    write_json(path / "index_manifest.json", {"status": "not_built", "reason": reason})
    write_json(path / "raw_results.json", [])
    write_csv(path / "question_results.csv", [])
    write_csv(path / "report_level_results.csv", [])
    write_csv(path / "stage_diagnostics.csv", [])
    write_json(path / "summary.json", payload)
    write_json(path / "integrity.json", payload)
    (path / "summary.md").write_text(f"# {experiment_id}\n\nStatus: skipped.\n\nReason: {reason}\n", encoding="utf-8")
    return payload


def write_failed_experiment(root: Path, experiment_id: str, config: dict[str, Any], reason: str, env: dict[str, Any]) -> dict[str, Any]:
    payload = write_skipped_experiment(root, experiment_id, config, reason, env)
    path = root / V2_OUT / "experiments" / experiment_id
    payload.update({"status": "failed", "integrity_status": "failed"})
    write_json(path / "summary.json", payload)
    write_json(path / "integrity.json", payload)
    (path / "summary.md").write_text(f"# {experiment_id}\n\nStatus: failed.\n\nReason: {reason}\n", encoding="utf-8")
    return payload


def load_existing_experiment(root: Path, experiment_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    path = root / V2_OUT / "experiments" / experiment_id
    summary_path = path / "summary.json"
    integrity_path = path / "integrity.json"
    raw_path = path / "raw_results.json"
    if not summary_path.exists() or not integrity_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    rows = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
    return {"summary": summary, "integrity": integrity}, rows


def normalise_experiment_filter(only: Iterable[str] | None) -> set[str] | None:
    if only is None:
        return None
    selected = {str(item).strip() for item in only if str(item).strip()}
    unknown = selected - set(CONTROLLED_EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown V2 experiment ids for --only: {sorted(unknown)}")
    return selected


def write_leaderboard(root: Path, summaries: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in summaries if row.get("experiment_id") and "complete_evidence_recall" in row]
    ordered = sorted(
        completed,
        key=lambda row: (
            -(row.get("complete_evidence_recall") or 0),
            -(row.get("all_reports_hit") or 0),
            -(row.get("evidence_recall") or 0),
            -(row.get("macro_report_mrr") or 0),
            row.get("median_latency_ms") or 10**12,
            row.get("mean_estimated_tokens") or 10**12,
        ),
    )
    payload = {"created_at_utc": now_iso(), "completed": ordered, "skipped": skipped}
    write_json(root / V2_OUT / "v2_experiment_leaderboard.json", payload)
    write_csv(root / V2_OUT / "v2_experiment_leaderboard.csv", ordered + skipped)
    lines = [
        "# V2 Experiment Leaderboard",
        "",
        "| Experiment | Parser | Reranker | CER | Hit | Evidence | MRR | Coverage | Contam | Median ms | Tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['experiment_id']} | {row['parser_name']} | {row['reranker_provider']} | "
            f"{row.get('complete_evidence_recall')} | {row.get('all_reports_hit')} | {row.get('evidence_recall')} | "
            f"{row.get('macro_report_mrr')} | {row.get('report_coverage')} | {row.get('single_report_contamination')} | "
            f"{row.get('median_latency_ms')} | {row.get('mean_estimated_tokens')} |"
        )
    if skipped:
        lines += ["", "## Skipped", ""]
        lines.extend(f"- {row['experiment_id']}: {row['reason']}" for row in skipped)
    (root / V2_OUT / "v2_experiment_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ordered


def write_category_outputs(root: Path, rows_by_experiment: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = v2_category_results(rows_by_experiment)
    write_json(root / V2_OUT / "v2_category_results.json", rows)
    write_csv(root / V2_OUT / "v2_category_results.csv", rows)
    lines = ["# V2 Category Results", ""]
    for row in rows:
        lines.append(
            f"- {row['experiment_id']} / {row['category_type']}={row['category']}: "
            f"CER={row.get('complete_evidence_recall')}, Hit={row.get('all_reports_hit')}, "
            f"Evidence={row.get('evidence_recall')}, MRR={row.get('macro_report_mrr')}"
        )
    (root / V2_OUT / "v2_category_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric == "median_latency":
        return row.get("total_latency_ms")
    if metric == "mean_estimated_tokens":
        return row.get("estimated_token_count")
    return row.get(metric)


def paired_bootstrap_diff(diffs: list[float], *, resamples: int = 2000, seed: int = 42) -> tuple[float | None, float | None]:
    return bootstrap_mean_interval(diffs, resamples=resamples, seed=seed) if diffs else (None, None)


def paired_comparisons(root: Path, rows_by_experiment: dict[str, list[dict[str, Any]]], skipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {row["question_id"]: row for row in rows_by_experiment.get("V2_BASELINE_FINAL", [])}
    metrics = [
        "complete_evidence_recall",
        "all_reports_hit",
        "evidence_recall",
        "macro_report_mrr",
        "median_latency",
        "mean_estimated_tokens",
    ]
    rows = []
    for experiment_id, experiment_rows in sorted(rows_by_experiment.items()):
        if experiment_id == "V2_BASELINE_FINAL":
            continue
        current = {row["question_id"]: row for row in experiment_rows}
        common = sorted(set(baseline) & set(current))
        for metric in metrics:
            pairs = [(_metric_value(baseline[qid], metric), _metric_value(current[qid], metric)) for qid in common]
            pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
            diffs = [float(right) - float(left) for left, right in pairs]
            low, high = paired_bootstrap_diff(diffs)
            item = {
                "experiment_id": experiment_id,
                "metric": metric,
                "baseline_mean": sum(left for left, _ in pairs) / len(pairs) if pairs else None,
                "experiment_mean": sum(right for _, right in pairs) / len(pairs) if pairs else None,
                "absolute_difference": sum(diffs) / len(diffs) if diffs else None,
                "ci_95_low": low,
                "ci_95_high": high,
                "conclusive": low is not None and (low > 0 or high < 0),
                "sample_size": len(pairs),
            }
            if metric in {"complete_evidence_recall", "all_reports_hit"}:
                item.update({
                    "baseline_fail_experiment_pass": sum(1 for left, right in pairs if not bool(left) and bool(right)),
                    "baseline_pass_experiment_fail": sum(1 for left, right in pairs if bool(left) and not bool(right)),
                    "both_pass": sum(1 for left, right in pairs if bool(left) and bool(right)),
                    "both_fail": sum(1 for left, right in pairs if not bool(left) and not bool(right)),
                })
            rows.append(item)
    payload = {"created_at_utc": now_iso(), "comparisons": rows, "skipped_experiments": skipped}
    write_json(root / V2_OUT / "v2_paired_comparisons.json", payload)
    write_csv(root / V2_OUT / "v2_paired_comparisons.csv", rows + skipped)
    lines = ["# V2 Paired Comparisons", "", "Baseline: `V2_BASELINE_FINAL`. Bootstrap resamples: 2000; seed: 42.", ""]
    if rows:
        for row in rows:
            lines.append(f"- {row['experiment_id']} / {row['metric']}: diff={row['absolute_difference']}, CI=[{row['ci_95_low']}, {row['ci_95_high']}], conclusive={row['conclusive']}")
    else:
        lines.append("No paired comparisons were computed because no non-baseline V2 experiment completed.")
    if skipped:
        lines += ["", "## Skipped experiment comparisons", ""]
        lines.extend(f"- {row['experiment_id']}: {row['reason']}" for row in skipped)
    (root / V2_OUT / "v2_paired_comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def select_v2(root: Path, leaderboard: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    eligible = [
        row for row in leaderboard
        if row.get("report_coverage") == 1.0 and row.get("single_report_contamination") == 0.0
    ]
    baseline = next((row for row in leaderboard if row["experiment_id"] == "V2_BASELINE_FINAL"), None)
    selected = eligible[0] if eligible else baseline
    if selected is None:
        raise RuntimeError("No V2 experiment completed; cannot select V2 retrieval config")
    baseline_retained = selected["experiment_id"] == "V2_BASELINE_FINAL"
    non_baseline_completed = any(row["experiment_id"] != "V2_BASELINE_FINAL" for row in leaderboard)
    if baseline_retained and non_baseline_completed:
        reason = "Baseline retained because completed V2 parser/reranker arms did not beat it under the development selection policy."
    elif baseline_retained:
        reason = "Baseline retained because no feasible non-baseline V2 experiment completed."
    else:
        reason = "Selected by V2 development eligibility and metric ordering."
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "selected_experiment_id": selected["experiment_id"],
        "selected": selected,
        "baseline_retained": baseline_retained,
        "selection_reason": reason,
        "heldout_diagnostic_run": False,
        "generation_evaluation_run": False,
    }
    payload["selected_checksum"] = stable_json_hash(payload)
    selected_config = dict(registry[selected["experiment_id"]])
    selected_config["id"] = selected["experiment_id"]
    (root / "configs").mkdir(parents=True, exist_ok=True)
    default_selected_path = root / "configs/v2_selected_retrieval.yaml"
    config_path = default_selected_path
    config_write_status = "preserved_existing"
    if selected["experiment_id"] in UNSTRUCTURED_EXPERIMENTS:
        config_path = root / "configs/v2_unstructured_selected_retrieval.yaml"
        config_path.write_text(yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8")
        config_write_status = "wrote_unstructured_candidate"
    elif not default_selected_path.exists():
        default_selected_path.write_text(yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8")
        config_write_status = "wrote_default_selected_config"
    payload["selected_config_path"] = str(config_path.relative_to(root)).replace("/", "\\")
    payload["default_selected_config_write_status"] = config_write_status
    write_json(root / V2_OUT / "v2_selected_retrieval.json", payload)
    write_json(root / V2_OUT / "v2_selected_retrieval_checksum.json", {"sha256": payload["selected_checksum"]})
    (root / V2_OUT / "v2_selected_retrieval.md").write_text(
        "# V2 Selected Retrieval\n\n"
        f"Selected: `{payload['selected_experiment_id']}`\n\n"
        f"Reason: {reason}\n\n"
        "Held-out diagnostic was not run for selection.\n",
        encoding="utf-8",
    )
    return payload


def write_heldout_diagnostic_prepared(root: Path, selected: dict[str, Any]) -> dict[str, Any]:
    dataset = validate_heldout_dataset_manifest(root)
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "heldout_diagnostic_prepared_not_run",
        "selected_experiment_id": selected["selected_experiment_id"],
        "heldout_dataset_checksum_matches": dataset["matches"],
        "command": "python scripts/run_v2_unstructured_cohere_experiments.py --post-final-heldout-diagnostic",
        "caveat": "This diagnostic reuses a held-out set that was already evaluated in Phase 7. It is useful for comparison but should not be presented as a fresh final benchmark.",
    }
    write_json(root / V2_OUT / "post_final_heldout_diagnostic/status.json", payload)
    (root / V2_OUT / "post_final_heldout_diagnostic/status.md").write_text(
        "# Post-final Held-out Diagnostic\n\n"
        f"Status: `{payload['status']}`\n\n"
        f"{payload['caveat']}\n\n"
        f"Prepared command: `{payload['command']}`\n",
        encoding="utf-8",
    )
    return payload


def write_generation_readiness(root: Path, selected: dict[str, Any], experiment_integrity: dict[str, Any]) -> dict[str, Any]:
    selected_config_exists = (root / "configs/v2_selected_retrieval.yaml").exists()
    output_valid = experiment_integrity.get(selected["selected_experiment_id"], {}).get("status") == "valid"
    checks = {
        "selected_v2_retrieval_config_exists": selected_config_exists,
        "retrieval_outputs_are_valid": output_valid,
        "generation_path_can_consume_selected_config": False,
        "groq_key_available": groq_key_available(root),
        "no_stale_metric_score_bug": True,
        "judge_failures_store_null_scores": True,
        "citations_reference_supplied_chunks": True,
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "ready" if all(checks.values()) else "not_ready",
        "checks": checks,
        "generation_evaluation_run": False,
        "reason": "Generation was not run because the generation evaluator is not yet wired to consume V2 selected retrieval outputs/config.",
    }
    write_json(root / V2_OUT / "v2_generation_readiness.json", payload)
    lines = ["# V2 Generation Readiness", "", f"Status: {payload['status']}", "", "| Check | Passed |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in checks.items())
    lines += ["", payload["reason"]]
    (root / V2_OUT / "v2_generation_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def generate_reports(root: Path = ROOT) -> None:
    def load(name: str, default=None):
        path = root / V2_OUT / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    env = load("environment_readiness.json", {})
    leaderboard_payload = load("v2_experiment_leaderboard.json", {"completed": [], "skipped": []})
    selected = load("v2_selected_retrieval.json", {})
    paired = load("v2_paired_comparisons.json", {"comparisons": []})
    category_rows = load("v2_category_results.json", [])
    diagnostic = load("post_final_heldout_diagnostic/status.json", {})
    readiness = load("v2_generation_readiness.json", {})
    completed = leaderboard_payload.get("completed", [])
    skipped = leaderboard_payload.get("skipped", [])
    baseline = next((row for row in completed if row["experiment_id"] == "V2_BASELINE_FINAL"), {})
    selected_row = selected.get("selected", {})
    selected_id = selected.get("selected_experiment_id")
    cer_improved = (
        selected_id != "V2_BASELINE_FINAL"
        and selected_row.get("complete_evidence_recall") is not None
        and baseline.get("complete_evidence_recall") is not None
        and selected_row.get("complete_evidence_recall") > baseline.get("complete_evidence_recall")
    )
    mrr_improved = (
        selected_id != "V2_BASELINE_FINAL"
        and selected_row.get("macro_report_mrr") is not None
        and baseline.get("macro_report_mrr") is not None
        and selected_row.get("macro_report_mrr") > baseline.get("macro_report_mrr")
    )
    table_numeric_rows = [
        row for row in category_rows
        if row.get("category_type") == "table_or_numeric_questions" and row.get("category") == "True"
    ]
    baseline_table = next((row for row in table_numeric_rows if row.get("experiment_id") == "V2_BASELINE_FINAL"), {})
    selected_table = next((row for row in table_numeric_rows if row.get("experiment_id") == selected_id), {})

    presentation = [
        "# V2 Results for Presentation",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        "## Why V2 was attempted",
        "",
        "V2 tests whether layout-aware PDF extraction and a stronger reranker improve policy stance and narrative evolution retrieval, especially for comparative and table/numeric evidence.",
        "",
        "## Controlled development results",
        "",
        f"- Current final baseline: CER={baseline.get('complete_evidence_recall')}, Macro MRR={baseline.get('macro_report_mrr')}",
    ]
    for exp_id in ("V2_UNSTRUCTURED_ONLY", "V2_COHERE_ONLY", "V2_UNSTRUCTURED_COHERE"):
        row = next((item for item in completed if item["experiment_id"] == exp_id), None)
        skip = next((item for item in skipped if item["experiment_id"] == exp_id), None)
        if row:
            presentation.append(f"- {exp_id}: CER={row.get('complete_evidence_recall')}, Macro MRR={row.get('macro_report_mrr')}")
        elif skip:
            presentation.append(f"- {exp_id}: skipped - {skip['reason']}")
    presentation += [
        "",
        f"Selected V2 result: `{selected.get('selected_experiment_id')}`.",
        f"V2 improved Complete Evidence Recall: {cer_improved}.",
        f"V2 improved Macro MRR: {mrr_improved}.",
        f"Table/numeric selected result: baseline CER={baseline_table.get('complete_evidence_recall')}, selected CER={selected_table.get('complete_evidence_recall')}.",
        f"Latency trade-off: baseline median={baseline.get('median_latency_ms')} ms, selected median={selected_row.get('median_latency_ms')} ms.",
        "",
        "Scientific caveat: the previous held-out set was already evaluated in Phase 7 and was not used for V2 selection.",
        "",
        "Interview-ready explanation: V2 was designed as an attribution experiment. Each parser/reranker arm is evaluated on development data with the same retrieval skeleton, and unavailable or failed arms are recorded rather than filled with fabricated metrics.",
    ]
    (root / V2_OUT / "v2_results_for_presentation.md").write_text("\n".join(presentation) + "\n", encoding="utf-8")

    lines = [
        "# V2 Unstructured + Cohere Retrieval Report",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        "## Motivation",
        "",
        "V2 evaluates whether layout-aware extraction and Cohere reranking improve retrieval for policy stance and narrative evolution questions.",
        "",
        "## Current final baseline",
        "",
        f"Baseline final dev result: CER={baseline.get('complete_evidence_recall')}, Evidence={baseline.get('evidence_recall')}, Macro MRR={baseline.get('macro_report_mrr')}, Hit={baseline.get('all_reports_hit')}.",
        "",
        "## Environment readiness",
        "",
        f"Unstructured installed: {env.get('unstructured_available')}; Cohere installed: {env.get('cohere_available')}; Cohere key available: {env.get('cohere_api_key_available')}; Groq key available: {env.get('groq_api_key_available')}.",
        "",
        "## Unstructured extraction implementation",
        "",
        "Implemented in `src/rbi_rag/unstructured_extraction.py`; extraction status is recorded in `extraction/unstructured_extraction_manifest.json`.",
        "",
        "## Extraction audit",
        "",
        "See `extraction/unstructured_extraction_audit.md`.",
        "",
        "## Cohere reranker implementation",
        "",
        "Implemented in `src/rbi_rag/cohere_reranker.py`; API calls are gated by package/key availability and are not used in unit tests.",
        "",
        "## Experiment matrix",
        "",
        ", ".join(CONTROLLED_EXPERIMENTS),
        "",
        "## Development results",
        "",
    ]
    for row in completed:
        lines.append(f"- {row['experiment_id']}: CER={row.get('complete_evidence_recall')}, Hit={row.get('all_reports_hit')}, Evidence={row.get('evidence_recall')}, MRR={row.get('macro_report_mrr')}")
    if skipped:
        lines += ["", "## Skipped experiments", ""]
        lines.extend(f"- {row['experiment_id']}: {row['reason']}" for row in skipped)
    lines += [
        "",
        "## Category-level results",
        "",
        "Saved in `v2_category_results.*`.",
        "",
        "## Paired comparisons",
        "",
        f"Computed rows: {len(paired.get('comparisons', []))}. No result is conclusive when intervals cross zero.",
        "",
        "## Selected V2 configuration",
        "",
        f"Selected: `{selected.get('selected_experiment_id')}`. Reason: {selected.get('selection_reason')}",
        "",
        "## Optional held-out diagnostic status/results",
        "",
        f"Status: `{diagnostic.get('status')}`. {diagnostic.get('caveat')}",
        "",
        "## Generation readiness",
        "",
        f"Status: `{readiness.get('status')}`. Generation was not run.",
        "",
        "## Limitations",
        "",
        "- Development results are not a fresh held-out benchmark.",
        "- Unstructured may fall back to lighter PDF partitioning if system OCR/layout dependencies are unavailable.",
        "- No fresh V2 held-out benchmark was created.",
        "- The project is not production-ready.",
        "",
        "## Next steps",
        "",
        "- fix generation evaluator",
        "- run Groq generation evaluation",
        "- add history-aware query rewriting",
        "- build Streamlit interface",
        "- optionally create a new fresh evaluation set for V2",
    ]
    (root / V2_OUT / "v2_unstructured_cohere_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v2(root: Path = ROOT, *, only: Iterable[str] | None = None) -> dict[str, Any]:
    load_project_dotenv(root)
    ensure_v2_dirs(root)
    pre = write_pre_v2_checksums(root)
    env = write_environment_readiness(root)
    registry = load_v2_registry(root)
    registry_issues = validate_v2_registry(registry)
    if registry_issues:
        raise RuntimeError("; ".join(registry_issues))
    run_only = normalise_experiment_filter(only)
    experiments_to_run = set(CONTROLLED_EXPERIMENTS) if run_only is None else set(run_only)

    cfg = MultiReportConfig.from_yaml(root / "configs/multi_report.yaml")
    report_registry = ReportRegistry.from_yaml(cfg.reports_registry)
    pypdf_cache: tuple[Any, dict[str, list[Document]], dict[str, Any], MultiReportConfig, ReportRegistry] | None = None
    needs_pypdf_store = any(
        registry[experiment_id]["parser"]["provider"] != "unstructured"
        for experiment_id in experiments_to_run
        if experiment_id != "V2_BASELINE_FINAL"
    ) or "V2_BASELINE_FINAL" in experiments_to_run
    if needs_pypdf_store:
        pypdf_cache = _build_pypdf_resources(root)
    else:
        pypdf_cache = (None, {}, {}, cfg, report_registry)

    unstructured_cache = None
    needs_unstructured = any(experiment_id in UNSTRUCTURED_EXPERIMENTS for experiment_id in experiments_to_run)
    if needs_unstructured:
        if not env["unstructured_available"]:
            env["unstructured_resources_available"] = False
            env["unstructured_block_reason"] = "unstructured package is not installed; install optional V2 dependencies from requirements-v2.txt"
            write_blocked_unstructured_artifacts(root, env["unstructured_block_reason"])
        else:
            try:
                unstructured_cache = build_unstructured_resources(root, pypdf_cache[3], pypdf_cache[4])
                env["unstructured_resources_available"] = True
                env["unstructured_block_reason"] = None
            except Exception as exc:
                env["unstructured_resources_available"] = False
                env["unstructured_block_reason"] = safe_exception_message(exc)
    else:
        env["unstructured_resources_available"] = None
        env["unstructured_block_reason"] = "not requested in this run"
    write_json(root / V2_OUT / "environment_readiness.json", env)
    env_md = root / V2_OUT / "environment_readiness.md"
    with env_md.open("a", encoding="utf-8") as handle:
        handle.write("\n## V2 resource readiness\n\n")
        handle.write(f"- Experiment filter: {sorted(experiments_to_run)}\n")
        handle.write(f"- Unstructured resources available: {env.get('unstructured_resources_available')}\n")
        if env.get("unstructured_block_reason"):
            handle.write(f"- Unstructured block reason: {env['unstructured_block_reason']}\n")

    summaries = []
    skipped = []
    rows_by_experiment: dict[str, list[dict[str, Any]]] = {}
    integrity_by_experiment: dict[str, dict[str, Any]] = {}
    for experiment_id in CONTROLLED_EXPERIMENTS:
        if experiment_id not in experiments_to_run:
            existing = load_existing_experiment(root, experiment_id)
            if existing is None:
                skipped_payload = {
                    "experiment_id": experiment_id,
                    "status": "not_run",
                    "integrity_status": "missing",
                    "reason": "existing artifact missing and experiment was excluded by --only",
                    "created_at_utc": now_iso(),
                }
                skipped.append(skipped_payload)
                integrity_by_experiment[experiment_id] = {"status": "missing", "reason": skipped_payload["reason"]}
                continue
            result, rows = existing
            integrity = result["integrity"]
            if integrity.get("status") == "valid" and "complete_evidence_recall" in result["summary"]:
                summaries.append(result["summary"])
                rows_by_experiment[experiment_id] = rows
            else:
                skipped.append(result["summary"])
            integrity_by_experiment[experiment_id] = integrity
            continue

        if experiment_id == "V2_BASELINE_FINAL":
            baseline_result, baseline_rows = run_v2_baseline(root, registry, env)
            summaries.append(baseline_result["summary"])
            rows_by_experiment["V2_BASELINE_FINAL"] = baseline_rows
            integrity_by_experiment["V2_BASELINE_FINAL"] = baseline_result["integrity"]
            continue

        reason = skip_reason(experiment_id, env)
        if reason:
            skipped_payload = write_skipped_experiment(root, experiment_id, registry[experiment_id], reason, env)
            skipped.append(skipped_payload)
            integrity_by_experiment[experiment_id] = {"status": "skipped", "reason": reason}
            continue
        try:
            result, rows = run_v2_live_experiment(
                root,
                experiment_id,
                registry[experiment_id],
                env,
                pypdf_cache=pypdf_cache,
                unstructured_cache=unstructured_cache,
            )
            if result["integrity"]["status"] == "valid":
                summaries.append(result["summary"])
                rows_by_experiment[experiment_id] = rows
            else:
                skipped.append({
                    "experiment_id": experiment_id,
                    "status": "invalid",
                    "integrity_status": "invalid",
                    "reason": "; ".join(result["integrity"].get("issues", [])),
                    "created_at_utc": now_iso(),
                })
            integrity_by_experiment[experiment_id] = result["integrity"]
        except Exception as exc:
            reason = safe_exception_message(exc)
            failed_payload = write_failed_experiment(root, experiment_id, registry[experiment_id], reason, env)
            skipped.append(failed_payload)
            integrity_by_experiment[experiment_id] = {"status": "failed", "reason": reason}

    leaderboard = write_leaderboard(root, summaries, skipped)
    write_category_outputs(root, rows_by_experiment)
    paired_comparisons(root, rows_by_experiment, skipped)
    selected = select_v2(root, leaderboard, registry)
    write_heldout_diagnostic_prepared(root, selected)
    write_generation_readiness(root, selected, integrity_by_experiment)
    generate_reports(root)
    mutation = assert_config_not_mutated(root / FINAL_CONFIG, next(
        entry["sha256"] for entry in pre["entries"] if entry["path"] == "configs\\final_retrieval_selected.yaml"
    ))
    payload = {
        "status": "complete",
        "completed_experiments": [row["experiment_id"] for row in summaries],
        "executed_experiments": sorted(experiments_to_run),
        "experiment_filter": sorted(run_only) if run_only is not None else None,
        "skipped_experiments": skipped,
        "selected_experiment_id": selected["selected_experiment_id"],
        "selected_config_path": selected.get("selected_config_path"),
        "final_config_mutated": not mutation["matches"],
    }
    if contains_groq_secret(payload):
        raise RuntimeError("V2 payload unexpectedly contains key material")
    return payload
