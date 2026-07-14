from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from langchain_core.documents import Document

from .env_loading import load_project_dotenv
from .final_evaluation import (
    contains_groq_secret,
    file_sha,
    report_level_rows,
    stable_json_hash,
    write_csv,
    write_json,
)
from .multi_config import MultiReportConfig
from .multi_index import build_multi_report_index
from .report_registry import ReportRegistry


MMR_OUT = Path("reports/mmr_experiments")
MMR_CONFIG = Path("configs/mmr_experiments.yaml")
V2_COHERE_RAW = Path("reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY/raw_results.json")
V2_COHERE_SUMMARY = Path("reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY/summary.json")

MMR_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "MMR_BASELINE_V2_COHERE": {"mmr_enabled": False, "mmr_lambda": None},
    "MMR_LAMBDA_06": {"mmr_enabled": True, "mmr_lambda": 0.6},
    "MMR_LAMBDA_07": {"mmr_enabled": True, "mmr_lambda": 0.7},
    "MMR_LAMBDA_08": {"mmr_enabled": True, "mmr_lambda": 0.8},
}

MMR_RAW_REQUIRED_FIELDS = {
    "question_id",
    "split",
    "query_type",
    "required_report_ids",
    "original_query",
    "normalised_query",
    "selected_chunks_by_report",
    "selected_pages",
    "accepted_pages",
    "expected_evidence",
    "retrieval_complete_evidence_recall",
    "retrieval_evidence_recall",
    "retrieval_all_reports_hit",
    "retrieval_macro_mrr",
    "report_coverage",
    "single_report_contamination",
    "loss_stage",
    "latency_by_stage",
    "total_latency_ms",
    "estimated_token_count",
    "unique_page_count",
    "repeated_text_ratio",
    "mmr_enabled",
    "mmr_lambda",
    "mmr_selected_chunk_ids",
    "mmr_rejected_chunk_ids",
    "mmr_trace",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def ensure_mmr_config(root: Path = Path(".")) -> dict[str, Any]:
    config = {
        "description": "True Maximal Marginal Relevance experiments over saved V2_COHERE_ONLY development reranker outputs.",
        "source_experiment": "V2_COHERE_ONLY",
        "source_raw_results": str(V2_COHERE_RAW).replace("/", "\\"),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_skeleton": {
            "parser": "PyPDFLoader",
            "dense_candidates_per_report": 50,
            "bm25_candidates_per_report": 50,
            "rrf_retained_per_report": 30,
            "reranker": "Cohere rerank-v3.5",
            "reranker_input_per_report": 30,
            "rrf_k": 60,
            "single_report_quota": 6,
            "pairwise_quota_per_report": 5,
            "trend_quota_per_report": 4,
            "adjacent_expansion": "none",
        },
        "experiments": MMR_EXPERIMENTS,
        "evaluation": {
            "split": "dev",
            "heldout_loaded": False,
            "generation_run": False,
            "bootstrap_resamples": 2000,
            "seed": 42,
        },
    }
    write_yaml(root / MMR_CONFIG, config)
    return config


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise_text(text: str) -> str:
    return " ".join(_tokens(text))


def token_cosine(a: str, b: str) -> float:
    ca = Counter(_tokens(a))
    cb = Counter(_tokens(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[token] * cb.get(token, 0) for token in ca)
    norm_a = math.sqrt(sum(value * value for value in ca.values()))
    norm_b = math.sqrt(sum(value * value for value in cb.values()))
    return float(dot / (norm_a * norm_b)) if norm_a and norm_b else 0.0


class SimilarityEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.provider = "deterministic_token_cosine_fallback"
        self.error: str | None = None
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.provider = "sentence_transformers"
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = f"{type(exc).__name__}: {exc}"

    def similarity_matrix(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            matrix: list[list[float]] = []
            for i in range(len(texts)):
                row: list[float] = []
                for j in range(len(texts)):
                    try:
                        row.append(float(vectors[i] @ vectors[j]))
                    except Exception:
                        row.append(0.0)
                matrix.append(row)
            return matrix
        return [[1.0 if i == j else token_cosine(a, b) for j, b in enumerate(texts)] for i, a in enumerate(texts)]


@dataclass
class Candidate:
    chunk_id: str
    report_id: str
    page_number: int | None
    text: str
    reranker_score: float
    rank: int
    normalised_relevance_score: float = 0.0


def normalise_scores(candidates: list[Candidate]) -> None:
    scores = [candidate.reranker_score for candidate in candidates]
    if not scores:
        return
    low, high = min(scores), max(scores)
    if high == low:
        for candidate in candidates:
            candidate.normalised_relevance_score = 1.0
        return
    for candidate in candidates:
        candidate.normalised_relevance_score = (candidate.reranker_score - low) / (high - low)


def quota_for_row(row: dict[str, Any]) -> int:
    required = row.get("required_report_ids") or []
    query_type = row.get("query_type")
    if query_type == "single_report" or len(required) == 1:
        return 6
    if query_type == "pairwise_comparison" or len(required) == 2:
        return 5
    return 4


def mmr_select(
    candidates: list[Candidate],
    *,
    quota: int,
    lambda_value: float,
    similarity_matrix: list[list[float]],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    normalise_scores(candidates)
    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))
    trace_by_index: dict[int, dict[str, Any]] = {}

    while remaining and len(selected_indices) < quota:
        scored: list[tuple[float, float, int, int]] = []
        for idx in sorted(remaining):
            candidate = candidates[idx]
            max_similarity = max((similarity_matrix[idx][j] for j in selected_indices), default=0.0)
            mmr_score = lambda_value * candidate.normalised_relevance_score - (1.0 - lambda_value) * max_similarity
            scored.append((mmr_score, candidate.normalised_relevance_score, -candidate.rank, idx))
            trace_by_index[idx] = {
                "chunk_id": candidate.chunk_id,
                "report_id": candidate.report_id,
                "page_number": candidate.page_number,
                "reranker_score": candidate.reranker_score,
                "normalised_relevance_score": candidate.normalised_relevance_score,
                "max_similarity_to_selected": max_similarity,
                "mmr_score": mmr_score,
                "selected": False,
                "selection_rank": None,
                "rejection_reason": "lower_mmr_score",
            }
        scored.sort(reverse=True)
        best_idx = scored[0][3]
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        trace_by_index[best_idx]["selected"] = True
        trace_by_index[best_idx]["selection_rank"] = len(selected_indices)
        trace_by_index[best_idx]["rejection_reason"] = None

    for idx in remaining:
        entry = trace_by_index.get(idx)
        if entry is None:
            candidate = candidates[idx]
            entry = {
                "chunk_id": candidate.chunk_id,
                "report_id": candidate.report_id,
                "page_number": candidate.page_number,
                "reranker_score": candidate.reranker_score,
                "normalised_relevance_score": candidate.normalised_relevance_score,
                "max_similarity_to_selected": max((similarity_matrix[idx][j] for j in selected_indices), default=0.0),
                "mmr_score": None,
                "selected": False,
                "selection_rank": None,
                "rejection_reason": "quota_full",
            }
        if len(selected_indices) >= quota:
            entry["rejection_reason"] = (
                "duplicate_or_near_duplicate"
                if entry.get("max_similarity_to_selected", 0.0) >= 0.97
                else "quota_full"
            )
        trace_by_index[idx] = entry

    selected = [candidates[idx] for idx in selected_indices]
    trace = [trace_by_index[idx] for idx in sorted(trace_by_index, key=lambda i: (trace_by_index[i].get("selection_rank") or 10_000, candidates[i].rank, candidates[i].chunk_id))]
    return selected, trace


def build_chunk_lookup(root: Path) -> dict[str, Document]:
    cfg = MultiReportConfig.from_yaml(root / "configs/multi_report.yaml")
    registry = ReportRegistry.from_yaml(cfg.reports_registry)
    _, chunks_by_report, _ = build_multi_report_index(cfg, registry)
    return {
        str(chunk.metadata["chunk_id"]): chunk
        for chunks in chunks_by_report.values()
        for chunk in chunks
    }


def candidates_for_report(row: dict[str, Any], report_id: str, chunk_lookup: dict[str, Document]) -> list[Candidate]:
    output = []
    for item in (row.get("reranker_output_by_report") or {}).get(report_id, []):
        chunk_id = item.get("chunk_id")
        doc = chunk_lookup.get(chunk_id)
        if not chunk_id or doc is None:
            continue
        output.append(
            Candidate(
                chunk_id=chunk_id,
                report_id=report_id,
                page_number=item.get("page"),
                text=doc.page_content,
                reranker_score=float(item.get("score") or 0.0),
                rank=int(item.get("rank") or len(output) + 1),
            )
        )
    return output


def _chunk_payload(candidate: Candidate, chunk_lookup: dict[str, Document]) -> dict[str, Any]:
    doc = chunk_lookup[candidate.chunk_id]
    return {
        "chunk_id": candidate.chunk_id,
        "report_id": candidate.report_id,
        "report_period": doc.metadata.get("report_period"),
        "page": candidate.page_number,
        "text": candidate.text,
    }


def _contains_expected(chunks: list[dict[str, Any]], expected: list[str]) -> list[bool]:
    combined = _normalise_text(" ".join(chunk.get("text") or "" for chunk in chunks))
    return [_normalise_text(text) in combined for text in expected]


def _first_page_rank(pages: list[int | None], accepted: list[int]) -> int | None:
    accepted_set = set(accepted)
    for index, page in enumerate(pages):
        if page in accepted_set:
            return index + 1
    return None


def recompute_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    required = list(row.get("required_report_ids") or [])
    if not required or row.get("query_type") == "unsupported_period":
        return {
            "retrieval_complete_evidence_recall": None,
            "retrieval_evidence_recall": None,
            "retrieval_all_reports_hit": None,
            "retrieval_macro_mrr": 0.0,
            "report_coverage": None,
            "single_report_contamination": False,
            "loss_stage": {},
        }

    selected_by_report = row.get("selected_chunks_by_report") or {}
    accepted_by_report = row.get("accepted_pages") or {}
    expected_by_report = row.get("expected_evidence") or {}
    report_hits: list[bool] = []
    evidence_hits: list[bool] = []
    reciprocal: list[float] = []
    loss_stage: dict[str, str] = {}
    for report_id in required:
        chunks = selected_by_report.get(report_id, [])
        pages = [chunk.get("page") for chunk in chunks]
        accepted = accepted_by_report.get(report_id, [])
        expected = expected_by_report.get(report_id, [])
        rank = _first_page_rank(pages, accepted)
        if accepted:
            report_hits.append(bool(rank))
        reciprocal.append(1.0 / rank if rank else 0.0)
        hits = _contains_expected(chunks, expected)
        evidence_hits.extend(hits)
        if hits and all(hits):
            loss_stage[report_id] = "evidence_found"
        elif row.get("mmr_enabled"):
            loss_stage[report_id] = "lost_by_mmr_selection"
        else:
            loss_stage[report_id] = (row.get("loss_stage") or {}).get(report_id, "lost_by_quota")
    selected_reports = {report_id for report_id, chunks in selected_by_report.items() if chunks}
    contamination = bool(row.get("query_type") == "single_report" and (selected_reports - set(required)))
    return {
        "retrieval_complete_evidence_recall": all(evidence_hits) if evidence_hits else None,
        "retrieval_evidence_recall": sum(evidence_hits) / len(evidence_hits) if evidence_hits else None,
        "retrieval_all_reports_hit": all(report_hits) if report_hits else None,
        "retrieval_macro_mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
        "report_coverage": len(selected_reports & set(required)) / len(required) if required else None,
        "single_report_contamination": contamination,
        "loss_stage": loss_stage,
    }


def context_stats(row: dict[str, Any]) -> dict[str, Any]:
    chunks = [chunk for values in (row.get("selected_chunks_by_report") or {}).values() for chunk in values]
    texts = [_normalise_text(chunk.get("text") or "") for chunk in chunks]
    non_empty = [text for text in texts if text]
    repeated = len(non_empty) - len(set(non_empty))
    pages = {(chunk.get("report_id"), chunk.get("page")) for chunk in chunks if chunk.get("page") is not None}
    chars = sum(len(chunk.get("text") or "") for chunk in chunks)
    return {
        "selected_character_count": chars,
        "estimated_token_count": int(chars / 4),
        "selected_chunk_count": len(chunks),
        "unique_page_count": len(pages),
        "repeated_text_ratio": repeated / len(non_empty) if non_empty else 0.0,
    }


def build_mmr_row(
    source_row: dict[str, Any],
    *,
    experiment_id: str,
    mmr_enabled: bool,
    mmr_lambda: float | None,
    chunk_lookup: dict[str, Document],
    similarity_engine: SimilarityEngine,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    row = copy.deepcopy(source_row)
    row["experiment_id"] = experiment_id
    row["mmr_enabled"] = mmr_enabled
    row["mmr_lambda"] = mmr_lambda
    row["split"] = row.get("split") or "dev"
    required = list(row.get("required_report_ids") or [])
    selected_by_report: dict[str, list[dict[str, Any]]] = {}
    selected_ids: list[str] = []
    rejected_ids: list[str] = []
    all_trace: list[dict[str, Any]] = []

    if not required or row.get("query_type") == "unsupported_period":
        selected_by_report = {}
    elif not mmr_enabled:
        for report_id in required:
            chunks = []
            selected_existing = (source_row.get("selected_chunks_by_report") or {}).get(report_id, [])
            existing_ids = [item.get("chunk_id") for item in selected_existing]
            for chunk_id in existing_ids:
                doc = chunk_lookup.get(chunk_id)
                if doc is None:
                    continue
                candidate = Candidate(
                    chunk_id=chunk_id,
                    report_id=report_id,
                    page_number=doc.metadata.get("page"),
                    text=doc.page_content,
                    reranker_score=0.0,
                    rank=len(chunks) + 1,
                    normalised_relevance_score=0.0,
                )
                chunks.append(_chunk_payload(candidate, chunk_lookup))
                selected_ids.append(chunk_id)
            selected_by_report[report_id] = chunks
            for item in (source_row.get("reranker_output_by_report") or {}).get(report_id, []):
                chunk_id = item.get("chunk_id")
                trace = {
                    "chunk_id": chunk_id,
                    "report_id": report_id,
                    "page_number": item.get("page"),
                    "reranker_score": item.get("score"),
                    "normalised_relevance_score": None,
                    "max_similarity_to_selected": None,
                    "mmr_score": None,
                    "selected": chunk_id in existing_ids,
                    "selection_rank": existing_ids.index(chunk_id) + 1 if chunk_id in existing_ids else None,
                    "rejection_reason": None if chunk_id in existing_ids else "quota_full",
                }
                all_trace.append(trace)
                if chunk_id not in existing_ids:
                    rejected_ids.append(chunk_id)
    else:
        for report_id in required:
            candidates = candidates_for_report(source_row, report_id, chunk_lookup)
            texts = [candidate.text for candidate in candidates]
            matrix = similarity_engine.similarity_matrix(texts)
            selected, trace = mmr_select(
                candidates,
                quota=quota_for_row(row),
                lambda_value=float(mmr_lambda),
                similarity_matrix=matrix,
            )
            selected_by_report[report_id] = [_chunk_payload(candidate, chunk_lookup) for candidate in selected]
            selected_ids.extend(candidate.chunk_id for candidate in selected)
            rejected_ids.extend(entry["chunk_id"] for entry in trace if not entry["selected"])
            all_trace.extend(trace)

    row["selected_chunks_by_report"] = selected_by_report
    row["selected_pages"] = {
        report_id: [chunk.get("page") for chunk in chunks if chunk.get("page") is not None]
        for report_id, chunks in selected_by_report.items()
    }
    row["all_selected_chunks"] = [chunk for report_id in required for chunk in selected_by_report.get(report_id, [])]
    row["accepted_pages"] = source_row.get("accepted_pages", {})
    row["expected_evidence"] = source_row.get("expected_evidence", {})
    row["mmr_selected_chunk_ids"] = selected_ids
    row["mmr_rejected_chunk_ids"] = rejected_ids
    row["mmr_trace"] = all_trace
    metrics = recompute_row_metrics(row)
    row.update(metrics)
    row["complete_evidence_recall"] = metrics["retrieval_complete_evidence_recall"]
    row["evidence_recall"] = metrics["retrieval_evidence_recall"]
    row["all_reports_hit"] = metrics["retrieval_all_reports_hit"]
    row["macro_report_mrr"] = metrics["retrieval_macro_mrr"]
    row["macro_mrr"] = metrics["retrieval_macro_mrr"]
    row.update(context_stats(row))
    mmr_latency = (time.perf_counter() - start) * 1000
    latency = dict(row.get("latency_by_stage") or {})
    latency["mmr_selection_latency_ms"] = mmr_latency
    row["latency_by_stage"] = latency
    base_total = row.get("total_latency_ms") or row.get("total_retrieval_latency_ms") or row.get("total_latency") or 0.0
    row["total_latency_ms"] = float(base_total) + (mmr_latency if mmr_enabled else 0.0)
    row["total_latency"] = row["total_latency_ms"]
    row["total_retrieval_latency_ms"] = row["total_latency_ms"]
    row["warnings"] = list(row.get("warnings") or [])
    row["errors"] = list(row.get("errors") or [])
    return row, all_trace


def _mean(values: list[float | int | bool | None]) -> float | None:
    items = [float(value) for value in values if isinstance(value, (int, float, bool))]
    return sum(items) / len(items) if items else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))]


def summarise_rows(rows: list[dict[str, Any]], experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("required_report_ids") and row.get("query_type") != "unsupported_period"]
    latencies = [float(row["total_latency_ms"]) for row in valid if isinstance(row.get("total_latency_ms"), (int, float))]
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "experiment_id": experiment_id,
        "mmr_enabled": config["mmr_enabled"],
        "mmr_lambda": config["mmr_lambda"],
        "status": "completed",
        "case_count": len(rows),
        "scored_case_count": len(valid),
        "complete_evidence_recall": _mean([row.get("retrieval_complete_evidence_recall") for row in valid]),
        "all_reports_hit": _mean([row.get("retrieval_all_reports_hit") for row in valid]),
        "evidence_recall": _mean([row.get("retrieval_evidence_recall") for row in valid]),
        "macro_report_mrr": _mean([row.get("retrieval_macro_mrr") for row in valid]),
        "report_coverage": _mean([row.get("report_coverage") for row in valid]),
        "single_report_contamination": _mean([row.get("single_report_contamination") for row in valid]),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "mean_latency_ms": _mean(latencies),
        "p95_latency_ms": _p95(latencies),
        "mean_estimated_tokens": _mean([row.get("estimated_token_count") for row in valid]),
        "mean_unique_pages": _mean([row.get("unique_page_count") for row in valid]),
        "mean_repeated_text_ratio": _mean([row.get("repeated_text_ratio") for row in valid]),
        "mean_selected_chunks": _mean([row.get("selected_chunk_count") for row in valid]),
        "eligibility": "eligible" if _mean([row.get("report_coverage") for row in valid]) == 1.0 and _mean([row.get("single_report_contamination") for row in valid]) == 0.0 else "ineligible",
        "configuration_checksum": stable_json_hash(config),
    }


def write_experiment_artifacts(
    root: Path,
    experiment_id: str,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    out = root / MMR_OUT / "experiments" / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise_rows(rows, experiment_id, config)
    integrity_issues = validate_raw_rows(rows, expected_config=config)
    integrity = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "valid" if not integrity_issues else "invalid",
        "issue_count": len(integrity_issues),
        "issues": integrity_issues,
    }
    write_yaml(out / "config_snapshot.yaml", {"experiment_id": experiment_id, **config})
    write_json(out / "environment.json", environment)
    write_json(out / "raw_results.json", rows)
    write_csv(out / "question_results.csv", rows)
    report_rows = report_level_rows(rows)
    write_csv(out / "report_level_results.csv", report_rows)
    write_csv(out / "stage_diagnostics.csv", report_rows)
    write_json(out / "summary.json", summary)
    write_json(out / "mmr_trace.json", traces)
    write_json(out / "integrity.json", integrity)
    write_markdown(
        out / "summary.md",
        [
            f"# {experiment_id}",
            "",
            f"Status: {summary['status']}",
            f"Integrity: {integrity['status']}",
            f"MMR enabled: {config['mmr_enabled']}",
            f"MMR lambda: {config['mmr_lambda']}",
            "",
            f"CER: {summary['complete_evidence_recall']}",
            f"All-Reports Hit: {summary['all_reports_hit']}",
            f"Evidence Recall: {summary['evidence_recall']}",
            f"Macro MRR: {summary['macro_report_mrr']}",
        ],
    )
    return {"summary": summary, "integrity": integrity}


def validate_raw_rows(rows: list[dict[str, Any]], expected_config: dict[str, Any] | None = None) -> list[str]:
    issues: list[str] = []
    for row in rows:
        qid = row.get("question_id", "unknown")
        if contains_groq_secret(row):
            issues.append(f"{qid}:api_key_serialized")
        for field in MMR_RAW_REQUIRED_FIELDS:
            if field not in row:
                issues.append(f"{qid}:missing_field:{field}")
        if row.get("split") == "heldout":
            issues.append(f"{qid}:heldout_row_present")
        if expected_config:
            if row.get("mmr_enabled") != expected_config["mmr_enabled"]:
                issues.append(f"{qid}:mmr_enabled_mismatch")
            if row.get("mmr_lambda") != expected_config["mmr_lambda"]:
                issues.append(f"{qid}:mmr_lambda_mismatch")
        selected_by_report = row.get("selected_chunks_by_report") or {}
        required = set(row.get("required_report_ids") or [])
        for report_id, chunks in selected_by_report.items():
            if report_id not in required:
                issues.append(f"{qid}:selected_non_required_report:{report_id}")
            for chunk in chunks:
                if chunk.get("report_id") not in required:
                    issues.append(f"{qid}:selected_chunk_non_required_report:{chunk.get('chunk_id')}")
        if not isinstance(row.get("latency_by_stage"), dict) or not isinstance(row.get("total_latency_ms"), (int, float)):
            issues.append(f"{qid}:missing_latency")
        for field in ("estimated_token_count", "unique_page_count", "repeated_text_ratio"):
            if not isinstance(row.get(field), (int, float)):
                issues.append(f"{qid}:missing_context_stat:{field}")
        if row.get("mmr_enabled") and not row.get("mmr_trace") and row.get("query_type") != "unsupported_period":
            issues.append(f"{qid}:missing_mmr_trace")
    return issues


def blocked_artifacts(root: Path, reason: str) -> dict[str, Any]:
    ensure_mmr_config(root)
    out = root / MMR_OUT
    out.mkdir(parents=True, exist_ok=True)
    skipped = []
    for experiment_id, config in MMR_EXPERIMENTS.items():
        exp_out = out / "experiments" / experiment_id
        exp_out.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "experiment_id": experiment_id,
            "mmr_enabled": config["mmr_enabled"],
            "mmr_lambda": config["mmr_lambda"],
            "status": "blocked",
            "reason": reason,
        }
        write_yaml(exp_out / "config_snapshot.yaml", {"experiment_id": experiment_id, **config})
        write_json(exp_out / "environment.json", {"heldout_loaded": False, "generation_run": False})
        write_json(exp_out / "raw_results.json", [])
        write_csv(exp_out / "question_results.csv", [])
        write_csv(exp_out / "report_level_results.csv", [])
        write_csv(exp_out / "stage_diagnostics.csv", [])
        write_json(exp_out / "summary.json", payload)
        write_json(exp_out / "mmr_trace.json", [])
        write_json(exp_out / "integrity.json", {"status": "blocked", "reason": reason, "issue_count": 0, "issues": []})
        write_markdown(exp_out / "summary.md", [f"# {experiment_id}", "", f"Status: blocked", f"Reason: {reason}"])
        skipped.append(payload)
    write_json(out / "mmr_leaderboard.json", {"created_at_utc": now_iso(), "completed": [], "skipped": skipped})
    write_csv(out / "mmr_leaderboard.csv", skipped)
    write_markdown(out / "mmr_leaderboard.md", ["# MMR Leaderboard", "", f"Status: blocked", f"Reason: {reason}"])
    decision = {"schema_version": 1, "created_at_utc": now_iso(), "status": "blocked", "selected_experiment_id": "V2_COHERE_ONLY", "reason": reason}
    write_json(out / "mmr_selection_decision.json", decision)
    write_markdown(out / "mmr_selection_decision.md", ["# MMR Selection Decision", "", "Status: blocked", f"Reason: {reason}"])
    write_json(out / "mmr_paired_comparisons.json", {"comparisons": [], "status": "blocked", "reason": reason})
    write_csv(out / "mmr_paired_comparisons.csv", [])
    write_markdown(out / "mmr_paired_comparisons.md", ["# MMR Paired Comparisons", "", "Status: blocked"])
    write_markdown(out / "mmr_results_for_presentation.md", ["# MMR Results for Presentation", "", f"MMR blocked: {reason}"])
    return {"status": "blocked", "reason": reason}


def paired_bootstrap_diff(baseline: list[float], variant: list[float], *, resamples: int = 2000, seed: int = 42) -> tuple[float | None, float | None]:
    if not baseline or len(baseline) != len(variant):
        return None, None
    import random

    rng = random.Random(seed)
    diffs = []
    n = len(baseline)
    for _ in range(resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(variant[i] - baseline[i] for i in sample) / n)
    diffs.sort()
    return diffs[int(0.025 * (len(diffs) - 1))], diffs[int(0.975 * (len(diffs) - 1))]


def write_paired_comparisons(root: Path, rows_by_experiment: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline = {row["question_id"]: row for row in rows_by_experiment.get("MMR_BASELINE_V2_COHERE", [])}
    comparisons: list[dict[str, Any]] = []
    metrics = [
        "retrieval_complete_evidence_recall",
        "retrieval_all_reports_hit",
        "retrieval_evidence_recall",
        "retrieval_macro_mrr",
        "repeated_text_ratio",
        "unique_page_count",
    ]
    for experiment_id, rows in rows_by_experiment.items():
        if experiment_id == "MMR_BASELINE_V2_COHERE":
            continue
        variant = {row["question_id"]: row for row in rows}
        common = [qid for qid in baseline if qid in variant and baseline[qid].get("query_type") != "unsupported_period"]
        for metric in metrics:
            b_values = [float(baseline[qid].get(metric) or 0.0) for qid in common]
            v_values = [float(variant[qid].get(metric) or 0.0) for qid in common]
            low, high = paired_bootstrap_diff(b_values, v_values)
            row = {
                "experiment_id": experiment_id,
                "baseline_experiment_id": "MMR_BASELINE_V2_COHERE",
                "metric": metric,
                "n": len(common),
                "baseline_mean": _mean(b_values),
                "experiment_mean": _mean(v_values),
                "mean_difference": (_mean(v_values) or 0.0) - (_mean(b_values) or 0.0),
                "ci_95_low": low,
                "ci_95_high": high,
                "resamples": 2000,
                "seed": 42,
            }
            if metric in {"retrieval_complete_evidence_recall", "retrieval_all_reports_hit"}:
                row.update(
                    {
                        "baseline_fail_to_experiment_pass": sum((not bool(baseline[qid].get(metric))) and bool(variant[qid].get(metric)) for qid in common),
                        "baseline_pass_to_experiment_fail": sum(bool(baseline[qid].get(metric)) and not bool(variant[qid].get(metric)) for qid in common),
                        "both_pass": sum(bool(baseline[qid].get(metric)) and bool(variant[qid].get(metric)) for qid in common),
                        "both_fail": sum((not bool(baseline[qid].get(metric))) and not bool(variant[qid].get(metric)) for qid in common),
                    }
                )
            comparisons.append(row)
    out = root / MMR_OUT
    write_json(out / "mmr_paired_comparisons.json", {"created_at_utc": now_iso(), "comparisons": comparisons})
    write_csv(out / "mmr_paired_comparisons.csv", comparisons)
    lines = ["# MMR Paired Comparisons", "", "Intervals use 2,000 paired bootstrap resamples with seed 42.", ""]
    for row in comparisons:
        lines.append(f"- {row['experiment_id']} / {row['metric']}: diff={row['mean_difference']}, 95% CI=({row['ci_95_low']}, {row['ci_95_high']})")
    write_markdown(out / "mmr_paired_comparisons.md", lines)
    return comparisons


def write_leaderboard(root: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        summaries,
        key=lambda row: (
            -(row.get("complete_evidence_recall") or 0),
            -(row.get("all_reports_hit") or 0),
            -(row.get("evidence_recall") or 0),
            -(row.get("macro_report_mrr") or 0),
            row.get("median_latency_ms") or 10**12,
        ),
    )
    rows = []
    for item in ordered:
        rows.append(
            {
                "experiment_id": item["experiment_id"],
                "mmr_enabled": item["mmr_enabled"],
                "mmr_lambda": item["mmr_lambda"],
                "CER": item["complete_evidence_recall"],
                "All-Reports Hit": item["all_reports_hit"],
                "Evidence Recall": item["evidence_recall"],
                "Macro MRR": item["macro_report_mrr"],
                "Report Coverage": item["report_coverage"],
                "Contamination": item["single_report_contamination"],
                "Median latency": item["median_latency_ms"],
                "Mean latency": item["mean_latency_ms"],
                "P95 latency": item["p95_latency_ms"],
                "Mean estimated tokens": item["mean_estimated_tokens"],
                "Unique pages": item["mean_unique_pages"],
                "Repeated-text ratio": item["mean_repeated_text_ratio"],
                "Eligibility": item["eligibility"],
                "Status": item["status"],
            }
        )
    out = root / MMR_OUT
    write_json(out / "mmr_leaderboard.json", {"created_at_utc": now_iso(), "completed": ordered, "skipped": []})
    write_csv(out / "mmr_leaderboard.csv", rows)
    lines = [
        "# MMR Leaderboard",
        "",
        "| Experiment | MMR | Lambda | CER | Hit | Evidence | Macro MRR | Coverage | Contam | Median ms | Tokens | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_id']} | {row['mmr_enabled']} | {row['mmr_lambda']} | {row['CER']} | {row['All-Reports Hit']} | "
            f"{row['Evidence Recall']} | {row['Macro MRR']} | {row['Report Coverage']} | {row['Contamination']} | "
            f"{row['Median latency']} | {row['Mean estimated tokens']} | {row['Status']} |"
        )
    write_markdown(out / "mmr_leaderboard.md", lines)
    return ordered


def select_mmr(root: Path, leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in leaderboard if row["experiment_id"] == "MMR_BASELINE_V2_COHERE")
    variants = [row for row in leaderboard if row["experiment_id"] != "MMR_BASELINE_V2_COHERE"]
    eligible = [
        row
        for row in variants
        if row.get("report_coverage") == 1.0 and row.get("single_report_contamination") == 0.0
    ]
    candidates = [
        row
        for row in eligible
        if (row.get("complete_evidence_recall") or 0) > (baseline.get("complete_evidence_recall") or 0)
        or (row.get("all_reports_hit") or 0) > (baseline.get("all_reports_hit") or 0)
    ]
    selected = None
    status = "evaluated_not_selected"
    reason = "MMR did not improve Complete Evidence Recall or All-Reports Hit over V2_COHERE_ONLY."
    if candidates:
        selected = sorted(
            candidates,
            key=lambda row: (
                -(row.get("complete_evidence_recall") or 0),
                -(row.get("all_reports_hit") or 0),
                -(row.get("evidence_recall") or 0),
                -(row.get("macro_report_mrr") or 0),
                row.get("median_latency_ms") or 10**12,
            ),
        )[0]
        status = "evaluated_selected"
        reason = "MMR improved development retrieval quality while preserving coverage and contamination."
        config = read_json(root / MMR_OUT / "experiments" / selected["experiment_id"] / "summary.json", {})
        selected_config = yaml.safe_load((root / "configs/v2_selected_retrieval.yaml").read_text(encoding="utf-8"))
        selected_config["id"] = selected["experiment_id"]
        selected_config["context_selection"]["mmr_enabled"] = True
        selected_config["context_selection"]["mmr_lambda"] = selected.get("mmr_lambda")
        selected_config["mmr_source_summary"] = config
        write_yaml(root / "configs/v2_mmr_selected_retrieval.yaml", selected_config)
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": status,
        "baseline_experiment_id": "MMR_BASELINE_V2_COHERE",
        "selected_experiment_id": selected["experiment_id"] if selected else "V2_COHERE_ONLY",
        "selected_mmr_experiment": selected,
        "baseline": baseline,
        "reason": reason,
        "v2_selected_config_overwritten": False,
    }
    write_json(root / MMR_OUT / "mmr_selection_decision.json", payload)
    lines = [
        "# MMR Selection Decision",
        "",
        f"Status: `{status}`",
        f"Selected: `{payload['selected_experiment_id']}`",
        "",
        reason,
        "",
        "`configs/v2_selected_retrieval.yaml` was not overwritten.",
    ]
    write_markdown(root / MMR_OUT / "mmr_selection_decision.md", lines)
    return payload


def write_presentation_summary(root: Path, leaderboard: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    by_id = {row["experiment_id"]: row for row in leaderboard}
    lines = [
        "# MMR Results for Presentation",
        "",
        "MMR was tested to see whether a diversity-aware context-selection step could reduce repeated context while preserving evidence completeness.",
        "",
        "Formula: `lambda * relevance_score - (1 - lambda) * max_similarity_to_selected_documents`.",
        "",
        "MRR is Mean Reciprocal Rank, a metric. MMR is Maximal Marginal Relevance, a selection technique.",
        "",
    ]
    for exp_id in MMR_EXPERIMENTS:
        row = by_id.get(exp_id, {})
        lines.append(
            f"- {exp_id}: CER={row.get('complete_evidence_recall')}, Hit={row.get('all_reports_hit')}, "
            f"Evidence={row.get('evidence_recall')}, Macro MRR={row.get('macro_report_mrr')}, "
            f"Repeated={row.get('mean_repeated_text_ratio')}"
        )
    baseline = by_id.get("MMR_BASELINE_V2_COHERE", {})
    best = by_id.get(decision.get("selected_experiment_id"), {}) if decision.get("selected_experiment_id") in by_id else baseline
    lines += [
        "",
        f"MMR improved Hit Rate: {bool((best.get('all_reports_hit') or 0) > (baseline.get('all_reports_hit') or 0))}",
        f"MMR improved Macro MRR: {bool((best.get('macro_report_mrr') or 0) > (baseline.get('macro_report_mrr') or 0))}",
        f"MMR was selected: {decision.get('status') == 'evaluated_selected'}",
        "",
        "Interview-ready explanation: MMR was evaluated as a controlled diversity-selection layer after Cohere reranking. It was only eligible if evidence completeness stayed intact; diversity alone was not enough to replace the selected V2 Cohere retrieval system.",
    ]
    write_markdown(root / MMR_OUT / "mmr_results_for_presentation.md", lines)


def run_mmr_experiments(root: Path = Path(".")) -> dict[str, Any]:
    load_project_dotenv(root)
    ensure_mmr_config(root)
    out = root / MMR_OUT
    out.mkdir(parents=True, exist_ok=True)
    if not (root / V2_COHERE_RAW).exists():
        return blocked_artifacts(root, f"Saved V2_COHERE_ONLY raw results missing: {V2_COHERE_RAW}")
    source_rows = read_json(root / V2_COHERE_RAW, [])
    if not source_rows:
        return blocked_artifacts(root, "Saved V2_COHERE_ONLY raw results are empty")
    try:
        chunk_lookup = build_chunk_lookup(root)
    except Exception as exc:
        return blocked_artifacts(root, f"Candidate text reconstruction failed: {type(exc).__name__}: {exc}")
    if not chunk_lookup:
        return blocked_artifacts(root, "Candidate text reconstruction failed: chunk lookup is empty")
    similarity_engine = SimilarityEngine()
    environment = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "heldout_loaded": False,
        "generation_run": False,
        "cohere_api_key_available": bool(os.getenv("COHERE_API_KEY")),
        "groq_api_key_available": bool(os.getenv("GROQ_API_KEY")),
        "source_raw_results": str(V2_COHERE_RAW).replace("/", "\\"),
        "source_raw_sha256": file_sha(root / V2_COHERE_RAW),
        "similarity_provider": similarity_engine.provider,
        "similarity_model": similarity_engine.model_name,
        "similarity_fallback_error": similarity_engine.error,
    }
    summaries: list[dict[str, Any]] = []
    rows_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for experiment_id, config in MMR_EXPERIMENTS.items():
        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for source_row in source_rows:
            row, trace = build_mmr_row(
                source_row,
                experiment_id=experiment_id,
                mmr_enabled=config["mmr_enabled"],
                mmr_lambda=config["mmr_lambda"],
                chunk_lookup=chunk_lookup,
                similarity_engine=similarity_engine,
            )
            rows.append(row)
            traces.append({"question_id": row.get("question_id"), "trace": trace})
        result = write_experiment_artifacts(root, experiment_id, config, rows, traces, environment)
        summaries.append(result["summary"])
        rows_by_experiment[experiment_id] = rows
    leaderboard = write_leaderboard(root, summaries)
    write_paired_comparisons(root, rows_by_experiment)
    decision = select_mmr(root, leaderboard)
    write_presentation_summary(root, leaderboard, decision)
    return {
        "status": "completed",
        "experiments": list(MMR_EXPERIMENTS),
        "selected_experiment_id": decision["selected_experiment_id"],
        "similarity_provider": similarity_engine.provider,
    }


def validate_mmr_artifacts(root: Path = Path(".")) -> dict[str, Any]:
    out = root / MMR_OUT
    issues: list[str] = []
    checks: list[dict[str, Any]] = []
    for experiment_id, config in MMR_EXPERIMENTS.items():
        exp = out / "experiments" / experiment_id
        required_files = [
            "config_snapshot.yaml",
            "environment.json",
            "raw_results.json",
            "question_results.csv",
            "report_level_results.csv",
            "summary.json",
            "summary.md",
            "stage_diagnostics.csv",
            "mmr_trace.json",
            "integrity.json",
        ]
        for name in required_files:
            if not (exp / name).exists():
                issues.append(f"{experiment_id}:missing_file:{name}")
        rows = read_json(exp / "raw_results.json", [])
        integrity = read_json(exp / "integrity.json", {})
        if integrity.get("status") == "blocked":
            checks.append({"experiment_id": experiment_id, "status": "blocked", "row_count": 0})
            continue
        row_issues = validate_raw_rows(rows, expected_config=config)
        issues.extend(f"{experiment_id}:{issue}" for issue in row_issues)
        checks.append({"experiment_id": experiment_id, "status": "valid" if not row_issues else "invalid", "row_count": len(rows)})
    for name in [
        "mmr_leaderboard.json",
        "mmr_leaderboard.csv",
        "mmr_leaderboard.md",
        "mmr_paired_comparisons.json",
        "mmr_paired_comparisons.csv",
        "mmr_paired_comparisons.md",
        "mmr_selection_decision.json",
        "mmr_selection_decision.md",
        "mmr_results_for_presentation.md",
    ]:
        if not (out / name).exists():
            issues.append(f"missing_artifact:{name}")
    serialized = json.dumps(read_json(out / "mmr_leaderboard.json", {}), default=str)
    for key_name in ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY"):
        secret = os.getenv(key_name)
        if secret and secret in serialized:
            issues.append("api_key_value_serialized")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(set(issues)),
        "issues": sorted(set(issues)),
        "checks": checks,
        "heldout_loaded": False,
        "generation_run": False,
    }
    write_json(out / "mmr_integrity.json", payload)
    lines = ["# MMR Integrity", "", f"Status: {payload['status']}", f"Issues: {payload['issue_count']}", ""]
    if payload["issues"]:
        lines += ["## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    write_markdown(out / "mmr_integrity.md", lines)
    return payload


def generate_mmr_report(root: Path = Path(".")) -> dict[str, Any]:
    out = root / MMR_OUT
    leaderboard = read_json(out / "mmr_leaderboard.json", {"completed": [], "skipped": []})
    decision = read_json(out / "mmr_selection_decision.json", {})
    integrity = read_json(out / "mmr_integrity.json", {})
    lines = [
        "# True MMR Experiment Report",
        "",
        "This report evaluates Maximal Marginal Relevance as a final context-selection layer over saved V2_COHERE_ONLY development reranker outputs.",
        "",
        "MRR is Mean Reciprocal Rank, a metric. MMR is Maximal Marginal Relevance, a selection technique.",
        "",
        "## Formula",
        "",
        "`MMR(document) = lambda * relevance_score - (1 - lambda) * max_similarity_to_selected_documents`",
        "",
        "## Results",
        "",
    ]
    for row in leaderboard.get("completed", []):
        lines.append(f"- {row['experiment_id']}: CER={row.get('complete_evidence_recall')}, Hit={row.get('all_reports_hit')}, MRR={row.get('macro_report_mrr')}")
    for row in leaderboard.get("skipped", []):
        lines.append(f"- {row['experiment_id']}: skipped/blocker={row.get('reason')}")
    lines += [
        "",
        "## Decision",
        "",
        f"Status: `{decision.get('status')}`",
        f"Selected: `{decision.get('selected_experiment_id')}`",
        f"Reason: {decision.get('reason')}",
        "",
        "## Integrity",
        "",
        f"Validation: `{integrity.get('status')}`",
        "",
        "Held-out evaluation was not run. Generation was not run.",
    ]
    write_markdown(out / "mmr_report.md", lines)
    return {"status": "complete", "report": str(out / "mmr_report.md")}

