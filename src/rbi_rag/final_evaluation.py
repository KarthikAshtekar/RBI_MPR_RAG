from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .experiment_tracing import LATENCY_FIELDS, recompute_loss_stage, validate_latency_schema
from .multi_evaluation import load_jsonl
from .uncertainty import bootstrap_mean_interval, wilson_interval


FINAL_STATUS_VALUES = {
    "heldout_retrieval_complete_generation_complete",
    "heldout_retrieval_complete_generation_partial",
    "heldout_retrieval_complete_generation_not_run",
    "heldout_retrieval_failed_generation_not_run",
}

REQUIRED_FINAL_CONFIG = {
    "id": "ADJ00",
    "retrieval.dense_k": 50,
    "retrieval.bm25_k": 50,
    "fusion.retain": 30,
    "fusion.k": 60,
    "context_selection.single": 6,
    "context_selection.pairwise": 5,
    "context_selection.trend": 4,
    "context_selection.structural_mode": "none",
    "chunking.strategy": "recursive",
    "chunking.chunk_size": 1000,
    "chunking.chunk_overlap": 300,
    "embedding.model": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker.model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

CHECKSUM_TARGETS = [
    Path("configs/final_retrieval_selected.yaml"),
    Path("reports/structural_optimisation"),
    Path("reports/optimisation"),
    Path("reports/multi_report"),
    Path("reports/current"),
    Path("data/evaluation"),
]

REQUIRED_TRACE_FIELDS = {
    "dense_candidate_ids",
    "dense_candidate_pages",
    "bm25_candidate_ids",
    "bm25_candidate_pages",
    "candidate_union_ids",
    "candidate_union_pages",
    "rrf_candidate_ids",
    "rrf_candidate_pages",
    "reranker_input_ids",
    "reranker_input_pages",
    "reranker_output_ids",
    "reranker_output_pages",
    "selected_chunk_ids_after_dedup",
    "selected_pages",
    "accepted_pages",
    "expected_evidence",
    "loss_stage",
}

CONTEXT_STAT_FIELDS = {
    "selected_character_count",
    "estimated_token_count",
    "selected_chunk_count",
    "unique_page_count",
    "repeated_text_ratio",
}

GENERATION_REQUIRED_FIELDS = {
    "question_id",
    "split",
    "query_type",
    "required_report_ids",
    "retrieved_contexts",
    "selected_chunk_ids",
    "selected_pages",
    "expected_answer",
    "generated_answer",
    "citations",
    "prompt_version",
    "model_name",
    "temperature",
    "generation_latency_ms",
    "evaluation_metric_results",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def rel_path(path: Path, root: Path = Path(".")) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("/", "\\")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=dict) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, default=dict)
                if isinstance(value, (dict, list, tuple, Counter))
                else value
                for key, value in row.items()
            })


def nested_get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def make_checksum_manifest(root: Path = Path("."), targets: list[Path] | None = None) -> dict[str, Any]:
    targets = targets or CHECKSUM_TARGETS
    entries: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    for target in targets:
        path = root / target
        if path.is_file():
            entries.append({"path": rel_path(path, root), "sha256": file_sha(path)})
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    entries.append({"path": rel_path(child, root), "sha256": file_sha(child)})
        else:
            missing_targets.append(str(target))
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "targets": [str(target).replace("/", "\\") for target in targets],
        "entry_count": len(entries),
        "missing_targets": missing_targets,
        "entries": entries,
    }


def verify_checksum_entries(root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = root / entry["path"]
        exists = path.exists()
        current_sha = file_sha(path) if exists else None
        rows.append({
            "path": entry["path"],
            "expected_sha256": entry["sha256"],
            "actual_sha256": current_sha,
            "exists": exists,
            "matches": exists and current_sha == entry["sha256"],
        })
    failures = [row for row in rows if not row["matches"]]
    return {
        "entry_count": len(rows),
        "match_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "rows": rows,
    }


def _mapping_to_entries(mapping: dict[str, str]) -> list[dict[str, str]]:
    return [{"path": path, "sha256": digest} for path, digest in sorted(mapping.items())]


def validate_heldout_dataset_manifest(
    root: Path = Path("."),
    manifest_path: Path = Path("data/evaluation/temporal_dataset_manifest.json"),
) -> dict[str, Any]:
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    test_info = manifest["files"]["test"]
    test_path = root / test_info["path"]
    cases = load_jsonl(test_path)
    verified = [case for case in cases if case.get("verification_status") == "verified"]
    actual = file_sha(test_path)
    return {
        "path": test_info["path"],
        "expected_sha256": test_info["sha256"],
        "actual_sha256": actual,
        "matches": actual == test_info["sha256"],
        "manifest_case_count": test_info["case_count"],
        "actual_case_count": len(cases),
        "manifest_verified_scored_count": manifest["verified_scored_counts"]["test"],
        "actual_verified_scored_count": len(verified),
        "case_count_matches": len(cases) == test_info["case_count"],
        "verified_count_matches": len(verified) == manifest["verified_scored_counts"]["test"],
    }


def validate_reference_checksums(root: Path = Path(".")) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    pre_optimisation_path = root / "reports/optimisation/pre_optimisation_checksums.json"
    if pre_optimisation_path.exists():
        mapping = json.loads(pre_optimisation_path.read_text(encoding="utf-8"))
        april_entries = {
            path: digest for path, digest in mapping.items()
            if path.startswith("reports\\current\\")
        }
        phase5_entries = {
            path: digest for path, digest in mapping.items()
            if path.startswith("reports\\multi_report\\")
            or path == "data\\evaluation\\temporal_dataset_manifest.json"
        }
        for name, subset in (
            ("frozen_april_baseline", april_entries),
            ("frozen_phase5_temporal_baseline", phase5_entries),
        ):
            result = verify_checksum_entries(root, _mapping_to_entries(subset))
            checks.append({"name": name, **{k: v for k, v in result.items() if k != "rows"}})
    else:
        checks.append({"name": "pre_optimisation_checksums", "failure_count": 1, "failures": [{"path": str(pre_optimisation_path), "reason": "missing"}]})

    pre_phase6b_path = root / "reports/structural_optimisation/pre_phase6b_checksums.json"
    if pre_phase6b_path.exists():
        payload = json.loads(pre_phase6b_path.read_text(encoding="utf-8"))
        result = verify_checksum_entries(root, payload["entries"])
        checks.append({"name": "stage_a_and_phase6b_input_artifacts", **{k: v for k, v in result.items() if k != "rows"}})
    else:
        checks.append({"name": "pre_phase6b_checksums", "failure_count": 1, "failures": [{"path": str(pre_phase6b_path), "reason": "missing"}]})

    stage_a_selected = root / "reports/optimisation/stage_a_selected.json"
    stage_a_checksum = root / "reports/optimisation/stage_a_selected_checksum.json"
    if stage_a_selected.exists() and stage_a_checksum.exists():
        selected = json.loads(stage_a_selected.read_text(encoding="utf-8"))
        expected = json.loads(stage_a_checksum.read_text(encoding="utf-8"))["sha256"]
        actual = selected.get("selected_checksum")
        checks.append({
            "name": "stage_a_selected_checksum",
            "entry_count": 1,
            "match_count": int(actual == expected),
            "failure_count": 0 if actual == expected else 1,
            "failures": [] if actual == expected else [{"path": rel_path(stage_a_selected, root), "expected_sha256": expected, "actual_sha256": actual}],
        })
    else:
        checks.append({"name": "stage_a_selected_checksum", "failure_count": 1, "failures": [{"reason": "missing_stage_a_selected_or_checksum"}]})

    phase6b_selected = root / "reports/structural_optimisation/final_retrieval_selected.json"
    phase6b_checksum = root / "reports/structural_optimisation/final_retrieval_selected_checksum.json"
    if phase6b_selected.exists() and phase6b_checksum.exists():
        selected = json.loads(phase6b_selected.read_text(encoding="utf-8"))
        expected = json.loads(phase6b_checksum.read_text(encoding="utf-8"))["sha256"]
        actual = selected.get("selected_checksum")
        checks.append({
            "name": "phase6b_selected_retrieval_checksum",
            "entry_count": 1,
            "match_count": int(actual == expected),
            "failure_count": 0 if actual == expected else 1,
            "failures": [] if actual == expected else [{"path": rel_path(phase6b_selected, root), "expected_sha256": expected, "actual_sha256": actual}],
        })
    else:
        checks.append({"name": "phase6b_selected_retrieval_checksum", "failure_count": 1, "failures": [{"reason": "missing_phase6b_selected_or_checksum"}]})

    heldout = validate_heldout_dataset_manifest(root)
    heldout_ok = heldout["matches"] and heldout["case_count_matches"] and heldout["verified_count_matches"]
    checks.append({
        "name": "heldout_dataset_checksum",
        "entry_count": 1,
        "match_count": int(heldout_ok),
        "failure_count": 0 if heldout_ok else 1,
        "failures": [] if heldout_ok else [heldout],
        "details": heldout,
    })

    failure_count = sum(check.get("failure_count", 0) for check in checks)
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if failure_count == 0 else "failed",
        "failure_count": failure_count,
        "checks": checks,
    }


def write_pre_final_checksums(root: Path = Path("."), out_dir: Path = Path("reports/final_evaluation")) -> dict[str, Any]:
    out = root / out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_checksum_manifest(root)
    verification = validate_reference_checksums(root)
    payload = {**manifest, "verification": verification}
    write_json(out / "pre_final_eval_checksums.json", payload)
    lines = [
        "# Pre-final Evaluation Checksums",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Files captured: {payload['entry_count']}",
        f"Verification status: {verification['status']}",
        "",
        "## Frozen artifact checks",
        "",
    ]
    for check in verification["checks"]:
        lines.append(f"- {check['name']}: {check.get('match_count', 0)}/{check.get('entry_count', 0)} matched; failures={check.get('failure_count', 0)}")
    lines += ["", "## Captured files", ""]
    lines.extend(f"- `{entry['path']}`: `{entry['sha256']}`" for entry in payload["entries"])
    (out / "pre_final_eval_checksums.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def validate_final_retrieval_config(
    root: Path = Path("."),
    config_path: Path = Path("configs/final_retrieval_selected.yaml"),
) -> dict[str, Any]:
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    checks = []
    for dotted, expected in REQUIRED_FINAL_CONFIG.items():
        actual = config.get(dotted) if "." not in dotted else nested_get(config, dotted)
        checks.append({
            "field": dotted,
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        })

    selected_path = root / "reports/structural_optimisation/final_retrieval_selected.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    checksum = stable_json_hash(config)
    selected_checksum = selected.get("selected", {}).get("configuration_checksum")
    checks.append({
        "field": "configuration_checksum",
        "expected": selected_checksum,
        "actual": checksum,
        "matches": bool(selected_checksum) and checksum == selected_checksum,
    })
    checks.append({
        "field": "evaluation.heldout_loaded",
        "expected": False,
        "actual": nested_get(config, "evaluation.heldout_loaded"),
        "matches": nested_get(config, "evaluation.heldout_loaded") is False,
    })
    status = "passed" if all(check["matches"] for check in checks) else "failed"
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": status,
        "config_path": str(config_path).replace("/", "\\"),
        "config_sha256": file_sha(root / config_path),
        "configuration_checksum": checksum,
        "selected_experiment_id": config.get("id"),
        "checks": checks,
    }


def write_final_config_validation(root: Path = Path("."), out_dir: Path = Path("reports/final_evaluation")) -> dict[str, Any]:
    payload = validate_final_retrieval_config(root)
    out = root / out_dir
    write_json(out / "final_retrieval_config_validation.json", payload)
    lines = [
        "# Final Retrieval Configuration Validation",
        "",
        f"Status: {payload['status']}",
        f"Config: `{payload['config_path']}`",
        f"Selected experiment: `{payload['selected_experiment_id']}`",
        f"File SHA-256: `{payload['config_sha256']}`",
        f"Configuration checksum: `{payload['configuration_checksum']}`",
        "",
        "| Field | Expected | Actual | Match |",
        "|---|---:|---:|---:|",
    ]
    for check in payload["checks"]:
        lines.append(f"| `{check['field']}` | `{check['expected']}` | `{check['actual']}` | {check['matches']} |")
    (out / "final_retrieval_config_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def ensure_one_time_guard(out_dir: Path, *, confirm: bool, archive_failed: bool = False) -> None:
    if not confirm:
        raise RuntimeError("Refusing held-out retrieval without explicit one-time confirmation.")
    raw = out_dir / "heldout_retrieval_raw_results.json"
    summary = out_dir / "heldout_retrieval_summary.json"
    manifest = out_dir / "heldout_retrieval_run_manifest.json"
    if raw.exists() or summary.exists():
        if archive_failed:
            return
        raise RuntimeError("Held-out retrieval output already exists; refusing to rerun without archived technical failure.")
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            raise RuntimeError("Held-out retrieval is already marked completed; refusing to rerun.")


def assert_config_not_mutated(config_path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = file_sha(config_path)
    return {
        "path": str(config_path).replace("/", "\\"),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "matches": actual == expected_sha256,
    }


def _candidate_objects(ids: list[Any], pages: list[Any], scores: list[Any] | None = None, ranks: list[Any] | None = None) -> list[dict[str, Any]]:
    output = []
    for index, chunk_id in enumerate(ids or []):
        item = {
            "rank": ranks[index] if ranks and index < len(ranks) else index + 1,
            "chunk_id": chunk_id,
            "page": pages[index] if pages and index < len(pages) else None,
        }
        if scores is not None:
            item["score"] = scores[index] if index < len(scores) else None
        output.append(item)
    return output


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def canonicalise_retrieval_row(row: dict[str, Any], case: dict[str, Any] | None = None, chunk_lookup: dict[str, Any] | None = None) -> dict[str, Any]:
    chunk_lookup = chunk_lookup or {}
    case = case or {}
    per_report = row.get("per_report", {})
    required = list(row.get("required_report_ids") or case.get("required_report_ids") or [])
    selected_chunks_by_report: dict[str, list[dict[str, Any]]] = {}
    all_selected_chunks: list[dict[str, Any]] = []
    selected_pages: dict[str, list[int]] = {}
    accepted_pages: dict[str, list[int]] = {}
    expected_evidence: dict[str, list[str]] = {}
    loss_stage_by_report: dict[str, str] = {}
    dense, bm25, union, rrf, rerank_in, rerank_out = {}, {}, {}, {}, {}, {}

    report_ids_for_trace = sorted(set(per_report) | set(required))
    for report_id in report_ids_for_trace:
        trace = dict(per_report.get(report_id, {}))
        ground_truth = (case.get("ground_truth") or {}).get(report_id, {})
        trace.setdefault("accepted_pages", list(ground_truth.get("accepted_pages", [])))
        trace.setdefault("expected_evidence", list(ground_truth.get("expected_evidence", [])))
        for field in (
            "dense_candidate_ids",
            "dense_candidate_pages",
            "dense_candidate_scores",
            "bm25_candidate_ids",
            "bm25_candidate_pages",
            "bm25_candidate_scores",
            "candidate_union_ids",
            "candidate_union_pages",
            "rrf_candidate_ids",
            "rrf_candidate_pages",
            "rrf_scores",
            "reranker_input_ids",
            "reranker_input_pages",
            "reranker_input_ranks",
            "reranker_output_ids",
            "reranker_output_pages",
            "reranker_scores",
            "selected_chunk_ids_after_dedup",
            "selected_pages",
        ):
            trace.setdefault(field, [])
        trace.setdefault(
            "loss_stage",
            "unsupported_period" if row.get("query_type") == "unsupported_period" else recompute_loss_stage(trace),
        )
        row.setdefault("per_report", {})[report_id] = trace
        dense[report_id] = _candidate_objects(trace.get("dense_candidate_ids", []), trace.get("dense_candidate_pages", []), trace.get("dense_candidate_scores", []))
        bm25[report_id] = _candidate_objects(trace.get("bm25_candidate_ids", []), trace.get("bm25_candidate_pages", []), trace.get("bm25_candidate_scores", []))
        union[report_id] = _candidate_objects(trace.get("candidate_union_ids", []), trace.get("candidate_union_pages", []))
        rrf[report_id] = _candidate_objects(trace.get("rrf_candidate_ids", []), trace.get("rrf_candidate_pages", []), trace.get("rrf_scores", []))
        rerank_in[report_id] = _candidate_objects(trace.get("reranker_input_ids", []), trace.get("reranker_input_pages", []), ranks=trace.get("reranker_input_ranks", []))
        rerank_out[report_id] = _candidate_objects(trace.get("reranker_output_ids", []), trace.get("reranker_output_pages", []), trace.get("reranker_scores", []))
        selected_pages[report_id] = list(trace.get("selected_pages", []))
        accepted_pages[report_id] = list(trace.get("accepted_pages", []))
        expected_evidence[report_id] = list(trace.get("expected_evidence", []))
        loss_stage_by_report[report_id] = trace.get("loss_stage")
        chunks = []
        for chunk_id in trace.get("selected_chunk_ids_after_dedup", []):
            doc = chunk_lookup.get(chunk_id)
            item = {
                "chunk_id": chunk_id,
                "report_id": report_id,
                "page": None,
                "text": None,
            }
            if doc is not None:
                item.update({
                    "report_id": doc.metadata.get("report_id", report_id),
                    "report_period": doc.metadata.get("report_period"),
                    "page": doc.metadata.get("page"),
                    "text": doc.page_content,
                })
            chunks.append(item)
            all_selected_chunks.append(item)
        selected_chunks_by_report[report_id] = chunks

    latency_by_stage = {field: row.get(field) for field in LATENCY_FIELDS}
    raw = dict(row)
    raw.update({
        "category": case.get("category", row.get("category")),
        "source_information_type": case.get("source_information_type", row.get("source_information_type", [])),
        "retrieved_dense_candidates_by_report": dense,
        "retrieved_bm25_candidates_by_report": bm25,
        "candidate_union_by_report": union,
        "rrf_candidates_by_report": rrf,
        "reranker_input_by_report": rerank_in,
        "reranker_output_by_report": rerank_out,
        "selected_chunks_by_report": selected_chunks_by_report,
        "all_selected_chunks": all_selected_chunks,
        "selected_pages": selected_pages,
        "accepted_pages": accepted_pages,
        "expected_evidence": expected_evidence,
        "loss_stage": loss_stage_by_report,
        "loss_stage_by_report": loss_stage_by_report,
        "macro_report_mrr": row.get("macro_report_mrr", row.get("macro_mrr")),
        "single_report_contamination": bool(row.get("contamination", 0)) if row.get("query_type") == "single_report" else None,
        "latency_by_stage": latency_by_stage,
        "total_latency": row.get("total_retrieval_latency_ms"),
        "warnings": row.get("warnings", []),
        "required_report_ids": required,
    })
    return raw


def _row_metric_values(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("required_report_ids") or row.get("query_type") == "unsupported_period":
        return {
            "report_coverage": row.get("report_coverage", 0.0),
            "all_reports_hit": row.get("all_reports_hit"),
            "evidence_recall": row.get("evidence_recall"),
            "complete_evidence_recall": row.get("complete_evidence_recall"),
            "macro_report_mrr": row.get("macro_report_mrr", row.get("macro_mrr", 0.0)),
            "single_report_contamination": row.get("single_report_contamination"),
        }
    required = list(row["required_report_ids"])
    selected = row.get("all_selected_chunks") or [
        chunk for chunks in (row.get("selected_chunks_by_report") or {}).values() for chunk in chunks
    ]
    selected_by_report = {report_id: [chunk for chunk in selected if chunk.get("report_id") == report_id] for report_id in required}
    represented = {chunk.get("report_id") for chunk in selected}
    per_report_hit = {}
    reciprocal = {}
    evidence_values: list[bool] = []
    for report_id in required:
        accepted = set((row.get("accepted_pages") or {}).get(report_id, []))
        pages = [chunk.get("page") for chunk in selected_by_report[report_id]]
        ranks = [rank for rank, page in enumerate(pages, 1) if page in accepted]
        per_report_hit[report_id] = bool(ranks) if accepted else None
        reciprocal[report_id] = 1.0 / min(ranks) if ranks else 0.0
        combined = _normalise_text(" ".join(chunk.get("text") or "" for chunk in selected_by_report[report_id]))
        for evidence in (row.get("expected_evidence") or {}).get(report_id, []):
            evidence_values.append(_normalise_text(evidence) in combined)
    scored_hits = [value for value in per_report_hit.values() if value is not None]
    contamination = any(chunk.get("report_id") not in set(required) for chunk in selected)
    return {
        "report_coverage": sum(report_id in represented for report_id in required) / len(required) if required else 0.0,
        "all_reports_hit": all(scored_hits) if scored_hits else None,
        "evidence_recall": sum(evidence_values) / len(evidence_values) if evidence_values else None,
        "complete_evidence_recall": all(evidence_values) if evidence_values else None,
        "macro_report_mrr": sum(reciprocal.values()) / len(required) if required else 0.0,
        "single_report_contamination": contamination if row.get("query_type") == "single_report" else None,
    }


def recompute_retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("required_report_ids") and row.get("query_type") != "unsupported_period"]
    values = [_row_metric_values(row) for row in valid]
    scored = [value for value in values if value["all_reports_hit"] is not None]
    evidence = [value for value in values if value["evidence_recall"] is not None]
    cer = [value for value in values if value["complete_evidence_recall"] is not None]
    contam = [value for value in values if value["single_report_contamination"] is not None]
    latencies = [row.get("total_retrieval_latency_ms", row.get("total_latency")) for row in valid]
    latencies = [float(value) for value in latencies if isinstance(value, (int, float))]
    token_counts = [row.get("estimated_token_count") for row in valid if isinstance(row.get("estimated_token_count"), (int, float))]
    chunk_counts = [row.get("selected_chunk_count") for row in valid if isinstance(row.get("selected_chunk_count"), (int, float))]

    def mean(items: list[Any]) -> float | None:
        return sum(items) / len(items) if items else None

    p95 = None
    if latencies:
        ordered_latencies = sorted(latencies)
        p95 = ordered_latencies[min(len(ordered_latencies) - 1, int((len(ordered_latencies) - 1) * 0.95))]
    return {
        "case_count": len(rows),
        "scored_case_count": len(valid),
        "report_coverage": mean([value["report_coverage"] for value in values]),
        "all_reports_hit": mean([float(value["all_reports_hit"]) for value in scored]),
        "evidence_recall": mean([value["evidence_recall"] for value in evidence]),
        "complete_evidence_recall": mean([float(value["complete_evidence_recall"]) for value in cer]),
        "macro_report_mrr": mean([value["macro_report_mrr"] for value in values]),
        "single_report_contamination": mean([float(value["single_report_contamination"]) for value in contam]),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": p95,
        "mean_estimated_tokens": mean(token_counts),
        "mean_selected_chunks": mean(chunk_counts),
    }


def confidence_intervals(rows: list[dict[str, Any]], *, resamples: int = 2000, seed: int = 42) -> dict[str, Any]:
    valid = [row for row in rows if row.get("required_report_ids") and row.get("query_type") != "unsupported_period"]
    values = [_row_metric_values(row) for row in valid]

    def binary_interval(name: str) -> dict[str, Any]:
        items = [value[name] for value in values if value.get(name) is not None]
        successes = sum(bool(item) for item in items)
        low, high = wilson_interval(successes, len(items))
        return {"method": "wilson", "n": len(items), "ci_95_low": low, "ci_95_high": high}

    def continuous_interval(name: str, source: list[float] | None = None) -> dict[str, Any]:
        items = source if source is not None else [value[name] for value in values if value.get(name) is not None]
        low, high = bootstrap_mean_interval(items, resamples=resamples, seed=seed)
        return {"method": "bootstrap_mean", "n": len(items), "ci_95_low": low, "ci_95_high": high, "resamples": resamples, "seed": seed}

    latencies = [float(row.get("total_retrieval_latency_ms", row.get("total_latency"))) for row in valid if isinstance(row.get("total_retrieval_latency_ms", row.get("total_latency")), (int, float))]
    tokens = [float(row["estimated_token_count"]) for row in valid if isinstance(row.get("estimated_token_count"), (int, float))]
    chunks = [float(row["selected_chunk_count"]) for row in valid if isinstance(row.get("selected_chunk_count"), (int, float))]
    return {
        "complete_evidence_recall": binary_interval("complete_evidence_recall"),
        "all_reports_hit": binary_interval("all_reports_hit"),
        "single_report_contamination": binary_interval("single_report_contamination"),
        "report_coverage": continuous_interval("report_coverage"),
        "evidence_recall": continuous_interval("evidence_recall"),
        "macro_report_mrr": continuous_interval("macro_report_mrr"),
        "mean_latency_ms": continuous_interval("mean_latency_ms", latencies),
        "mean_estimated_tokens": continuous_interval("mean_estimated_tokens", tokens),
        "mean_selected_chunks": continuous_interval("mean_selected_chunks", chunks),
    }


def category_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[("query_type", row.get("query_type", "unknown"))].append(row)
        grouped[("category", row.get("category") or "unknown")].append(row)
        source_types = row.get("source_information_type") or ["unknown"]
        for source_type in source_types:
            grouped[("source_information_type", source_type)].append(row)
    output = []
    for (category_type, category), values in sorted(grouped.items()):
        metrics = recompute_retrieval_metrics(values)
        output.append({"category_type": category_type, "category": category, **metrics})
    return output


def build_retrieval_summary(
    rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    *,
    split: str,
    config_checksum: str,
    config_file_sha256: str,
    dataset_sha256: str,
    index_fingerprint: str | None,
    started_at_utc: str,
    finished_at_utc: str,
) -> dict[str, Any]:
    metrics = recompute_retrieval_metrics(rows)
    return {
        "schema_version": 1,
        "split": split,
        "created_at_utc": finished_at_utc,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "case_count": metrics["case_count"],
        "scored_case_count": metrics["scored_case_count"],
        "report_level_row_count": len(report_rows),
        "configuration_checksum": config_checksum,
        "config_file_sha256": config_file_sha256,
        "dataset_sha256": dataset_sha256,
        "index_fingerprint": index_fingerprint,
        **{key: value for key, value in metrics.items() if key not in {"case_count", "scored_case_count"}},
        "confidence_intervals": confidence_intervals(rows),
        "category_metrics": category_metrics(rows),
        "loss_stage_counts": dict(Counter(item.get("loss_stage") for item in report_rows)),
    }


def report_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for report_id in row.get("required_report_ids", []):
            selected = (row.get("selected_chunks_by_report") or {}).get(report_id, [])
            accepted = (row.get("accepted_pages") or {}).get(report_id, [])
            expected = (row.get("expected_evidence") or {}).get(report_id, [])
            selected_pages = [chunk.get("page") for chunk in selected]
            evidence_hits = [
                _normalise_text(text) in _normalise_text(" ".join(chunk.get("text") or "" for chunk in selected))
                for text in expected
            ]
            rank = next((index + 1 for index, page in enumerate(selected_pages) if page in set(accepted)), None)
            output.append({
                "question_id": row["question_id"],
                "query_type": row.get("query_type"),
                "category": row.get("category"),
                "report_id": report_id,
                "selected_pages": selected_pages,
                "accepted_pages": accepted,
                "expected_evidence_count": len(expected),
                "evidence_recall": sum(evidence_hits) / len(evidence_hits) if evidence_hits else None,
                "final_found": bool(rank) if accepted else None,
                "mrr": 1.0 / rank if rank else 0.0,
                "loss_stage": (row.get("loss_stage_by_report") or {}).get(report_id),
            })
    return output


def validate_heldout_raw_schema(rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    required_top = {
        "question_id",
        "query_type",
        "required_report_ids",
        "original_query",
        "normalised_query",
        "expanded_queries",
        "facet_queries",
        "retrieved_dense_candidates_by_report",
        "retrieved_bm25_candidates_by_report",
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
        "total_latency",
        "selected_character_count",
        "estimated_token_count",
        "unique_page_count",
        "repeated_text_ratio",
        "warnings",
    }
    for row in rows:
        qid = row.get("question_id", "unknown")
        for field in required_top:
            if field not in row:
                issues.append(f"{qid}:missing_top_field:{field}")
        issues.extend(f"{qid}:{issue}" for issue in validate_latency_schema(row))
        for field in CONTEXT_STAT_FIELDS:
            if not isinstance(row.get(field), (int, float)):
                issues.append(f"{qid}:missing_or_non_numeric_context_stat:{field}")
        for report_id, trace in (row.get("per_report") or {}).items():
            for field in REQUIRED_TRACE_FIELDS:
                if field not in trace:
                    issues.append(f"{qid}:{report_id}:missing_trace_field:{field}")
    return issues


def validate_heldout_integrity(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    pre_checksums: dict[str, Any],
    config_path: Path,
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    issues = validate_heldout_raw_schema(rows)
    pre_config_entry = next(
        (entry for entry in pre_checksums.get("entries", []) if entry["path"] == str(config_path).replace("/", "\\")),
        None,
    )
    if not pre_config_entry:
        issues.append("final_config_missing_from_pre_final_manifest")
    else:
        mutation = assert_config_not_mutated(config_path, pre_config_entry["sha256"])
        if not mutation["matches"]:
            issues.append("final_config_checksum_changed_after_heldout_run")
    if not dataset_validation["matches"]:
        issues.append("heldout_dataset_checksum_mismatch")

    recomputed = recompute_retrieval_metrics(rows)
    for key in (
        "report_coverage",
        "all_reports_hit",
        "evidence_recall",
        "complete_evidence_recall",
        "macro_report_mrr",
        "single_report_contamination",
    ):
        actual = summary.get(key)
        expected = recomputed.get(key)
        if actual is None and expected is None:
            continue
        if actual is None or expected is None or abs(float(actual) - float(expected)) > 1e-12:
            issues.append(f"summary_metric_mismatch:{key}:expected={expected}:actual={actual}")

    report_rows = report_level_rows(rows)
    recomputed_loss = dict(Counter(item["loss_stage"] for item in report_rows))
    if summary.get("loss_stage_counts") != recomputed_loss:
        issues.append("loss_stage_counts_mismatch")
    if category_metrics(rows) != summary.get("category_metrics"):
        issues.append("category_metrics_mismatch")
    if confidence_intervals(rows) != summary.get("confidence_intervals"):
        issues.append("confidence_intervals_mismatch")
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": sorted(set(issues)),
        "recomputed_metrics": recomputed,
        "recomputed_loss_stage_counts": recomputed_loss,
        "dataset_validation": dataset_validation,
    }


def build_dev_vs_heldout_comparison(
    dev_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dev_metrics = recompute_retrieval_metrics(dev_rows)
    heldout_metrics = recompute_retrieval_metrics(heldout_rows)
    dev_ci = confidence_intervals(dev_rows)
    heldout_ci = confidence_intervals(heldout_rows)
    rows = []
    for metric in (
        "complete_evidence_recall",
        "all_reports_hit",
        "evidence_recall",
        "macro_report_mrr",
        "report_coverage",
        "single_report_contamination",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "mean_estimated_tokens",
        "mean_selected_chunks",
    ):
        dev = dev_metrics.get(metric)
        heldout = heldout_metrics.get(metric)
        diff = None if dev is None or heldout is None else heldout - dev
        relative = None if diff is None or not dev else diff / dev
        rows.append({
            "metric": metric,
            "development": dev,
            "heldout": heldout,
            "absolute_difference_heldout_minus_dev": diff,
            "relative_difference": relative,
            "development_ci": dev_ci.get(metric),
            "heldout_ci": heldout_ci.get(metric),
        })
    return rows


def write_dev_vs_heldout_comparison(
    dev_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows = build_dev_vs_heldout_comparison(dev_rows, heldout_rows)
    category_dev = {(row["category_type"], row["category"]): row for row in category_metrics(dev_rows)}
    category_heldout = {(row["category_type"], row["category"]): row for row in category_metrics(heldout_rows)}
    category_rows = []
    for key in sorted(set(category_dev) | set(category_heldout)):
        left, right = category_dev.get(key, {}), category_heldout.get(key, {})
        category_rows.append({
            "category_type": key[0],
            "category": key[1],
            "development_case_count": left.get("case_count"),
            "heldout_case_count": right.get("case_count"),
            "development_complete_evidence_recall": left.get("complete_evidence_recall"),
            "heldout_complete_evidence_recall": right.get("complete_evidence_recall"),
            "development_all_reports_hit": left.get("all_reports_hit"),
            "heldout_all_reports_hit": right.get("all_reports_hit"),
            "development_evidence_recall": left.get("evidence_recall"),
            "heldout_evidence_recall": right.get("evidence_recall"),
            "development_macro_report_mrr": left.get("macro_report_mrr"),
            "heldout_macro_report_mrr": right.get("macro_report_mrr"),
        })
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "metric_comparison": rows,
        "category_comparison": category_rows,
        "note": "Held-out results are reported for evaluation only and were not used for retrieval tuning.",
    }
    write_json(out_dir / "dev_vs_heldout_retrieval_comparison.json", payload)
    write_csv(out_dir / "dev_vs_heldout_retrieval_comparison.csv", rows + category_rows)
    lines = [
        "# Development vs Held-out Retrieval Comparison",
        "",
        "Held-out results are evaluation-only; no retrieval tuning is performed from this comparison.",
        "",
        "| Metric | Development | Held-out | Abs diff | Rel diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['development']} | {row['heldout']} | {row['absolute_difference_heldout_minus_dev']} | {row['relative_difference']} |")
    lines += ["", "## Category-level differences", ""]
    for row in category_rows:
        lines.append(f"- {row['category_type']}={row['category']}: dev CER={row['development_complete_evidence_recall']}, held-out CER={row['heldout_complete_evidence_recall']}")
    (out_dir / "dev_vs_heldout_retrieval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def groq_key_available(root: Path = Path(".")) -> bool:
    if os.getenv("GROQ_API_KEY"):
        return True
    env_path = root / ".env"
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GROQ_API_KEY") and "=" in line:
            return bool(line.split("=", 1)[1].strip().strip('"').strip("'"))
    return False


def decide_generation_readiness(
    *,
    heldout_completed: bool,
    heldout_integrity: dict[str, Any],
    config_not_mutated: bool,
    groq_available: bool,
    frozen_generation_supported: bool,
    retrieval_tuning_after_heldout: bool,
) -> dict[str, Any]:
    checks = {
        "heldout_retrieval_completed": heldout_completed,
        "heldout_retrieval_integrity_passed": heldout_integrity.get("status") == "passed",
        "final_retrieval_config_unchanged": config_not_mutated,
        "groq_api_key_available": groq_available,
        "generation_can_use_frozen_retrieval_outputs": frozen_generation_supported,
        "no_retrieval_tuning_after_heldout": not retrieval_tuning_after_heldout,
    }
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "ready" if all(checks.values()) else "not_ready",
        "checks": checks,
        "generation_evaluation_run": False,
    }


def metric_result(score: float | None, *, success: bool, attempts: int = 1, error_type: str | None = None, error_message: str | None = None) -> dict[str, Any]:
    return {
        "success": success,
        "score": None if not success else score,
        "attempts": attempts,
        "error_type": error_type,
        "error_message": error_message,
    }


def summarise_metric_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({
        name
        for row in rows
        for name in (row.get("evaluation_metric_results") or row.get("metrics") or {})
    })
    output: dict[str, Any] = {}
    for name in names:
        values = [
            (row.get("evaluation_metric_results") or row.get("metrics") or {}).get(name)
            for row in rows
            if name in (row.get("evaluation_metric_results") or row.get("metrics") or {})
        ]
        scores = [float(value["score"]) for value in values if value and value.get("success") and value.get("score") is not None]
        output[name] = {
            "successful_evaluations": len(scores),
            "failed_evaluations": len(values) - len(scores),
            "total_evaluations": len(values),
            "coverage_percentage": 100 * len(scores) / len(values) if values else 0.0,
            "mean_over_successful_cases": statistics.mean(scores) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "standard_deviation": statistics.stdev(scores) if len(scores) > 1 else None,
        }
    return output


def validate_citation_references(row: dict[str, Any]) -> bool:
    supplied = set(row.get("selected_chunk_ids") or [])
    for citation in row.get("citations") or []:
        chunk_id = citation.get("chunk_id") if isinstance(citation, dict) else getattr(citation, "chunk_id", None)
        if chunk_id not in supplied:
            return False
    return True


def deterministic_generation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    supplied_reports = set(row.get("required_report_ids") or [])
    citation_reports = {
        citation.get("report_id")
        for citation in row.get("citations") or []
        if isinstance(citation, dict) and citation.get("report_id")
    }
    answer = (row.get("generated_answer") or "").lower()
    abstained = any(token in answer for token in ("could not find", "not available", "not in the registered corpus", "not in the supplied"))
    unsupported = row.get("query_type") == "unsupported_period"
    citation_correct = validate_citation_references(row)
    citation_complete = supplied_reports <= citation_reports if supplied_reports else not citation_reports
    abstention_correct = abstained if unsupported else not abstained
    return {
        "citation_correctness": metric_result(1.0 if citation_correct else 0.0, success=True),
        "citation_completeness": metric_result(1.0 if citation_complete else 0.0, success=True),
        "abstention_correctness": metric_result(1.0 if abstention_correct else 0.0, success=True),
    }


def temporal_attribution_failure_wrong_report(row: dict[str, Any]) -> bool:
    required = set(row.get("required_report_ids") or [])
    for citation in row.get("citations") or []:
        if isinstance(citation, dict) and citation.get("report_id") and citation["report_id"] not in required:
            return True
    return False


def validate_generation_integrity(rows: list[dict[str, Any]], *, expected_split: str | None = None) -> dict[str, Any]:
    issues: list[str] = []
    for row in rows:
        qid = row.get("question_id", "unknown")
        for field in GENERATION_REQUIRED_FIELDS:
            if field not in row:
                issues.append(f"{qid}:missing_field:{field}")
        if expected_split and row.get("split") != expected_split:
            issues.append(f"{qid}:split_mismatch:{row.get('split')}")
        if not validate_citation_references(row):
            issues.append(f"{qid}:citation_not_in_supplied_context")
        for meta_field in ("prompt_version", "model_name", "temperature"):
            if row.get(meta_field) in (None, ""):
                issues.append(f"{qid}:missing_generation_metadata:{meta_field}")
        for metric_name, result in (row.get("evaluation_metric_results") or {}).items():
            if result.get("success") and not isinstance(result.get("score"), (int, float)):
                issues.append(f"{qid}:{metric_name}:successful_metric_non_numeric_score")
            if not result.get("success") and result.get("score") is not None:
                issues.append(f"{qid}:{metric_name}:failed_metric_score_not_null")
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": sorted(set(issues)),
    }


def validate_final_status(value: str) -> bool:
    return value in FINAL_STATUS_VALUES


def contains_groq_secret(value: Any) -> bool:
    text = json.dumps(value, default=str)
    return "GROQ_API_KEY" in text or "gsk_" in text
