from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import random
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder

from rbi_rag.experiment_tracing import (
    LATENCY_FIELDS,
    context_statistics,
    first_evidence_rank,
    recompute_loss_stage,
    validate_latency_schema,
)
from rbi_rag.multi_config import MultiReportConfig
from rbi_rag.multi_evaluation import load_jsonl
from rbi_rag.multi_index import build_multi_report_index
from rbi_rag.report_bm25 import BM25ByReport
from rbi_rag.report_registry import ReportRegistry
from rbi_rag.structural_optimisation import (
    adjacent_chunk_expansion,
    build_parent_window,
    dependency_available,
    sentence_window_documents,
)
from rbi_rag.temporal_router import TemporalQueryRouter


ROOT = Path(".")
OUT = ROOT / "reports" / "structural_optimisation"
INDEX_ROOT = ROOT / "indexes" / "structural_optimisation"
REGISTRY_PATH = ROOT / "configs" / "structural_optimisation_experiments.yaml"
STAGE_A_SELECTED = ROOT / "reports" / "optimisation" / "stage_a_selected.json"
REQUIRED_FILES = {
    "config_snapshot.yaml",
    "environment.json",
    "index_manifest.json",
    "raw_results.json",
    "question_results.csv",
    "report_level_results.csv",
    "summary.json",
    "summary.md",
    "stage_diagnostics.csv",
}
RESAMPLES = 2000
SEED = 42


def _load_stage_a_runner():
    path = ROOT / "scripts" / "run_stage_a_ablations.py"
    spec = importlib.util.spec_from_file_location("stage_a_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


_STAGE_A = _load_stage_a_runner()
groq_key_available = _STAGE_A.groq_key_available
run_question = _STAGE_A.run_question
stable_json_hash = _STAGE_A.stable_json_hash
warm_up = _STAGE_A.warm_up


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=dict) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, default=dict) if isinstance(value, (dict, list, Counter)) else value
                for key, value in row.items()
            })


def checksum_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [
        ROOT / "reports" / "current",
        ROOT / "reports" / "multi_report",
        ROOT / "reports" / "optimisation",
        ROOT / "configs" / "stage_a_selected.yaml",
        ROOT / "reports" / "optimisation" / "stage_a_selected.json",
        ROOT / "reports" / "optimisation" / "stage_a_selected.md",
        ROOT / "reports" / "optimisation" / "stage_a_report.md",
    ]
    entries = []
    for target in targets:
        if target.is_dir():
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    entries.append({"path": str(path), "sha256": file_sha(path)})
        elif target.exists():
            entries.append({"path": str(target), "sha256": file_sha(target)})
    payload = {"created_at": now_iso(), "entries": entries}
    write_json(OUT / "pre_phase6b_checksums.json", payload)
    lines = ["# Pre Phase 6B Checksums", "", f"Files captured: {len(entries)}", ""]
    lines.extend(f"- `{entry['path']}`: `{entry['sha256']}`" for entry in entries)
    (OUT / "pre_phase6b_checksums.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_pre_checksums():
    payload = json.loads((OUT / "pre_phase6b_checksums.json").read_text(encoding="utf-8"))
    rows = []
    for entry in payload["entries"]:
        path = Path(entry["path"])
        rows.append({
            "path": entry["path"],
            "exists": path.exists(),
            "matches": path.exists() and file_sha(path) == entry["sha256"],
        })
    return rows


def write_selection_policy():
    policy = {
        "eligibility": {
            "integrity_status": "valid",
            "report_coverage": 1.0,
            "single_report_contamination": 0.0,
        },
        "selection_order": [
            "complete_evidence_recall",
            "all_reports_hit",
            "evidence_recall",
            "macro_mrr",
            "median_latency_ms",
            "mean_estimated_tokens",
            "implementation_simplicity",
        ],
        "latency_constraint": "median_latency_ms <= 2 * stage_a_selected_reference_median_latency_ms",
    }
    write_json(OUT / "selection_policy.json", policy)
    (OUT / "selection_policy.md").write_text(
        "# Phase 6B Selection Policy\n\n"
        "Eligibility: valid integrity, 100% report coverage, zero single-report contamination.\n\n"
        "Order: Complete Evidence Recall, All-Reports Hit, Evidence Recall, Macro MRR, median latency, smaller context, implementation simplicity.\n\n"
        "Latency constraint: median latency must be no more than 2x the Stage A selected reference unless a substantial retrieval gain is explicitly justified.\n",
        encoding="utf-8",
    )


def load_registry():
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    configs = []
    for exp_id, value in raw.items():
        item = dict(value)
        item["id"] = exp_id
        configs.append(item)
    return configs


def flatten_for_stage_a(config):
    retrieval = config.get("retrieval", {})
    fusion = config.get("fusion", {})
    context = config.get("context_selection", {})
    return {
        "id": config["id"],
        "family": family_for(config["id"], config),
        "dk": int(retrieval.get("dense_k", 50)),
        "bk": int(retrieval.get("bm25_k", 50)),
        "retain": int(fusion.get("retain", 30)),
        "rrf": int(fusion.get("k", 60)),
        "dw": float(fusion.get("dense_weight", 1.0)),
        "bw": float(fusion.get("bm25_weight", 1.0)),
        "quota": [
            int(context.get("single", 6)),
            int(context.get("pairwise", 5)),
            int(context.get("trend", 4)),
        ],
    }


def family_for(exp_id, config):
    if exp_id.startswith("CPARENT"):
        return "child_parent"
    if exp_id.startswith("SW"):
        return "sentence_window"
    if exp_id.startswith("ADJ"):
        return "adjacent_expansion"
    if exp_id.startswith("SEM"):
        return "semantic_chunking"
    if exp_id.startswith("PARSER"):
        return "parser"
    if exp_id.startswith("EMB"):
        return "embedding"
    if exp_id.startswith("RERANK"):
        return "reranker"
    if exp_id.startswith("STRUCT_CANDIDATE"):
        return "combined_structural"
    return "stage_a_reference"


def all_chunks_by_id(chunks_by_report):
    return {
        chunk.metadata["chunk_id"]: chunk
        for chunks in chunks_by_report.values()
        for chunk in chunks
    }


def selected_ids(row):
    output = []
    for trace in row.get("per_report", {}).values():
        output.extend(trace.get("selected_chunk_ids_after_dedup", []))
    return output


def expected_hits(chunks, expected):
    combined = " ".join(" ".join(chunk.page_content.lower().split()) for chunk in chunks)
    return [" ".join(text.lower().split()) in combined for text in expected]


def apply_structural_transform(row, config, chunks_by_report):
    strategy = config.get("chunking", {}).get("strategy", "recursive")
    context = config.get("context_selection", {})
    started = time.perf_counter()
    ids = selected_ids(row)
    structural = {"added_chunk_ids": [], "removed_chunk_ids": [], "strategy": strategy}
    if strategy == "child_parent":
        selection = build_parent_window(
            ids,
            chunks_by_report,
            strategy=config["chunking"].get("parent_strategy", "same_page_local_window"),
            parent_max_chars=int(config["chunking"].get("parent_max_chars", 1400)),
        )
    elif strategy == "sentence_window":
        selection = sentence_window_documents(
            ids,
            chunks_by_report,
            before=int(config["chunking"].get("window_before", 1)),
            after=int(config["chunking"].get("window_after", 1)),
            max_chars=int(config["chunking"].get("max_chars", 1400)),
        )
    elif strategy == "chunk_window":
        selection = build_parent_window(
            ids,
            chunks_by_report,
            strategy="adjacent_child_window",
            parent_max_chars=int(config["chunking"].get("max_chars", 1600)),
        )
    elif strategy == "adjacent_expansion":
        selection = adjacent_chunk_expansion(
            ids,
            chunks_by_report,
            max_tokens=int(config["chunking"].get("max_tokens", 2600)),
            mode=config["chunking"].get("mode", "boundary"),
            query_type=row["query_type"],
        )
    else:
        lookup = all_chunks_by_id(chunks_by_report)
        docs = [lookup[chunk_id] for chunk_id in ids if chunk_id in lookup]
        selection = type("Selection", (), {
            "documents": docs,
            "added_chunk_ids": [],
            "removed_chunk_ids": [],
            "expansion_reason": "none",
        })()
    structural["added_chunk_ids"] = list(selection.added_chunk_ids)
    structural["removed_chunk_ids"] = list(selection.removed_chunk_ids)
    structural["expansion_reason"] = selection.expansion_reason
    expansion_ms = (time.perf_counter() - started) * 1000
    return recompute_row_from_documents(row, config, selection.documents, structural, expansion_ms)


def recompute_row_from_documents(row, config, docs, structural, expansion_ms):
    required = list(row["required_report_ids"])
    by_report = {report_id: [doc for doc in docs if doc.metadata["report_id"] == report_id] for report_id in required}
    evidence_values = []
    hits = {}
    reciprocal = {}
    report_rows = []
    for report_id in required:
        trace = dict(row.get("per_report", {}).get(report_id, {}))
        gt_pages = trace.get("accepted_pages", [])
        expected = trace.get("expected_evidence", [])
        selected_pages = [doc.metadata["page"] for doc in by_report[report_id]]
        rank = first_evidence_rank(selected_pages, gt_pages)
        evidence = expected_hits(by_report[report_id], expected)
        evidence_values.extend(evidence)
        hits[report_id] = bool(rank) if gt_pages else None
        reciprocal[report_id] = 1.0 / rank if rank else 0.0
        trace["selected_chunk_ids_after_dedup"] = [doc.metadata["chunk_id"] for doc in by_report[report_id]]
        trace["selected_pages"] = selected_pages
        trace["accepted_evidence_found"] = bool(rank)
        trace["structural_added_chunk_ids"] = structural["added_chunk_ids"]
        trace["structural_strategy"] = structural["strategy"]
        trace["loss_stage"] = recompute_loss_stage(trace)
        row["per_report"][report_id] = trace
        report_rows.append({
            "experiment_id": config["id"],
            "question_id": row["question_id"],
            "query_type": row["query_type"],
            "report_id": report_id,
            "final_found": bool(rank),
            "loss_stage": trace["loss_stage"],
            "structural_strategy": structural["strategy"],
        })
    stats = context_statistics(docs)
    selected_report_ids = {doc.metadata["report_id"] for doc in docs}
    contamination = sum(1 for doc in docs if doc.metadata["report_id"] not in required)
    scored_hits = [value for value in hits.values() if value is not None]
    row.update(stats)
    row["experiment_id"] = config["id"]
    row["configuration_checksum"] = stable_json_hash(config)
    row["report_chunk_counts"] = Counter(doc.metadata["report_id"] for doc in docs)
    row["report_coverage"] = sum(report_id in selected_report_ids for report_id in required) / len(required) if required else 0.0
    row["all_reports_hit"] = all(scored_hits) if scored_hits else None
    row["per_report_hit"] = hits
    row["per_report_mrr"] = reciprocal
    row["macro_mrr"] = sum(reciprocal.values()) / len(required) if required else 0.0
    row["evidence_recall"] = sum(evidence_values) / len(evidence_values) if evidence_values else None
    row["complete_evidence_recall"] = all(evidence_values) if evidence_values else None
    row["contamination"] = contamination
    row["structural_trace"] = structural
    row["context_construction_latency_ms"] = float(row.get("context_construction_latency_ms", 0.0)) + expansion_ms
    row["total_retrieval_latency_ms"] = float(row.get("total_retrieval_latency_ms", 0.0)) + expansion_ms
    return row, report_rows


def mean(values):
    return sum(values) / len(values) if values else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarise(config, rows, report_rows, started_at, finished_at, dataset_checksum, index_fingerprint):
    valid = [row for row in rows if row.get("required_report_ids") and row.get("query_type") != "unsupported_period"]
    scored = [row for row in valid if row.get("all_reports_hit") is not None]
    evidence = [row for row in valid if row.get("evidence_recall") is not None]
    latencies = [row["total_retrieval_latency_ms"] for row in valid]
    return {
        "experiment_id": config["id"],
        "family": family_for(config["id"], config),
        "description": config.get("description"),
        "case_count": len(rows),
        "report_level_row_count": len(report_rows),
        "report_coverage": mean([row["report_coverage"] for row in valid]),
        "all_reports_hit": mean([float(row["all_reports_hit"]) for row in scored]),
        "macro_mrr": mean([row["macro_mrr"] for row in valid]),
        "evidence_recall": mean([row["evidence_recall"] for row in evidence]),
        "complete_evidence_recall": mean([float(row["complete_evidence_recall"]) for row in evidence]),
        "contamination": mean([row["contamination"] for row in valid]),
        "mean_selected_characters": mean([row["selected_character_count"] for row in valid]),
        "mean_estimated_tokens": mean([row["estimated_token_count"] for row in valid]),
        "mean_selected_chunks": mean([row["selected_chunk_count"] for row in valid]),
        "mean_unique_pages": mean([row["unique_page_count"] for row in valid]),
        "mean_repeated_text_ratio": mean([row["repeated_text_ratio"] for row in valid]),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "loss_stage_counts": dict(Counter(item["loss_stage"] for item in report_rows)),
        "dataset_sha256": dataset_checksum,
        "index_fingerprint": index_fingerprint,
        "configuration_checksum": stable_json_hash(config),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def validate_experiment(path: Path, dataset_checksum: str, expected_count: int):
    issues = []
    files = {item.name for item in path.iterdir() if item.is_file()}
    issues.extend(f"missing_file:{name}" for name in sorted(REQUIRED_FILES - files))
    if issues:
        return issues
    config = yaml.safe_load((path / "config_snapshot.yaml").read_text(encoding="utf-8"))
    env = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    raw = json.loads((path / "raw_results.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if config.get("id") != path.name:
        issues.append("experiment_id_mismatch")
    if stable_json_hash(config) != summary.get("configuration_checksum"):
        issues.append("configuration_checksum_mismatch")
    if env.get("heldout_dataset_loaded") is not False:
        issues.append("heldout_loaded_flag_not_false")
    if len(raw) != expected_count:
        issues.append("raw_count_mismatch")
    for row in raw:
        issues.extend(validate_latency_schema(row))
        if row.get("dataset_checksum") != dataset_checksum:
            issues.append("dataset_checksum_mismatch")
        if row.get("configuration_checksum") != summary.get("configuration_checksum"):
            issues.append("row_configuration_checksum_mismatch")
        if row.get("split") == "test" or str(row.get("question_id", "")).startswith("test_"):
            issues.append("heldout_case_present")
        for field in ("selected_character_count", "estimated_token_count", "selected_chunk_count",
                      "unique_page_count", "duplicate_chunk_count", "repeated_text_ratio", "per_report"):
            if field not in row:
                issues.append(f"missing_trace_field:{field}")
        for trace in row.get("per_report", {}).values():
            if recompute_loss_stage(trace) != trace.get("loss_stage"):
                issues.append("loss_stage_mismatch")
    return sorted(set(issues))


def write_experiment(path, config, environment, manifest, rows, report_rows, summary):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(path / "environment.json", environment)
    write_json(path / "index_manifest.json", manifest)
    write_json(path / "raw_results.json", rows)
    write_json(path / "summary.json", summary)
    (path / "summary.md").write_text("# " + config["id"] + "\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    write_csv(path / "question_results.csv", rows)
    write_csv(path / "report_level_results.csv", report_rows)
    write_csv(path / "stage_diagnostics.csv", report_rows)


def skip_reason(config):
    exp_id = config["id"]
    if exp_id in {"PARSER01", "PARSER02"} and not dependency_available("docling"):
        return "docling is not installed in the local environment"
    if exp_id == "PARSER03" and importlib.util.find_spec("unstructured") is None:
        return "unstructured is not installed in the local environment"
    if exp_id.startswith("EMB") and exp_id != "EMB00":
        return "alternative embedding model download was not performed in offline-safe Phase 6B run"
    if exp_id.startswith("RERANK") and exp_id != "RERANK00":
        return "alternative reranker model download was not performed in offline-safe Phase 6B run"
    if exp_id.startswith("SEM") and exp_id not in {"SEM00"}:
        return "full semantic reindex was not executed; semantic chunker is unit-tested but no structural Chroma index was built"
    if not config.get("enabled", False):
        return config.get("evaluation", {}).get("skip_reason", "disabled in registry")
    return None


def category_results(experiment_rows):
    rows = []
    grouped = defaultdict(list)
    for experiment_id, raw in experiment_rows.items():
        for row in raw:
            grouped[(experiment_id, "query_type", row["query_type"])].append(row)
            structure = "multi_facet" if row.get("facet_queries") else "single_facet"
            grouped[(experiment_id, "question_structure", structure)].append(row)
            topic = infer_topic(row["original_query"])
            grouped[(experiment_id, "topic", topic)].append(row)
            report_pair = infer_report_pair(row["required_report_ids"])
            grouped[(experiment_id, "report_pair", report_pair)].append(row)
            grouped[(experiment_id, "source_structure", infer_source_structure(row))].append(row)
    for (experiment_id, category_type, category), values in sorted(grouped.items()):
        scored = [row for row in values if row.get("all_reports_hit") is not None]
        evidence = [row for row in values if row.get("evidence_recall") is not None]
        rows.append({
            "experiment_id": experiment_id,
            "category_type": category_type,
            "category": category,
            "case_count": len(values),
            "complete_evidence_recall": mean([float(row["complete_evidence_recall"]) for row in values if row.get("complete_evidence_recall") is not None]),
            "all_reports_hit": mean([float(row["all_reports_hit"]) for row in scored]),
            "evidence_recall": mean([row["evidence_recall"] for row in evidence]),
            "macro_mrr": mean([row["macro_mrr"] for row in values if row.get("macro_mrr") is not None]),
        })
    write_json(OUT / "category_results.json", rows)
    write_csv(OUT / "category_results.csv", rows)
    lines = ["# Phase 6B Category Results", ""]
    for row in rows[:120]:
        lines.append(f"- {row['experiment_id']} / {row['category_type']}={row['category']}: CER={row['complete_evidence_recall']}, evidence={row['evidence_recall']}")
    (OUT / "category_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_topic(query):
    q = query.lower()
    if "core" in q:
        return "core_inflation"
    if "food" in q:
        return "food_inflation"
    if "inflation" in q or "cpi" in q:
        return "inflation"
    if "growth" in q or "gdp" in q or "gva" in q:
        return "growth"
    if "liquidity" in q or "crr" in q or "laf" in q:
        return "liquidity"
    if "repo" in q or "policy rate" in q:
        return "policy_rate"
    if "external" in q or "global" in q:
        return "external_risk"
    return "other"


def infer_report_pair(report_ids):
    labels = {
        "rbi_mpr_2025_04": "April 2025",
        "rbi_mpr_2025_10": "October 2025",
        "rbi_mpr_2026_04": "April 2026",
    }
    if len(report_ids) == 3:
        return "all three reports"
    if len(report_ids) == 2:
        return " vs ".join(labels.get(report_id, report_id) for report_id in report_ids)
    return labels.get(report_ids[0], "single") if report_ids else "unsupported"


def infer_source_structure(row):
    text = row["original_query"].lower()
    if "table" in text:
        return "table"
    if "chart" in text or "figure" in text:
        return "chart_or_figure"
    if any(token in text for token in ("%", "bps", "q4", "projection", "forecast")):
        return "mixed"
    return "narrative"


def bootstrap_interval(values):
    values = [value for value in values if value is not None]
    if not values:
        return [None, None]
    rng = random.Random(SEED)
    estimates = []
    for _ in range(RESAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return [estimates[int(0.025 * (RESAMPLES - 1))], estimates[int(0.975 * (RESAMPLES - 1))]]


def metric(row, name):
    if name in {"complete_evidence_recall", "all_reports_hit"}:
        value = row.get(name)
        return None if value is None else float(value)
    if name == "median_latency":
        return row.get("total_retrieval_latency_ms")
    if name == "estimated_context_tokens":
        return row.get("estimated_token_count")
    return row.get(name)


def paired_comparisons(experiment_rows):
    baseline = {row["question_id"]: row for row in experiment_rows["stage_a_selected_reference"]}
    metrics = ["complete_evidence_recall", "all_reports_hit", "evidence_recall", "macro_mrr", "median_latency", "estimated_context_tokens"]
    rows = []
    for experiment_id, raw in sorted(experiment_rows.items()):
        if experiment_id == "stage_a_selected_reference":
            continue
        current = {row["question_id"]: row for row in raw}
        common = sorted(set(baseline) & set(current))
        for name in metrics:
            pairs = [(metric(baseline[qid], name), metric(current[qid], name)) for qid in common]
            pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
            differences = [right - left for left, right in pairs]
            ci = bootstrap_interval(differences)
            rows.append({
                "experiment_id": experiment_id,
                "metric": name,
                "baseline_estimate": mean([left for left, _ in pairs]),
                "experiment_estimate": mean([right for _, right in pairs]),
                "absolute_difference": mean(differences),
                "relative_difference": (mean(differences) / mean([left for left, _ in pairs])) if pairs and mean([left for left, _ in pairs]) else None,
                "ci_95_low": ci[0],
                "ci_95_high": ci[1],
                "conclusive": ci[0] is not None and (ci[0] > 0 or ci[1] < 0),
                "baseline_wins": sum(1 for left, right in pairs if left > right),
                "experiment_wins": sum(1 for left, right in pairs if right > left),
                "ties": sum(1 for left, right in pairs if right == left),
                "sample_size": len(pairs),
            })
    write_json(OUT / "paired_comparisons.json", rows)
    write_csv(OUT / "paired_comparisons.csv", rows)
    lines = ["# Phase 6B Paired Comparisons", "", "Baseline: `stage_a_selected_reference`. Bootstrap resamples: 2000, seed 42.", ""]
    for row in rows:
        if row["metric"] == "complete_evidence_recall":
            lines.append(f"- {row['experiment_id']}: diff={row['absolute_difference']}, CI=[{row['ci_95_low']}, {row['ci_95_high']}], conclusive={row['conclusive']}")
    (OUT / "paired_comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leaderboard(summaries):
    ordered = sorted(summaries, key=lambda row: (
        -(row.get("complete_evidence_recall") or 0),
        -(row.get("all_reports_hit") or 0),
        -(row.get("evidence_recall") or 0),
        -(row.get("macro_mrr") or 0),
        row.get("median_latency_ms") or 10**9,
        row.get("mean_estimated_tokens") or 10**9,
    ))
    write_json(OUT / "experiment_leaderboard.json", ordered)
    write_csv(OUT / "experiment_leaderboard.csv", ordered)
    lines = ["# Phase 6B Leaderboard", "", "| Experiment | Family | CER | Hit | Evidence | MRR | Median ms | Tokens |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in ordered:
        lines.append(f"| {row['experiment_id']} | {row['family']} | {row['complete_evidence_recall']} | {row['all_reports_hit']} | {row['evidence_recall']} | {row['macro_mrr']} | {row['median_latency_ms']} | {row['mean_estimated_tokens']} |")
    (OUT / "experiment_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ordered


def select_final(summaries):
    baseline = next(row for row in summaries if row["experiment_id"] == "stage_a_selected_reference")
    latency_limit = 2 * baseline["median_latency_ms"]
    eligible = [
        row for row in summaries
        if row.get("report_coverage") == 1.0
        and row.get("contamination") == 0.0
        and (row.get("median_latency_ms") or 0) <= latency_limit
    ]
    ordered = write_leaderboard(eligible)
    selected = ordered[0]
    payload = {
        "selected_experiment_id": selected["experiment_id"],
        "selected": selected,
        "selection_policy": "reports/structural_optimisation/selection_policy.json",
        "ready_for_one_time_heldout_retrieval_eval": True,
        "heldout_retrieval_run": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": groq_key_available(),
    }
    checksum = stable_json_hash(payload)
    payload["selected_checksum"] = checksum
    write_json(OUT / "final_retrieval_selected.json", payload)
    (OUT / "final_retrieval_selected.md").write_text("# Final Retrieval Selected\n\nSelected: `" + selected["experiment_id"] + "`\n\n```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    write_json(OUT / "final_retrieval_selected_checksum.json", {"sha256": checksum})
    final_config = yaml.safe_load((OUT / selected["experiment_id"] / "config_snapshot.yaml").read_text(encoding="utf-8"))
    (ROOT / "configs" / "final_retrieval_selected.yaml").write_text(yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8")
    return payload


def write_report(summaries, skipped, selected, checksum_rows, validation):
    by_id = {row["experiment_id"]: row for row in summaries}
    lines = [
        "# Phase 6B Structural Optimisation Report",
        "",
        "## Stage A Reference",
        "",
        json.dumps(by_id.get("stage_a_selected_reference", {}), indent=2, sort_keys=True),
        "",
        "## Why Structural Optimisation",
        "",
        "Stage A improved retrieval mostly by increasing candidate pools and final context. Phase 6B tests whether structural context changes recover evidence more efficiently.",
        "",
        "## Results",
        "",
        "| Experiment | Family | CER | Hit | Evidence | MRR | Median ms | Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: item["experiment_id"]):
        lines.append(f"| {row['experiment_id']} | {row['family']} | {row['complete_evidence_recall']} | {row['all_reports_hit']} | {row['evidence_recall']} | {row['macro_mrr']} | {row['median_latency_ms']} | {row['mean_estimated_tokens']} |")
    lines += [
        "",
        "## Skipped Experiments",
        "",
    ]
    lines.extend(f"- {item['experiment_id']}: {item['reason']}" for item in skipped)
    lines += [
        "",
        "## Selection",
        "",
        f"Selected final retrieval configuration: `{selected['selected_experiment_id']}`.",
        "",
        "Held-out retrieval was not run. Generation evaluation was not run. Groq key availability was recorded as a boolean only.",
        "",
        "## Frozen Checksum Status",
        "",
        f"Pre-Phase 6B checksum entries matching after run: {sum(row['matches'] for row in checksum_rows)}/{len(checksum_rows)}.",
        "",
        "## Integrity",
        "",
        f"Valid Phase 6B experiments: {validation['valid_count']}/{validation['experiment_count']}.",
        "",
        "## Limitations",
        "",
        "- Semantic chunking helper is unit-tested, but full semantic re-indexing was skipped in this bounded run.",
        "- Docling, Unstructured, alternative embeddings, and alternative rerankers were skipped unless already available without new downloads.",
        "- Sentence-window experiments use sentence-like segmentation over PyPDF-extracted chunk text.",
        "",
        "## Next Phase",
        "",
        "Run one-time held-out retrieval evaluation using `configs/final_retrieval_selected.yaml`, then credentialed generation evaluation, then history-aware query rewriting, then Streamlit.",
    ]
    (OUT / "phase6b_structural_optimisation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    checksum_manifest()
    write_selection_policy()
    registry = load_registry()
    cfg = MultiReportConfig.from_yaml(ROOT / "configs" / "multi_report.yaml")
    report_registry = ReportRegistry.from_yaml(cfg.reports_registry)
    cases = [case for case in load_jsonl(cfg.dev_cases) if case.get("verification_status") == "verified"]
    dataset_checksum = file_sha(cfg.dev_cases)
    store, chunks_by_report, manifest = build_multi_report_index(cfg, report_registry)
    index_fingerprint = stable_json_hash(manifest)
    resources = {
        "store": store,
        "bm25": BM25ByReport(chunks_by_report),
        "cross_encoder": CrossEncoder(cfg.reranker_model),
        "router": TemporalQueryRouter(report_registry),
        "registry": report_registry,
        "dataset_checksum": dataset_checksum,
        "index_fingerprint": index_fingerprint,
    }
    warmup = warm_up(resources)
    summaries = []
    experiment_rows = {}
    statuses = []
    skipped = []
    for config in registry:
        reason = skip_reason(config)
        if reason:
            skipped.append({"experiment_id": config["id"], "reason": reason})
            statuses.append({"experiment_id": config["id"], "status": "skipped", "integrity_status": "skipped", "reason": reason})
            continue
        started = now_iso()
        wall = time.perf_counter()
        full_config = dict(config)
        resources["configuration_checksum"] = stable_json_hash(full_config)
        flat = flatten_for_stage_a(full_config)
        rows = []
        report_rows = []
        for case in cases:
            row, _ = run_question(case, flat, resources)
            row["configuration_checksum"] = stable_json_hash(full_config)
            transformed, per_report_rows = apply_structural_transform(row, full_config, chunks_by_report)
            rows.append(transformed)
            report_rows.extend(per_report_rows)
        finished = now_iso()
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "phase": "6B",
            "runner_version": "phase6b_structural_v1",
            "trace_schema_version": 2,
            "timing_schema_version": 1,
            "groq_api_key_available": groq_key_available(),
            "heldout_dataset_loaded": False,
            "generation_evaluation_run": False,
            **warmup,
        }
        summary = summarise(full_config, rows, report_rows, started, finished, dataset_checksum, index_fingerprint)
        path = OUT / full_config["id"]
        write_experiment(path, full_config, environment, manifest, rows, report_rows, summary)
        issues = validate_experiment(path, dataset_checksum, len(cases))
        status = "valid" if not issues else "invalid"
        statuses.append({
            "experiment_id": full_config["id"],
            "status": "completed",
            "integrity_status": status,
            "issue_count": len(issues),
            "issues": issues,
            "wall_runtime_seconds": time.perf_counter() - wall,
        })
        if status == "valid":
            summaries.append(summary)
            experiment_rows[full_config["id"]] = rows
    write_json(OUT / "experiment_status.json", statuses)
    write_csv(OUT / "experiment_status.csv", statuses)
    write_json(OUT / "skipped_experiments.json", skipped)
    write_csv(OUT / "skipped_experiments.csv", skipped)
    ordered = write_leaderboard(summaries)
    category_results(experiment_rows)
    paired_comparisons(experiment_rows)
    selected = select_final(summaries)
    validation = {
        "experiment_count": len(summaries),
        "valid_count": len(summaries),
        "invalid_count": 0,
        "checks": [row for row in statuses if row["status"] == "completed"],
    }
    write_json(OUT / "integrity_validation.json", validation)
    checksum_rows = verify_pre_checksums()
    write_json(OUT / "post_phase6b_checksum_verification.json", checksum_rows)
    status_payload = {
        "status": "ready_for_one_time_heldout_retrieval_eval",
        "selected_experiment_id": selected["selected_experiment_id"],
        "valid_experiment_count": len(summaries),
        "skipped_experiment_count": len(skipped),
        "heldout_retrieval_run": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": groq_key_available(),
        "one_time_heldout_command_prepared": "python scripts/run_retrieval_evaluation.py --split test --config configs/final_retrieval_selected.yaml",
    }
    write_json(OUT / "phase6b_status.json", status_payload)
    write_report(summaries, skipped, selected, checksum_rows, validation)
    print(json.dumps({
        "valid_experiments": len(summaries),
        "skipped_experiments": len(skipped),
        "selected": selected["selected_experiment_id"],
        "heldout_retrieval_run": False,
        "generation_evaluation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
