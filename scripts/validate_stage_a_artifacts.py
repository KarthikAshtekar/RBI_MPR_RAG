from __future__ import annotations

import json
from pathlib import Path

import yaml

from rbi_rag.experiment_tracing import LATENCY_FIELDS, recompute_loss_stage, validate_latency_schema


ROOT = Path(".")
OUT = ROOT / "reports" / "optimisation"
REQUIRED = {
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


def stable_hash(value) -> str:
    from hashlib import sha256

    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def active_experiment_dirs():
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.startswith("invalid_runs") or directory.name.startswith("active_invalid"):
            continue
        if directory.name == "repaired_baseline":
            continue
        if (directory / "summary.json").exists():
            yield directory


def avg(values):
    return sum(values) / len(values) if values else None


def validate(directory: Path):
    issues = []
    files = {path.name for path in directory.iterdir() if path.is_file()}
    issues.extend(f"missing_file:{name}" for name in sorted(REQUIRED - files))
    if issues:
        return issues

    config = yaml.safe_load((directory / "config_snapshot.yaml").read_text(encoding="utf-8"))
    environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    raw = json.loads((directory / "raw_results.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

    if config.get("id") != directory.name:
        issues.append("experiment_id_mismatch")
    if summary.get("experiment_id") != directory.name:
        issues.append("summary_experiment_id_mismatch")
    if stable_hash(config) != summary.get("configuration_checksum"):
        issues.append("configuration_checksum_mismatch")
    if environment.get("heldout_dataset_loaded") is not False:
        issues.append("heldout_loaded_flag_not_false")
    if environment.get("groq_api_key_available") is not True:
        issues.append("groq_availability_not_true")
    if len(raw) != summary.get("case_count"):
        issues.append("raw_summary_count_mismatch")
    if any(row.get("split") == "test" or str(row.get("question_id", "")).startswith("test_") for row in raw):
        issues.append("heldout_case_present")

    valid = [row for row in raw if row.get("required_report_ids") and row.get("query_type") != "unsupported_period"]
    scored = [row for row in valid if row.get("all_reports_hit") is not None]
    evidence = [row for row in valid if row.get("evidence_recall") is not None]
    recomputed = {
        "report_coverage": avg([row["report_coverage"] for row in valid]),
        "all_reports_hit": avg([float(row["all_reports_hit"]) for row in scored]),
        "macro_mrr": avg([row["macro_mrr"] for row in valid]),
        "evidence_recall": avg([row["evidence_recall"] for row in evidence]),
        "complete_evidence_recall": avg([float(row["complete_evidence_recall"]) for row in evidence]),
        "contamination": avg([row["contamination"] for row in valid]),
    }
    for key, value in recomputed.items():
        if value is not None and abs(value - summary.get(key, -999)) > 1e-12:
            issues.append(f"summary_metric_mismatch:{key}")

    for row in raw:
        issues.extend(validate_latency_schema(row))
        if not all(field in row for field in LATENCY_FIELDS):
            issues.append("missing_latency_field")
        for field in (
            "selected_character_count",
            "estimated_token_count",
            "selected_chunk_count",
            "unique_page_count",
            "duplicate_chunk_count",
            "repeated_text_ratio",
            "dataset_checksum",
            "index_fingerprint",
            "configuration_checksum",
            "per_report",
        ):
            if field not in row:
                issues.append(f"missing_trace_field:{field}")
        if row.get("configuration_checksum") != summary.get("configuration_checksum"):
            issues.append("row_configuration_checksum_mismatch")
        for report_id, trace in row.get("per_report", {}).items():
            if trace.get("report_id") != report_id:
                issues.append("report_id_mismatch")
            if recompute_loss_stage(trace) != trace.get("loss_stage"):
                issues.append("loss_stage_mismatch")
            if len(trace.get("dense_candidate_ids", [])) > int(config.get("dk", 0)):
                issues.append("dense_candidate_limit_exceeded")
            if len(trace.get("bm25_candidate_ids", [])) > int(config.get("bk", 0)):
                issues.append("bm25_candidate_limit_exceeded")
            if len(trace.get("rrf_candidate_ids", [])) > int(config.get("retain", 0)):
                issues.append("rrf_candidate_limit_exceeded")
            if not set(trace.get("selected_chunk_ids_after_dedup", [])).issubset(set(trace.get("reranker_output_ids", []))):
                issues.append("selected_not_in_reranker_output")
    return sorted(set(issues))


def main():
    checks = []
    for directory in active_experiment_dirs():
        issues = validate(directory)
        checks.append({
            "experiment_id": directory.name,
            "valid": not issues,
            "issue_count": len(issues),
            "issues": issues,
        })
    payload = {
        "schema_version": 2,
        "experiment_count": len(checks),
        "valid_count": sum(item["valid"] for item in checks),
        "invalid_count": sum(not item["valid"] for item in checks),
        "checks": checks,
    }
    (OUT / "stage_a_integrity_validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage A Integrity Validation",
        "",
        f"Valid: {payload['valid_count']}/{payload['experiment_count']}",
        "",
        "| Experiment | Valid | Issues |",
        "|---|---:|---|",
    ]
    for item in checks:
        lines.append(f"| {item['experiment_id']} | {item['valid']} | {', '.join(item['issues'])} |")
    (OUT / "stage_a_integrity_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"experiments": payload["experiment_count"], "valid": payload["valid_count"], "invalid": payload["invalid_count"]}, indent=2))


if __name__ == "__main__":
    main()
