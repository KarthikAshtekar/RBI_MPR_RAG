from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbi_rag.final_evaluation import (
    assert_config_not_mutated,
    build_dev_vs_heldout_comparison,
    build_retrieval_summary,
    canonicalise_retrieval_row,
    category_metrics,
    contains_groq_secret,
    decide_generation_readiness,
    file_sha,
    groq_key_available,
    report_level_rows,
    stable_json_hash,
    validate_final_status,
    validate_heldout_dataset_manifest,
    validate_heldout_integrity,
    write_csv,
    write_dev_vs_heldout_comparison,
    write_final_config_validation,
    write_json,
    write_pre_final_checksums,
)
from rbi_rag.multi_config import MultiReportConfig
from rbi_rag.multi_evaluation import load_jsonl
from rbi_rag.multi_index import build_multi_report_index
from rbi_rag.report_bm25 import BM25ByReport
from rbi_rag.report_registry import ReportRegistry
from rbi_rag.stage_a_runner import run_question, warm_up
from rbi_rag.temporal_router import TemporalQueryRouter

from scripts.run_phase6b_structural_experiments import (
    apply_structural_transform,
    flatten_for_stage_a,
)


OUT = ROOT / "reports" / "final_evaluation"
CONFIG_PATH = ROOT / "configs" / "final_retrieval_selected.yaml"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 7 final held-out retrieval evaluation.")
    root.add_argument("--config", type=Path, default=Path("configs/final_retrieval_selected.yaml"))
    root.add_argument("--confirm-one-time-heldout", action="store_true")
    root.add_argument("--precheck-only", action="store_true")
    root.add_argument("--archive-failed-technical-rerun", action="store_true")
    return root


def _chunk_lookup(chunks_by_report: dict[str, list]) -> dict[str, object]:
    return {
        chunk.metadata["chunk_id"]: chunk
        for chunks in chunks_by_report.values()
        for chunk in chunks
    }


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_summary_md(summary: dict, path: Path) -> None:
    lines = [
        "# Held-out Retrieval Summary",
        "",
        "Split: held-out test",
        f"Cases: {summary['case_count']} total; {summary['scored_case_count']} scored retrieval cases",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---:|---|",
    ]
    metric_names = [
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
    ]
    for name in metric_names:
        ci = summary.get("confidence_intervals", {}).get(name)
        interval = "" if not ci else f"[{ci.get('ci_95_low')}, {ci.get('ci_95_high')}] ({ci.get('method')})"
        lines.append(f"| {name} | {summary.get(name)} | {interval} |")
    lines += ["", "## Category metrics", ""]
    for row in summary.get("category_metrics", []):
        lines.append(f"- {row['category_type']}={row['category']}: CER={row.get('complete_evidence_recall')}, Hit={row.get('all_reports_hit')}, Evidence={row.get('evidence_recall')}, MRR={row.get('macro_report_mrr')}")
    lines += ["", "## Loss stages", "", json.dumps(summary.get("loss_stage_counts", {}), indent=2, sort_keys=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_integrity_md(payload: dict, path: Path) -> None:
    lines = [
        "# Held-out Retrieval Integrity Validation",
        "",
        f"Status: {payload['status']}",
        f"Issues: {payload['issue_count']}",
        "",
    ]
    if payload["issues"]:
        lines += ["## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    lines += ["", "## Recomputed metrics", "", "```json", json.dumps(payload["recomputed_metrics"], indent=2, sort_keys=True), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readiness(readiness: dict) -> None:
    write_json(OUT / "generation_eval_readiness.json", readiness)
    lines = [
        "# Generation Evaluation Readiness",
        "",
        f"Status: {readiness['status']}",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, value in readiness["checks"].items():
        lines.append(f"| {name} | {value} |")
    if readiness.get("reason"):
        lines += ["", f"Reason: {readiness['reason']}"]
    (OUT / "generation_eval_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_dev_rows(chunk_lookup: dict[str, object]) -> list[dict]:
    raw = json.loads((ROOT / "reports/structural_optimisation/ADJ00/raw_results.json").read_text(encoding="utf-8"))
    cases = {
        case["question_id"]: case
        for case in load_jsonl(ROOT / "data/evaluation/multi_report_dev.jsonl")
        if case.get("verification_status") == "verified"
    }
    return [canonicalise_retrieval_row(row, cases.get(row["question_id"], {}), chunk_lookup) for row in raw]


def _write_final_report(status: dict) -> None:
    def load(name: str, default=None):
        path = OUT / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    checksum = load("pre_final_eval_checksums.json", {})
    config = load("final_retrieval_config_validation.json", {})
    heldout = load("heldout_retrieval_summary.json", {})
    integrity = load("heldout_retrieval_integrity.json", {})
    readiness = load("generation_eval_readiness.json", {})
    comparison = load("dev_vs_heldout_retrieval_comparison.json", {})
    lines = [
        "# Final Retrieval and Generation Evaluation Report",
        "",
        "Temporal multi-document RAG for RBI Monetary Policy Reports",
        "",
        "Scope: policy stance and narrative evolution across April 2025, October 2025, and April 2026.",
        "",
        "This report is generated from saved JSON/CSV artifacts. It is not production-ready.",
        "",
        "## Final retrieval configuration",
        "",
        f"Config validation: {config.get('status')}",
        f"Selected experiment: `{config.get('selected_experiment_id')}`",
        "",
        "## Pre-run checksum validation",
        "",
        f"Status: {checksum.get('verification', {}).get('status')}",
        "",
        "## One-time held-out retrieval evaluation",
        "",
        f"Status: {integrity.get('status')}",
        f"Cases: {heldout.get('case_count')} total; {heldout.get('scored_case_count')} scored.",
        "",
        "| Metric | Held-out value |",
        "|---|---:|",
    ]
    for name in [
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
    ]:
        lines.append(f"| {name} | {heldout.get(name)} |")
    lines += [
        "",
        "## Development versus held-out retrieval",
        "",
    ]
    for row in comparison.get("metric_comparison", []):
        lines.append(f"- {row['metric']}: dev={row['development']}, held-out={row['heldout']}, diff={row['absolute_difference_heldout_minus_dev']}")
    lines += [
        "",
        "## Retrieval category-level results",
        "",
    ]
    for row in heldout.get("category_metrics", []):
        lines.append(f"- {row['category_type']}={row['category']}: CER={row.get('complete_evidence_recall')}, Hit={row.get('all_reports_hit')}, Evidence={row.get('evidence_recall')}, MRR={row.get('macro_report_mrr')}")
    lines += [
        "",
        "## Retrieval latency and context-size results",
        "",
        f"Mean latency: {heldout.get('mean_latency_ms')} ms; median latency: {heldout.get('median_latency_ms')} ms; p95 latency: {heldout.get('p95_latency_ms')} ms.",
        f"Mean estimated tokens: {heldout.get('mean_estimated_tokens')}; mean selected chunks: {heldout.get('mean_selected_chunks')}.",
        "",
        "## Retrieval generalisation analysis",
        "",
        "The development-versus-held-out comparison is reported for evaluation only. No held-out failures were inspected for optimisation and no retrieval configuration was changed.",
        "",
        "## Generation evaluation readiness decision",
        "",
        f"Readiness: {readiness.get('status')}",
        "",
        "## Generation evaluation status",
        "",
        "Generation evaluation was not executed unless readiness passed. Missing generation metrics below are deliberately not fabricated.",
        "",
        "## Generation development metrics",
        "",
        "Not executed.",
        "",
        "## Generation held-out metrics",
        "",
        "Not executed.",
        "",
        "## Metric coverage and judge failure counts",
        "",
        "Not executed.",
        "",
        "## Citation correctness results",
        "",
        "Not executed.",
        "",
        "## Temporal attribution results",
        "",
        "Not executed.",
        "",
        "## Comparative correctness results",
        "",
        "Not executed.",
        "",
        "## Abstention correctness results",
        "",
        "Not executed.",
        "",
        "## Failure cases summary",
        "",
        "Held-out retrieval failures, if any, are summarized only through aggregate loss-stage counts. No tuning recommendations are made from held-out failures.",
        "",
        "## Known limitations",
        "",
        "- PyPDFLoader can flatten tables and chart structure.",
        "- Held-out retrieval sample size is small; confidence intervals should be read directly.",
        "- Generation evaluation requires a final-config-safe runner before metrics should be claimed.",
        "",
        "## Exact next phase",
        "",
        "- history-aware query rewriting",
        "- conversational interface",
        "- Streamlit application",
        "- optional future Docling/embedding/reranker upgrade if dependencies become available",
        "",
        "## Final project status",
        "",
        f"`{status['status']}`",
    ]
    (OUT / "final_retrieval_and_generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_status(status_value: str, details: dict) -> dict:
    if not validate_final_status(status_value):
        raise ValueError(f"Invalid final project status: {status_value}")
    payload = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status_value,
        **details,
    }
    if contains_groq_secret(payload):
        raise RuntimeError("Refusing to serialize payload containing Groq secret material.")
    write_json(OUT / "final_project_status.json", payload)
    lines = [
        "# Final Project Status",
        "",
        f"Status: `{status_value}`",
        "",
    ]
    for key, value in details.items():
        lines.append(f"- {key}: {value}")
    (OUT / "final_project_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_prechecks(config_path: Path) -> tuple[dict, dict]:
    pre = write_pre_final_checksums(ROOT, Path("reports/final_evaluation"))
    config_validation = write_final_config_validation(ROOT, Path("reports/final_evaluation"))
    if pre["verification"]["status"] != "passed":
        _write_status("heldout_retrieval_failed_generation_not_run", {
            "heldout_retrieval_run": False,
            "generation_evaluation_run": False,
            "failure_reason": "pre_final_checksum_validation_failed",
        })
        raise RuntimeError("Pre-final checksum validation failed; held-out retrieval was not run.")
    if config_validation["status"] != "passed":
        _write_status("heldout_retrieval_failed_generation_not_run", {
            "heldout_retrieval_run": False,
            "generation_evaluation_run": False,
            "failure_reason": "final_retrieval_config_validation_failed",
        })
        raise RuntimeError("Final retrieval config validation failed; held-out retrieval was not run.")
    return pre, config_validation


def run_heldout_retrieval(config_path: Path, *, confirm: bool, archive_failed: bool = False) -> dict:
    from rbi_rag.final_evaluation import ensure_one_time_guard

    ensure_one_time_guard(OUT, confirm=confirm, archive_failed=archive_failed)
    config_file_sha = file_sha(config_path)
    config = _read_config(config_path)
    config_checksum = stable_json_hash(config)
    cfg = MultiReportConfig.from_yaml(ROOT / "configs/multi_report.yaml")
    registry = ReportRegistry.from_yaml(cfg.reports_registry)
    dataset_validation = validate_heldout_dataset_manifest(ROOT)
    cases = [
        case for case in load_jsonl(cfg.test_cases)
        if case.get("verification_status") == "verified"
    ]
    store, chunks_by_report, manifest = build_multi_report_index(cfg, registry)
    chunks = _chunk_lookup(chunks_by_report)
    resources = {
        "store": store,
        "bm25": BM25ByReport(chunks_by_report),
        "cross_encoder": CrossEncoder(cfg.reranker_model),
        "router": TemporalQueryRouter(registry),
        "registry": registry,
        "dataset_checksum": dataset_validation["actual_sha256"],
        "index_fingerprint": stable_json_hash(manifest),
        "configuration_checksum": config_checksum,
    }
    warmup = warm_up(resources)
    flat = flatten_for_stage_a(config)
    rows = []
    report_rows = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_payload = {
        "status": "started",
        "started_at_utc": started_at,
        "config_path": str(config_path.relative_to(ROOT)).replace("/", "\\"),
        "config_file_sha256": config_file_sha,
        "configuration_checksum": config_checksum,
        "dataset_sha256": dataset_validation["actual_sha256"],
        "case_count": len(cases),
    }
    write_json(OUT / "heldout_retrieval_run_manifest.json", manifest_payload)
    for case in cases:
        row, _ = run_question(case, flat, resources)
        row["split"] = "test"
        row["configuration_checksum"] = config_checksum
        transformed, per_report = apply_structural_transform(row, config, chunks_by_report)
        transformed["split"] = "test"
        transformed["configuration_checksum"] = config_checksum
        canonical = canonicalise_retrieval_row(transformed, case, chunks)
        rows.append(canonical)
        report_rows.extend(report_level_rows([canonical]))
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = build_retrieval_summary(
        rows,
        report_rows,
        split="test",
        config_checksum=config_checksum,
        config_file_sha256=config_file_sha,
        dataset_sha256=dataset_validation["actual_sha256"],
        index_fingerprint=stable_json_hash(manifest),
        started_at_utc=started_at,
        finished_at_utc=finished_at,
    )
    write_json(OUT / "heldout_retrieval_raw_results.json", rows)
    write_csv(OUT / "heldout_retrieval_question_results.csv", rows)
    write_csv(OUT / "heldout_retrieval_report_level_results.csv", report_rows)
    write_json(OUT / "heldout_retrieval_summary.json", summary)
    _write_summary_md(summary, OUT / "heldout_retrieval_summary.md")
    manifest_payload.update({
        "status": "completed",
        "finished_at_utc": finished_at,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "phase": "7",
            "heldout_dataset_loaded": True,
            "generation_evaluation_run": False,
            "groq_api_key_available": groq_key_available(ROOT),
            **warmup,
        },
    })
    write_json(OUT / "heldout_retrieval_run_manifest.json", manifest_payload)
    return {
        "rows": rows,
        "report_rows": report_rows,
        "summary": summary,
        "dataset_validation": dataset_validation,
        "config_file_sha256": config_file_sha,
        "configuration_checksum": config_checksum,
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    pre, _ = run_prechecks(config_path)
    if args.precheck_only:
        print(json.dumps({"prechecks": "passed", "output": str(OUT)}, indent=2))
        return 0
    result = run_heldout_retrieval(
        config_path,
        confirm=args.confirm_one_time_heldout,
        archive_failed=args.archive_failed_technical_rerun,
    )
    integrity = validate_heldout_integrity(
        result["rows"],
        result["summary"],
        pre,
        Path("configs/final_retrieval_selected.yaml"),
        result["dataset_validation"],
    )
    write_json(OUT / "heldout_retrieval_integrity.json", integrity)
    _write_integrity_md(integrity, OUT / "heldout_retrieval_integrity.md")
    if integrity["status"] != "passed":
        status = _write_status("heldout_retrieval_failed_generation_not_run", {
            "heldout_retrieval_run": True,
            "generation_evaluation_run": False,
            "failure_reason": "heldout_retrieval_integrity_failed",
            "heldout_integrity_issue_count": integrity["issue_count"],
        })
        _write_final_report(status)
        raise RuntimeError("Held-out retrieval integrity failed; generation was not run.")

    cfg = MultiReportConfig.from_yaml(ROOT / "configs/multi_report.yaml")
    registry = ReportRegistry.from_yaml(cfg.reports_registry)
    _, chunks_by_report, _ = build_multi_report_index(cfg, registry)
    dev_rows = _load_dev_rows(_chunk_lookup(chunks_by_report))
    write_dev_vs_heldout_comparison(dev_rows, result["rows"], OUT)

    pre_config_entry = next(
        entry for entry in pre["entries"]
        if entry["path"] == "configs\\final_retrieval_selected.yaml"
    )
    mutation = assert_config_not_mutated(config_path, pre_config_entry["sha256"])
    frozen_generation_supported = False
    readiness = decide_generation_readiness(
        heldout_completed=True,
        heldout_integrity=integrity,
        config_not_mutated=mutation["matches"],
        groq_available=groq_key_available(ROOT),
        frozen_generation_supported=frozen_generation_supported,
        retrieval_tuning_after_heldout=False,
    )
    readiness["reason"] = (
        "Generation was not run because the checked-in multi-report generation command "
        "does not consume configs/final_retrieval_selected.yaml or the Phase 7 frozen "
        "retrieval outputs; running it would evaluate the older configs/multi_report.yaml "
        "retrieval settings instead of the final ADJ00 configuration."
    )
    _write_readiness(readiness)
    status = _write_status("heldout_retrieval_complete_generation_not_run", {
        "heldout_retrieval_run": True,
        "heldout_retrieval_integrity_status": integrity["status"],
        "retrieval_config_mutated": not mutation["matches"],
        "generation_readiness_status": readiness["status"],
        "generation_evaluation_run": False,
        "generation_not_run_reason": readiness["reason"],
    })
    _write_final_report(status)
    print(json.dumps({
        "status": status["status"],
        "heldout_retrieval": {
            key: result["summary"].get(key)
            for key in (
                "complete_evidence_recall",
                "all_reports_hit",
                "evidence_recall",
                "macro_report_mrr",
                "report_coverage",
                "single_report_contamination",
            )
        },
        "generation_readiness": readiness["status"],
        "output": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
