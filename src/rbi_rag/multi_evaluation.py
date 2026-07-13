from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .evaluation.reporting import atomic_write_json
from .multi_metrics import evaluate_multi_retrieval, summarize_multi_results


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_router(router, path: Path):
    rows = []
    for case in load_jsonl(path):
        plan = router.route(case["query"])
        correct = (plan.query_type == case["expected_query_type"] and
                   list(plan.report_ids) == case["expected_report_ids"])
        rows.append({
            "case_id": case["case_id"], "query": case["query"],
            "expected_query_type": case["expected_query_type"],
            "actual_query_type": plan.query_type,
            "expected_report_ids": case["expected_report_ids"],
            "actual_report_ids": list(plan.report_ids), "correct": correct,
            "routing_reason": plan.routing_reason,
        })
    return rows, sum(row["correct"] for row in rows) / len(rows) if rows else None


def run_multi_retrieval_evaluation(router, retriever, cases_path: Path, output_directory: Path,
                                   *, split="dev", resamples=2000, confidence=.95, seed=42):
    rows = []
    all_cases = load_jsonl(cases_path)
    scored_cases = [case for case in all_cases if case.get("verification_status") == "verified"]
    excluded = [case["question_id"] for case in all_cases if case.get("verification_status") != "verified"]
    for case in scored_cases:
        plan = router.route(case["question"])
        result = retriever.retrieve_from_query_plan(plan)
        metric = evaluate_multi_retrieval(case, result)
        metric["query_plan"] = asdict(plan)
        metric["retrieval_trace"] = {
            "reports_searched": result["report_ids_searched"],
            "dense_candidates_by_report": result["dense_results_by_report"],
            "bm25_candidates_by_report": result["bm25_results_by_report"],
            "rrf_candidates_by_report": result["rrf_results_by_report"],
            "reranked_candidates_by_report": result["reranked_results_by_report"],
            "selected_chunks": [{"report_id": d.metadata["report_id"], "page": d.metadata["page"],
                                  "chunk_id": d.metadata["chunk_id"]} for d in result["final_selected_chunks"]],
            "final_report_quotas": result["final_chunk_quota_by_report"],
            "stage_latencies": result["retrieval_latency_by_stage"],
            "warnings": result["missing_report_warnings"], "retrieval_errors": [],
        }
        rows.append(metric)
    payload = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(cases_path), "split": split, "rows": rows,
        "summary": summarize_multi_results(rows,resamples=resamples,confidence=confidence,seed=seed),
        "excluded_unverified_case_ids": excluded,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_directory / f"retrieval_{split}_raw_results.json", payload)
    if rows:
        with (output_directory / f"retrieval_{split}_question_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
    atomic_write_json(output_directory / f"retrieval_{split}_summary.json", payload["summary"])
    return payload


def write_router_results(rows, accuracy, output_directory: Path, split="dev"):
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / f"router_{split}_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    atomic_write_json(output_directory / f"router_{split}_summary.json", {"case_count": len(rows), "accuracy": accuracy})


def generate_multi_report(output_directory: Path, registry, manifest, router_accuracy, retrieval):
    missing = [report for report in manifest["reports"].values() if report["status"] == "missing"]
    summary = retrieval.get("summary", {})
    lines = [
        "# Temporal Multi-Document RAG for RBI Monetary Policy Reports", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Corpus status", "",
        "| Report | Availability | Pages | Chunks | Index status |", "|---|---|---:|---:|---|",
    ]
    for report in registry.enabled():
        info = manifest["reports"][report.report_id]
        lines.append(f"| {report.report_period} | {info['status']} | {info.get('page_count', 'n/a')} | {info.get('chunk_count', 'n/a')} | {info['indexing_status']} |")
    lines += ["", "## Router", "", f"Offline router accuracy: **{router_accuracy:.2%}**.", "",
              "## Retrieval evaluation", "",
              f"Scored cases: {summary.get('case_count', 0)}",
              f"Mean report coverage: {summary.get('mean_report_coverage')}",
              f"All-reports hit rate: {summary.get('all_reports_hit_rate')}",
              f"Macro report MRR: {summary.get('macro_report_mrr')}",
              f"Single-report cross-contamination rate: {summary.get('cross_report_contamination_rate')}", "",
              "Only verified April 2025 single-report cases are scored. Pairwise and trend factual cases await the missing PDFs.", "",
              "## Generation evaluation", "", "Not executed. GROQ_API_KEY was unavailable; no generation metrics are claimed.", "",
              "## Limitations", "", "- October 2025 and April 2026 PDFs are missing locally.",
              "- PyPDFLoader can lose table structure and chart semantics.",
              "- This system has no conversational memory or history-aware query rewriting.",
              "- This is an evaluated research baseline, not a production-ready system.", ""]
    (output_directory / "multi_report_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_full_temporal_report(output_directory: Path, registry, manifest):
    def load(name, default=None):
        path=output_directory/name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    router_dev=load("router_dev_summary.json",{}); router_test=load("router_test_summary.json",{})
    dev=load("retrieval_dev_summary.json",{}); test=load("retrieval_test_summary.json",{})
    comparison=load("architecture_comparison.json",{})
    audit=load("extraction_audit.json",{"records":[],"numeric_evaluation_exclusions":[]})
    dataset=json.loads(Path("data/evaluation/temporal_dataset_manifest.json").read_text(encoding="utf-8"))
    def metric(summary,name):
        value=summary.get(name,{})
        return f"{value.get('point_estimate','n/a')} [{value.get('ci_95_low','n/a')}, {value.get('ci_95_high','n/a')}] (n={value.get('n','n/a')})"
    lines=["# Full Temporal Retrieval Baseline","","Temporal multi-document RAG for RBI Monetary Policy Reports","",
           "## Corpus manifest","","| Report | SHA-256 | Pages | Chunks | Status |","|---|---|---:|---:|---|"]
    for report in registry.enabled():
        info=manifest["reports"][report.report_id]
        lines.append(f"| {report.report_period} | `{info.get('sha256','n/a')}` | {info.get('page_count','n/a')} | {info.get('chunk_count','n/a')} | {info.get('indexing_status')} |")
    lines += ["","## Extraction audit","",f"Audited pages: {len(audit['records'])}; pages requiring manual numeric verification: {len(audit['numeric_evaluation_exclusions'])}.","",
              "## Router evaluation","",f"Development: {router_dev.get('accuracy','n/a')} ({router_dev.get('case_count','n/a')} cases)",f"Held-out: {router_test.get('accuracy','n/a')} ({router_test.get('case_count','n/a')} cases)","",
              "## Dataset freeze","",f"Development cases: {dataset['files']['dev']['case_count']} total; {dataset['verified_scored_counts']['dev']} newly verified/scored.",f"Held-out cases: {dataset['verified_scored_counts']['test']} (frozen checksum `{dataset['files']['test']['sha256']}`).","",
              "## Report-aware retrieval","","| Split | Report coverage | All-reports hit | Macro report MRR | Evidence recall | Complete evidence recall | Contamination |","|---|---|---|---|---|---|---|",
              f"| Development | {metric(dev,'mean_report_coverage')} | {metric(dev,'all_reports_hit_rate')} | {metric(dev,'macro_report_mrr')} | {metric(dev,'evidence_recall')} | {metric(dev,'complete_evidence_recall')} | {metric(dev,'cross_report_contamination_rate')} |",
              f"| Held-out | {metric(test,'mean_report_coverage')} | {metric(test,'all_reports_hit_rate')} | {metric(test,'macro_report_mrr')} | {metric(test,'evidence_recall')} | {metric(test,'complete_evidence_recall')} | {metric(test,'cross_report_contamination_rate')} |","",
              "## Architecture comparison","","Report-aware retrieval preserves report coverage and eliminates single-report contamination; naive results and confidence intervals are saved in `architecture_comparison.json`. Overlapping intervals mean small differences should not be treated as conclusive.","",
              "## Development failure analysis","","See `dev_failure_analysis.md`; held-out failures were not inspected or used for tuning.","",
              "## Generation evaluation","","Not executed because GROQ_API_KEY is unavailable. Validated generation coverage is zero.","",
              "## Frozen configuration","","Dense/BM25 per report: 15/15; RRF k=60; reranker candidates=15; single quota=4; pairwise quota=3/report; trend quota=2/report; bootstrap=2000; seed=42.","",
              "## Limitations and next phase","","PyPDFLoader flattens tables/charts; numeric cases require manual verification. This is not production-ready. Next: run credentialed comparative generation evaluation, then add conversational memory only after generation quality is validated.",""]
    (output_directory/"full_temporal_retrieval_report.md").write_text("\n".join(lines),encoding="utf-8")
