from __future__ import annotations

import csv
from pathlib import Path


def analyze_dev_failures(payload: dict, output_directory: Path):
    failures=[]
    for row in payload["rows"]:
        if row["all_reports_hit"] is True and row.get("complete_evidence_recall") in (True,None):
            continue
        trace=row["retrieval_trace"]; category="other"
        if row["query_plan"]["query_type"] != row["query_type"]:
            category="router_error"
        elif row["query_type"]=="unsupported_period":
            category="unsupported_question_failure" if not trace["warnings"] else "other"
        else:
            accepted={rid:set(pages) for rid,pages in row["retrieved_pages_by_report"].items()}
            dense_pages={rid:{v["page"] for v in vals} for rid,vals in trace["dense_candidates_by_report"].items()}
            bm25_pages={rid:{v["page"] for v in vals} for rid,vals in trace["bm25_candidates_by_report"].items()}
            missed=[rid for rid,hit in row["per_report_hit"].items() if hit is False]
            if missed and all(not dense_pages.get(rid) for rid in missed): category="dense_miss"
            elif missed and all(not bm25_pages.get(rid) for rid in missed): category="bm25_miss"
            elif row.get("evidence_recall_rate") not in (None,1.0): category="evidence_not_preserved_by_extraction"
            else: category="reranker_error"
        failures.append({"question_id":row["question_id"],"query_type":row["query_type"],
                         "failure_category":category,"all_reports_hit":row["all_reports_hit"],
                         "complete_evidence_recall":row.get("complete_evidence_recall"),
                         "notes":"Development-only diagnostic; no held-out failures inspected."})
    csv_path=output_directory/"dev_failure_analysis.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        fields=["question_id","query_type","failure_category","all_reports_hit","complete_evidence_recall","notes"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(failures)
    lines=["# Development Failure Analysis","","Held-out failures were not inspected or used for tuning.","",
           f"Failures identified: {len(failures)}","", "| Question | Type | Category |", "|---|---|---|"]
    lines += [f"| {r['question_id']} | {r['query_type']} | {r['failure_category']} |" for r in failures]
    (output_directory/"dev_failure_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return failures
