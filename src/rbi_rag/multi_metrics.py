from __future__ import annotations


def evaluate_multi_retrieval(case: dict, result: dict) -> dict:
    required = case["required_report_ids"]
    chunks = result["final_selected_chunks"]
    by_report = {report_id: [chunk for chunk in chunks if chunk.metadata["report_id"] == report_id]
                 for report_id in required}
    represented = sum(bool(values) for values in by_report.values())
    ground_truth = case.get("ground_truth", {})
    hits, reciprocal = {}, {}
    evidence_hits = {}
    for report_id in required:
        accepted = set(ground_truth.get(report_id, {}).get("accepted_pages", []))
        pages = [chunk.metadata["page"] for chunk in by_report[report_id]]
        ranks = [rank for rank, page in enumerate(pages, 1) if page in accepted]
        hits[report_id] = bool(ranks) if accepted else None
        reciprocal[report_id] = 1.0 / min(ranks) if ranks else 0.0
        expected = ground_truth.get(report_id, {}).get("expected_evidence", [])
        combined = " ".join(" ".join(chunk.page_content.lower().split()) for chunk in by_report[report_id])
        evidence_hits[report_id] = [" ".join(text.lower().split()) in combined for text in expected]
    scored_hits = [value for value in hits.values() if value is not None]
    requested_single = case["query_type"] == "single_report"
    contamination = sum(chunk.metadata["report_id"] not in required for chunk in chunks)
    evidence_values = [value for values in evidence_hits.values() for value in values]
    latency = sum(stage for report in result.get("retrieval_latency_by_stage", {}).values() for stage in report.values())
    return {
        "question_id": case["question_id"], "query_type": case["query_type"],
        "required_report_ids": required,
        "report_coverage": represented / len(required) if required else 0.0,
        "all_reports_hit": all(scored_hits) if scored_hits else None,
        "per_report_hit": hits, "per_report_mrr": reciprocal,
        "macro_report_mrr": sum(reciprocal.values()) / len(required) if required else 0.0,
        "evidence_recall": evidence_hits,
        "evidence_recall_rate": sum(evidence_values)/len(evidence_values) if evidence_values else None,
        "complete_evidence_recall": all(evidence_values) if evidence_values else None,
        "cross_report_contamination_count": contamination if requested_single else None,
        "cross_report_contamination": bool(contamination) if requested_single else None,
        "final_chunk_count_by_report": result["final_chunk_quota_by_report"],
        "warnings": result["missing_report_warnings"],
        "retrieved_chunk_ids": [chunk.metadata["chunk_id"] for chunk in chunks],
        "retrieved_pages_by_report": {
            report_id: [chunk.metadata["page"] for chunk in values]
            for report_id, values in by_report.items()
        },
        "retrieval_latency_ms": latency,
    }


def summarize_multi_results(rows: list[dict], *, resamples=2000, confidence=.95, seed=42) -> dict:
    from .uncertainty import bootstrap_mean_interval, wilson_interval
    scored = [row for row in rows if row["all_reports_hit"] is not None]
    singles = [row for row in rows if row["cross_report_contamination"] is not None]
    evaluable=[r for r in rows if r["required_report_ids"] and r["query_type"] != "unsupported_period"]
    coverage=[r["report_coverage"] for r in evaluable]
    mrr=[r["macro_report_mrr"] for r in evaluable]
    evidence=[r["evidence_recall_rate"] for r in rows if r["evidence_recall_rate"] is not None]
    binary=[r["all_reports_hit"] for r in scored]
    contam=[r["cross_report_contamination"] for r in singles]
    def estimate(values, binary_metric=False):
        point=sum(values)/len(values) if values else None
        interval=wilson_interval(sum(values),len(values),confidence) if binary_metric else bootstrap_mean_interval(values,resamples=resamples,confidence=confidence,seed=seed)
        return {"point_estimate":point,"ci_95_low":interval[0],"ci_95_high":interval[1],"n":len(values)}
    return {
        "case_count": len(rows),
        "mean_report_coverage": estimate(coverage),
        "all_reports_hit_rate": estimate(binary, True),
        "macro_report_mrr": estimate(mrr),
        "evidence_recall": estimate(evidence),
        "complete_evidence_recall": estimate([r["complete_evidence_recall"] for r in rows if r["complete_evidence_recall"] is not None],True),
        "cross_report_contamination_rate": estimate(contam, True),
        "latency_ms": {"mean":sum(r["retrieval_latency_ms"] for r in rows)/len(rows) if rows else None,
                       "median":sorted(r["retrieval_latency_ms"] for r in rows)[len(rows)//2] if rows else None,
                       "p95":sorted(r["retrieval_latency_ms"] for r in rows)[min(len(rows)-1,int(.95*len(rows)))] if rows else None},
        "bootstrap_resamples": resamples, "confidence_level": confidence, "random_seed": seed,
    }
