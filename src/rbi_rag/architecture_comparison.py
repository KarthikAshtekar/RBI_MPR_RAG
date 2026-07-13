from __future__ import annotations

import csv,json
from pathlib import Path
from .evaluation.reporting import atomic_write_json
from .multi_evaluation import load_jsonl
from .multi_metrics import evaluate_multi_retrieval, summarize_multi_results


def evaluate_naive(router,retriever,cases_path,config,split):
    rows=[]
    for case in load_jsonl(cases_path):
        if case.get("verification_status")!="verified": continue
        result=retriever.retrieve_from_query_plan(router.route(case["question"]))
        rows.append(evaluate_multi_retrieval(case,result))
    payload={"split":split,"architecture":"naive_global","rows":rows,
             "summary":summarize_multi_results(rows,resamples=config.bootstrap_resamples,
                                               confidence=config.confidence_level,seed=config.random_seed)}
    atomic_write_json(config.output_directory/f"naive_{split}_raw_results.json",payload)
    return payload


def compare_architectures(config, naive_payloads):
    rows=[]; result={"splits":{}}
    for split,naive in naive_payloads.items():
        aware=json.loads((config.output_directory/f"retrieval_{split}_raw_results.json").read_text(encoding="utf-8"))
        result["splits"][split]={"report_aware":aware["summary"],"naive_global":naive["summary"]}
        for architecture,summary in result["splits"][split].items():
            row={"split":split,"architecture":architecture}
            for metric in ("mean_report_coverage","all_reports_hit_rate","macro_report_mrr","complete_evidence_recall","cross_report_contamination_rate"):
                value=summary[metric]; row[metric]=value["point_estimate"]; row[metric+"_ci_low"]=value["ci_95_low"]; row[metric+"_ci_high"]=value["ci_95_high"]
            rows.append(row)
    result["interpretation"]="Confidence intervals must be considered; overlapping intervals do not establish a conclusive difference."
    atomic_write_json(config.output_directory/"architecture_comparison.json",result)
    with (config.output_directory/"architecture_comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result
