# Final Retrieval and Generation Evaluation Report

Temporal multi-document RAG for RBI Monetary Policy Reports

Scope: policy stance and narrative evolution across April 2025, October 2025, and April 2026.

This report is generated from saved JSON/CSV artifacts. It is not production-ready.

## Final retrieval configuration

Config validation: passed
Selected experiment: `ADJ00`

## Pre-run checksum validation

Status: passed

## One-time held-out retrieval evaluation

Status: passed
Cases: 15 total; 13 scored.

| Metric | Held-out value |
|---|---:|
| complete_evidence_recall | 0.38461538461538464 |
| all_reports_hit | 0.38461538461538464 |
| evidence_recall | 0.5512820512820513 |
| macro_report_mrr | 0.258974358974359 |
| report_coverage | 1.0 |
| single_report_contamination | 0.0 |
| mean_latency_ms | 2623.7835846214484 |
| median_latency_ms | 2495.7724999985658 |
| p95_latency_ms | 4354.438200010918 |
| mean_estimated_tokens | 2134.846153846154 |

## Development versus held-out retrieval

- complete_evidence_recall: dev=0.36666666666666664, held-out=0.38461538461538464, diff=0.017948717948717996
- all_reports_hit: dev=0.4, held-out=0.38461538461538464, diff=-0.015384615384615385
- evidence_recall: dev=0.5055555555555555, held-out=0.5512820512820513, diff=0.045726495726495786
- macro_report_mrr: dev=0.3537037037037037, held-out=0.258974358974359, diff=-0.09472934472934469
- report_coverage: dev=1.0, held-out=1.0, diff=0.0
- single_report_contamination: dev=0.0, held-out=0.0, diff=0.0
- mean_latency_ms: dev=3156.8762200108417, held-out=2623.7835846214484, diff=-533.0926353893933
- median_latency_ms: dev=2947.2414500196464, held-out=2495.7724999985658, diff=-451.4689500210807
- p95_latency_ms: dev=6063.70609998703, held-out=4354.438200010918, diff=-1709.267899976112
- mean_estimated_tokens: dev=2150.9333333333334, held-out=2134.846153846154, diff=-16.087179487179583
- mean_selected_chunks: dev=9.0, held-out=8.923076923076923, diff=-0.07692307692307665

## Retrieval category-level results

- category=commodity_assumptions: CER=0.0, Hit=0.0, Evidence=0.5, MRR=0.5
- category=core_inflation: CER=0.0, Hit=0.0, Evidence=0.5, MRR=0.25
- category=fiscal_borrowing: CER=0.0, Hit=0.0, Evidence=0.5, MRR=0.25
- category=food_inflation: CER=1.0, Hit=1.0, Evidence=1.0, MRR=0.2
- category=global_growth: CER=0.0, Hit=0.0, Evidence=0.0, MRR=0.0
- category=growth_outlook: CER=1.0, Hit=1.0, Evidence=1.0, MRR=0.5
- category=inflation_outlook: CER=1.0, Hit=1.0, Evidence=1.0, MRR=0.41666666666666663
- category=monetary_policy: CER=0.0, Hit=0.0, Evidence=0.2222222222222222, MRR=0.1111111111111111
- category=unsupported: CER=None, Hit=None, Evidence=None, MRR=None
- query_type=pairwise_comparison: CER=0.0, Hit=0.0, Evidence=0.3, MRR=0.2
- query_type=single_report: CER=0.8333333333333334, Hit=0.8333333333333334, Evidence=0.8333333333333334, MRR=0.33888888888888885
- query_type=trend_all_reports: CER=0.0, Hit=0.0, Evidence=0.3333333333333333, MRR=0.16666666666666666
- query_type=unsupported_period: CER=None, Hit=None, Evidence=None, MRR=None
- source_information_type=chart_manually_verified: CER=0.0, Hit=0.0, Evidence=0.5, MRR=0.5
- source_information_type=narrative: CER=0.4166666666666667, Hit=0.4166666666666667, Evidence=0.5555555555555556, MRR=0.2388888888888889
- source_information_type=table: CER=0.0, Hit=0.0, Evidence=0.16666666666666666, MRR=0.16666666666666666
- source_information_type=unknown: CER=None, Hit=None, Evidence=None, MRR=None

## Retrieval latency and context-size results

Mean latency: 2623.7835846214484 ms; median latency: 2495.7724999985658 ms; p95 latency: 4354.438200010918 ms.
Mean estimated tokens: 2134.846153846154; mean selected chunks: 8.923076923076923.

## Retrieval generalisation analysis

The development-versus-held-out comparison is reported for evaluation only. No held-out failures were inspected for optimisation and no retrieval configuration was changed.

## Generation evaluation readiness decision

Readiness: not_ready

## Generation evaluation status

Generation evaluation was not executed unless readiness passed. Missing generation metrics below are deliberately not fabricated.

## Generation development metrics

Not executed.

## Generation held-out metrics

Not executed.

## Metric coverage and judge failure counts

Not executed.

## Citation correctness results

Not executed.

## Temporal attribution results

Not executed.

## Comparative correctness results

Not executed.

## Abstention correctness results

Not executed.

## Failure cases summary

Held-out retrieval failures, if any, are summarized only through aggregate loss-stage counts. No tuning recommendations are made from held-out failures.

## Known limitations

- PyPDFLoader can flatten tables and chart structure.
- Held-out retrieval sample size is small; confidence intervals should be read directly.
- Generation evaluation requires a final-config-safe runner before metrics should be claimed.

## Exact next phase

- history-aware query rewriting
- conversational interface
- Streamlit application
- optional future Docling/embedding/reranker upgrade if dependencies become available

## Final project status

`heldout_retrieval_complete_generation_not_run`
