# Development vs Held-out Retrieval Comparison

Held-out results are evaluation-only; no retrieval tuning is performed from this comparison.

| Metric | Development | Held-out | Abs diff | Rel diff |
|---|---:|---:|---:|---:|
| complete_evidence_recall | 0.36666666666666664 | 0.38461538461538464 | 0.017948717948717996 | 0.048951048951049084 |
| all_reports_hit | 0.4 | 0.38461538461538464 | -0.015384615384615385 | -0.038461538461538464 |
| evidence_recall | 0.5055555555555555 | 0.5512820512820513 | 0.045726495726495786 | 0.09044801352493673 |
| macro_report_mrr | 0.3537037037037037 | 0.258974358974359 | -0.09472934472934469 | -0.2678211840515504 |
| report_coverage | 1.0 | 1.0 | 0.0 | 0.0 |
| single_report_contamination | 0.0 | 0.0 | 0.0 | None |
| mean_latency_ms | 3156.8762200108417 | 2623.7835846214484 | -533.0926353893933 | -0.16886713264530925 |
| median_latency_ms | 2947.2414500196464 | 2495.7724999985658 | -451.4689500210807 | -0.15318356425058802 |
| p95_latency_ms | 6063.70609998703 | 4354.438200010918 | -1709.267899976112 | -0.2818850174779691 |
| mean_estimated_tokens | 2150.9333333333334 | 2134.846153846154 | -16.087179487179583 | -0.007479162295676102 |
| mean_selected_chunks | 9.0 | 8.923076923076923 | -0.07692307692307665 | -0.008547008547008517 |

## Category-level differences

- category=commodity_assumptions: dev CER=0.4, held-out CER=0.0
- category=core_inflation: dev CER=0.5, held-out CER=0.0
- category=fiscal_borrowing: dev CER=0.8, held-out CER=0.0
- category=food_inflation: dev CER=0.6, held-out CER=1.0
- category=global_growth: dev CER=0.0, held-out CER=0.0
- category=growth_outlook: dev CER=0.0, held-out CER=1.0
- category=inflation_outlook: dev CER=0.0, held-out CER=1.0
- category=monetary_policy: dev CER=0.16666666666666666, held-out CER=0.0
- category=unsupported: dev CER=None, held-out CER=None
- query_type=pairwise_comparison: dev CER=0.25, held-out CER=0.0
- query_type=single_report: dev CER=0.5833333333333334, held-out CER=0.8333333333333334
- query_type=trend_all_reports: dev CER=0.16666666666666666, held-out CER=0.0
- query_type=unsupported_period: dev CER=None, held-out CER=None
- source_information_type=chart_manually_verified: dev CER=0.0, held-out CER=0.0
- source_information_type=narrative: dev CER=0.391304347826087, held-out CER=0.4166666666666667
- source_information_type=table: dev CER=0.25, held-out CER=0.0
- source_information_type=unknown: dev CER=None, held-out CER=None
