# Held-out Retrieval Summary

Split: held-out test
Cases: 15 total; 13 scored retrieval cases

| Metric | Value | 95% CI |
|---|---:|---|
| complete_evidence_recall | 0.38461538461538464 | [0.17709707797762575, 0.644771084873343] (wilson) |
| all_reports_hit | 0.38461538461538464 | [0.17709707797762575, 0.644771084873343] (wilson) |
| evidence_recall | 0.5512820512820513 | [0.32051282051282054, 0.7564102564102565] (bootstrap_mean) |
| macro_report_mrr | 0.258974358974359 | [0.15, 0.358974358974359] (bootstrap_mean) |
| report_coverage | 1.0 | [1.0, 1.0] (bootstrap_mean) |
| single_report_contamination | 0.0 | [0.0, 0.39033428790216523] (wilson) |
| mean_latency_ms | 2468.2818538500355 | [1804.3284846129468, 3210.353215390709] (bootstrap_mean) |
| median_latency_ms | 2342.363799980376 |  |
| p95_latency_ms | 5376.127700030338 |  |
| mean_estimated_tokens | 2134.846153846154 | [1754.5384615384614, 2559.923076923077] (bootstrap_mean) |
| mean_selected_chunks | 8.923076923076923 | [7.3076923076923075, 10.692307692307692] (bootstrap_mean) |

## Category metrics

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

## Loss stages

{
  "evidence_found": 10,
  "lost_by_quota": 5,
  "lost_in_fusion": 4,
  "not_in_candidate_union": 4
}
