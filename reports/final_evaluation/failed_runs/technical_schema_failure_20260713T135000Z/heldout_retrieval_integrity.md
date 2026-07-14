# Held-out Retrieval Integrity Validation

Status: failed
Issues: 14

## Issues

- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:accepted_pages
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:bm25_candidate_ids
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:bm25_candidate_pages
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:candidate_union_ids
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:candidate_union_pages
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:dense_candidate_ids
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:dense_candidate_pages
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:expected_evidence
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:reranker_input_ids
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:reranker_input_pages
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:reranker_output_ids
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:reranker_output_pages
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:rrf_candidate_ids
- unsupported_test_002:rbi_mpr_2025_10:missing_trace_field:rrf_candidate_pages

## Recomputed metrics

```json
{
  "all_reports_hit": 0.38461538461538464,
  "case_count": 15,
  "complete_evidence_recall": 0.38461538461538464,
  "evidence_recall": 0.5512820512820513,
  "macro_report_mrr": 0.258974358974359,
  "mean_estimated_tokens": 2134.846153846154,
  "mean_latency_ms": 2468.2818538500355,
  "mean_selected_chunks": 8.923076923076923,
  "median_latency_ms": 2342.363799980376,
  "p95_latency_ms": 5376.127700030338,
  "report_coverage": 1.0,
  "scored_case_count": 13,
  "single_report_contamination": 0.0
}
```
