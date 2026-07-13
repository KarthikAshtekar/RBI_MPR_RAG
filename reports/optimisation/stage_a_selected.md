# Stage A Selected

Selected: `QUOTA_LARGE`

```json
{
  "configuration_checksum": "93eb8a21539ea2593dfc48a41c437e7466466917b072c99704762943b38f88b2",
  "dataset_checksum": "bf7cf18e02abb87a6e22fe9b686d8f93fa34ceb3f67abc575741c0a63080b56d",
  "generation_evaluation_run": false,
  "groq_api_key_available": true,
  "heldout_retrieval_run": false,
  "index_fingerprint": "d30cf1f43c1f86d055df51e25352b1f02a3c663c62794ffddcaf8e7f83b7c4b7",
  "selected": {
    "all_reports_hit": 0.4,
    "case_count": 34,
    "complete_evidence_recall": 0.36666666666666664,
    "config": {
      "bk": 50,
      "bw": 1,
      "dk": 50,
      "dw": 1,
      "family": "context_quota",
      "id": "QUOTA_LARGE",
      "quota": [
        6,
        5,
        4
      ],
      "retain": 30,
      "rrf": 60
    },
    "configuration_checksum": "93eb8a21539ea2593dfc48a41c437e7466466917b072c99704762943b38f88b2",
    "contamination": 0.0,
    "dataset_sha256": "bf7cf18e02abb87a6e22fe9b686d8f93fa34ceb3f67abc575741c0a63080b56d",
    "evidence_recall": 0.5055555555555555,
    "experiment_id": "QUOTA_LARGE",
    "family": "context_quota",
    "finished_at": "2026-07-13T10:17:18.034992+00:00",
    "index_fingerprint": "d30cf1f43c1f86d055df51e25352b1f02a3c663c62794ffddcaf8e7f83b7c4b7",
    "loss_stage_counts": {
      "evidence_found": 28,
      "lost_by_quota": 13,
      "lost_in_fusion": 7,
      "not_in_candidate_union": 6
    },
    "macro_mrr": 0.3537037037037037,
    "mean_estimated_tokens": 2150.9333333333334,
    "mean_latency_ms": 5207.6061633299105,
    "mean_repeated_text_ratio": 0.0,
    "mean_selected_characters": 8602.466666666667,
    "mean_selected_chunks": 9.0,
    "mean_unique_pages": 7.633333333333334,
    "median_latency_ms": 4855.378049978754,
    "p95_latency_ms": 9888.696599984542,
    "report_coverage": 1.0,
    "report_level_row_count": 54,
    "started_at": "2026-07-13T10:14:41.692684+00:00"
  },
  "selected_checksum": "c98c4fb4dc9d464fcdef2a5ca7674d065b558073ed70dbb750d52ff7ee917f5a",
  "selected_experiment_id": "QUOTA_LARGE",
  "selection_policy": "reports/optimisation/selection_policy.json"
}
```
