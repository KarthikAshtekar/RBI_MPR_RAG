# Phase 6B Structural Optimisation Report

## Stage A Reference

{
  "all_reports_hit": 0.4,
  "case_count": 34,
  "complete_evidence_recall": 0.36666666666666664,
  "configuration_checksum": "481f8363b1355011040c6290a05f37a24b1140c8c17c77011273567b8c0de5d0",
  "contamination": 0.0,
  "dataset_sha256": "bf7cf18e02abb87a6e22fe9b686d8f93fa34ceb3f67abc575741c0a63080b56d",
  "description": "Reproduce the Stage A selected QUOTA_LARGE configuration under Phase 6B output isolation.",
  "evidence_recall": 0.5055555555555555,
  "experiment_id": "stage_a_selected_reference",
  "family": "stage_a_reference",
  "finished_at": "2026-07-13T12:24:15.152838+00:00",
  "index_fingerprint": "778f986612e97ec5f39b23ee81e35abb281cff7ae92268fa38861ce77b7dec01",
  "loss_stage_counts": {
    "evidence_found": 28,
    "lost_by_quota": 13,
    "lost_in_fusion": 7,
    "not_in_candidate_union": 6
  },
  "macro_mrr": 0.3537037037037037,
  "mean_estimated_tokens": 2150.9333333333334,
  "mean_latency_ms": 3634.887443339297,
  "mean_repeated_text_ratio": 0.0,
  "mean_selected_characters": 8602.466666666667,
  "mean_selected_chunks": 9.0,
  "mean_unique_pages": 7.633333333333334,
  "median_latency_ms": 3406.285850040149,
  "p95_latency_ms": 6418.199100065976,
  "report_coverage": 1.0,
  "report_level_row_count": 54,
  "started_at": "2026-07-13T12:22:25.506170+00:00"
}

## Why Structural Optimisation

Stage A improved retrieval mostly by increasing candidate pools and final context. Phase 6B tests whether structural context changes recover evidence more efficiently.

## Results

| Experiment | Family | CER | Hit | Evidence | MRR | Median ms | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| ADJ00 | adjacent_expansion | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 2947.2414500196464 | 2150.9333333333334 |
| ADJ01 | adjacent_expansion | 0.3 | 0.3 | 0.45555555555555555 | 0.3025925925925926 | 3070.171599974856 | 2465.5666666666666 |
| ADJ02 | adjacent_expansion | 0.26666666666666666 | 0.3 | 0.41111111111111115 | 0.2754761904761905 | 3097.281850001309 | 2480.3333333333335 |
| ADJ03 | adjacent_expansion | 0.3333333333333333 | 0.36666666666666664 | 0.46111111111111114 | 0.3349206349206349 | 3017.500400019344 | 2034.7666666666667 |
| CPARENT01 | child_parent | 0.2 | 0.4 | 0.3055555555555555 | 0.3537037037037037 | 3400.0309000257403 | 2189.6666666666665 |
| CPARENT02 | child_parent | 0.06666666666666667 | 0.4 | 0.19444444444444445 | 0.3537037037037037 | 3534.3692999740597 | 1999.8333333333333 |
| CPARENT03 | child_parent | 0.13333333333333333 | 0.4 | 0.2611111111111111 | 0.3537037037037037 | 2955.548749974696 | 2191.133333333333 |
| EMB00 | embedding | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 4078.3949499891605 | 2150.9333333333334 |
| PARSER00 | parser | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 3005.472699966049 | 2150.9333333333334 |
| RERANK00 | reranker | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 3722.892000019783 | 2150.9333333333334 |
| SEM00 | semantic_chunking | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 3598.9076500118244 | 2150.9333333333334 |
| STRUCT_CANDIDATE_1 | combined_structural | 0.3 | 0.3 | 0.45555555555555555 | 0.3025925925925926 | 4559.01059997268 | 2465.5666666666666 |
| STRUCT_CANDIDATE_2 | combined_structural | 0.06666666666666667 | 0.4 | 0.19444444444444445 | 0.3537037037037037 | 4907.527200033655 | 1999.8333333333333 |
| STRUCT_CANDIDATE_3 | combined_structural | 0.1 | 0.4 | 0.18333333333333332 | 0.3537037037037037 | 3982.0645499858074 | 430.76666666666665 |
| SW01 | sentence_window | 0.06666666666666667 | 0.4 | 0.10555555555555556 | 0.3537037037037037 | 2646.137549978448 | 265.6666666666667 |
| SW02 | sentence_window | 0.1 | 0.4 | 0.18333333333333332 | 0.3537037037037037 | 2809.8250499751884 | 430.76666666666665 |
| SW03 | sentence_window | 0.2 | 0.4 | 0.3055555555555555 | 0.3537037037037037 | 3450.6834499770775 | 2474.9666666666667 |
| stage_a_selected_reference | stage_a_reference | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 3406.285850040149 | 2150.9333333333334 |

## Skipped Experiments

- SEM01: full semantic reindex was not executed; semantic chunker is unit-tested but no structural Chroma index was built
- SEM02: full semantic reindex was not executed; semantic chunker is unit-tested but no structural Chroma index was built
- SEM03: full semantic reindex was not executed; semantic chunker is unit-tested but no structural Chroma index was built
- PARSER01: docling is not installed in the local environment
- PARSER02: docling is not installed in the local environment
- PARSER03: unstructured is not installed in the local environment
- EMB01: alternative embedding model download was not performed in offline-safe Phase 6B run
- EMB02: alternative embedding model download was not performed in offline-safe Phase 6B run
- RERANK01: alternative reranker model download was not performed in offline-safe Phase 6B run
- RERANK02: alternative reranker model download was not performed in offline-safe Phase 6B run

## Selection

Selected final retrieval configuration: `ADJ00`.

Held-out retrieval was not run. Generation evaluation was not run. Groq key availability was recorded as a boolean only.

## Frozen Checksum Status

Pre-Phase 6B checksum entries matching after run: 716/716.

## Integrity

Valid Phase 6B experiments: 18/18.

## Limitations

- Semantic chunking helper is unit-tested, but full semantic re-indexing was skipped in this bounded run.
- Docling, Unstructured, alternative embeddings, and alternative rerankers were skipped unless already available without new downloads.
- Sentence-window experiments use sentence-like segmentation over PyPDF-extracted chunk text.

## Next Phase

Run one-time held-out retrieval evaluation using `python scripts/run_phase6b_heldout_retrieval.py --config configs/final_retrieval_selected.yaml --confirm-one-time-heldout`, then credentialed generation evaluation, then history-aware query rewriting, then Streamlit.
