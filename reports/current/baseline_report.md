# RBI April 2025 RAG Baseline Report

- Run timestamp: 2026-07-12T21:08:30.562968+00:00
- Repository commit: not available (not a Git repository)
- Configuration: `configs/baseline.yaml`
- Dataset version: `rbi_mpr_april_2025_dev_v1`
- PDF checksum: `6b505611e2232e6c8ce9af07018dc9ddba9f46d3efdf3f6ac3a2d22ef9b74752`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator: `llama-3.1-8b-instant`
- Judge: `llama-3.1-8b-instant`

## Retrieval comparison

| Pipeline | Hit-Rate@4 | MRR | Mean latency (ms) |
|---|---:|---:|---:|
| dense | 83.33% | 0.656 | 30.7 |
| dense_reranked | 90.00% | 0.828 | 1075.8 |
| bm25 | 83.33% | 0.692 | 2.2 |
| bm25_reranked | 86.67% | 0.767 | 785.5 |
| hybrid_rrf | 80.00% | 0.744 | 25.8 |
| hybrid_reranked | 93.33% | 0.822 | 1233.9 |

Reranking cannot recover a relevant chunk absent from its initial candidate pool. It can promote a relevant chunk from ranks 5–15 into the final top four, so it may improve both MRR and Hit-Rate@4.

## Generation evaluation

Not executed or not yet validated. No generation metric is reported without coverage.

## Result status

The retrieval table is validated from the current saved raw run. Generation remains unavailable until a credentialed evaluation completes with explicitly reported coverage. Metrics in archived artifacts are historical and are not carried into this report.

## Settings

- Chunking: 1000 characters, 300 overlap
- Candidate pools: dense=15, BM25=15, reranker=15
- RRF constant: 60; final context: top 4

## Known limitations

- The 30 questions are a development set, not a held-out test set.
- Relevance is judged at page level, not chunk level.
- Hosted LLM generation is not guaranteed byte-identical even at temperature zero.
- Archived PDF and preliminary metrics are not validated current results.
