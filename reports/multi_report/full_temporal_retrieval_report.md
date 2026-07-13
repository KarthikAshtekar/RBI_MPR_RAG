# Full Temporal Retrieval Baseline

Temporal multi-document RAG for RBI Monetary Policy Reports

## Corpus manifest

| Report | SHA-256 | Pages | Chunks | Status |
|---|---|---:|---:|---|
| April 2025 | `6b505611e2232e6c8ce9af07018dc9ddba9f46d3efdf3f6ac3a2d22ef9b74752` | 116 | 461 | reused |
| October 2025 | `4d7a955dd8bbb74fbd85b3c9e3085776dc7fbb4cbc70da2947352a0112b6022b` | 112 | 463 | reused |
| April 2026 | `dbcdb6edc969f198a67044c5eb4297b13c9ad3f691928e9e0661ec751cfce32c` | 118 | 481 | reused |

## Extraction audit

Audited pages: 45; pages requiring manual numeric verification: 19.

## Router evaluation

Development: 1.0 (30 cases)
Held-out: 1.0 (20 cases)

## Dataset freeze

Development cases: 43 total; 34 newly verified/scored.
Held-out cases: 15 (frozen checksum `183b7b3187cd7e53df343e0461e46311192a9ffc0bfdf49a2bc32ad64488af99`).

## Report-aware retrieval

| Split | Report coverage | All-reports hit | Macro report MRR | Evidence recall | Complete evidence recall | Contamination |
|---|---|---|---|---|---|---|
| Development | 1.0 [1.0, 1.0] (n=30) | 0.36666666666666664 [0.21873920806100713, 0.5448643634520512] (n=30) | 0.33888888888888885 [0.21944444444444444, 0.46388888888888885] (n=30) | 0.46111111111111114 [0.3, 0.6222222222222222] (n=30) | 0.3333333333333333 [0.19230498083676142, 0.5121994835545616] (n=30) | 0.0 [0.0, 0.2424940066552408] (n=12) |
| Held-out | 1.0 [1.0, 1.0] (n=13) | 0.38461538461538464 [0.17709707797762575, 0.644771084873343] (n=13) | 0.28846153846153844 [0.15384615384615385, 0.41025641025641024] (n=13) | 0.4743589743589744 [0.24358974358974358, 0.6923076923076923] (n=13) | 0.3076923076923077 [0.12680703655710515, 0.5763065681945096] (n=13) | 0.0 [0.0, 0.39033428790216523] (n=6) |

## Architecture comparison

Report-aware retrieval preserves report coverage and eliminates single-report contamination; naive results and confidence intervals are saved in `architecture_comparison.json`. Overlapping intervals mean small differences should not be treated as conclusive.

## Development failure analysis

See `dev_failure_analysis.md`; held-out failures were not inspected or used for tuning.

## Generation evaluation

Not executed because GROQ_API_KEY is unavailable. Validated generation coverage is zero.

## Frozen configuration

Dense/BM25 per report: 15/15; RRF k=60; reranker candidates=15; single quota=4; pairwise quota=3/report; trend quota=2/report; bootstrap=2000; seed=42.

## Limitations and next phase

PyPDFLoader flattens tables/charts; numeric cases require manual verification. This is not production-ready. Next: run credentialed comparative generation evaluation, then add conversational memory only after generation quality is validated.
