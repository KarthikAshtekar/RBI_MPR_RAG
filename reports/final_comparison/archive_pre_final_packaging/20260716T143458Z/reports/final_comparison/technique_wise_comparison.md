# Technique-Wise Comparison

## Dense vs BM25 vs Hybrid

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Single-document Dense | 0.8333 | — | — | 0.6556 | — | completed |
| Single-document BM25 | 0.8333 | — | — | 0.6917 | — | completed |
| Single-document Hybrid RRF | 0.8000 | — | — | 0.7444 | — | completed |
| Single-document Hybrid RRF + reranker | 0.9333 | — | — | 0.8222 | — | completed |
| Hybrid BM25 + Dense retrieval with RRF | 0.3667 | 0.3333 | 0.4611 | 0.3389 | 1333.17 | completed |
| V2 baseline final retrieval | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 3159.04 | completed |
| V2 Cohere retrieval | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12948.27 | completed |

Dense and BM25 both helped in the single-document baseline. Hybrid RRF plus reranking was strongest in the April-only setting. For the temporal setting, dense and BM25 were used together; standalone temporal dense-only/BM25-only metrics were not found.

## RRF and Weighted RRF

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| RRF fusion k=10 | 0.3333 | 0.3000 | 0.4444 | 0.3417 | 2884.28 | completed |
| RRF fusion k=30 | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 2847.66 | completed |
| RRF fusion k=60 | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 3145.51 | completed |
| RRF fusion k=100 | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 2800.32 | completed |
| Weighted RRF D1/B1 reference | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 2914.22 | completed |
| Weighted RRF dense weight 1.5 | 0.3667 | 0.3000 | 0.4389 | 0.3333 | 3027.17 | completed |
| Weighted RRF dense weight 2 | 0.3667 | 0.3000 | 0.4389 | 0.3333 | 3200.68 | completed |
| Weighted RRF BM25 weight 1.5 | 0.2667 | 0.2333 | 0.3111 | 0.2667 | 2462.29 | completed |
| Weighted RRF BM25 weight 2 | 0.2667 | 0.2333 | 0.3111 | 0.2667 | 2782.74 | completed |

The best saved RRF-k run by Macro MRR was RRF_K10, but Stage A selection favored the larger quota configuration because it improved Complete Evidence Recall and All-Reports Hit. Weighted RRF did not beat the selected quota configuration.

## Reranking

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Single-document Hybrid RRF + reranker | 0.9333 | — | — | 0.8222 | — | completed |
| Local cross-encoder reranking | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 3096.76 | completed |
| V2 baseline final retrieval | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 3159.04 | completed |
| V2 Cohere retrieval | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12948.27 | completed |

Local cross-encoder reranking was retained through the baseline. Cohere reranking improved development retrieval metrics but materially increased retrieval latency.

## MMR / diversity selection

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Exact-overlap diversity filter | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 3010.57 | completed |
| MMR / diversity selection | — | — | — | — | — | evaluated_selected |
| MMR baseline: V2 Cohere selected context | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12948.27 | completed |
| True MMR lambda 0.6 | 0.5667 | 0.5333 | 0.6500 | 0.4055 | 15217.14 | completed |
| True MMR lambda 0.7 | 0.5333 | 0.5000 | 0.6333 | 0.4145 | 15279.26 | completed |
| True MMR lambda 0.8 | 0.5000 | 0.4667 | 0.6000 | 0.4164 | 15228.44 | completed |
| Adjacent expansion boundary | 0.3000 | 0.3000 | 0.4556 | 0.3026 | 2952.91 | completed |
| Adjacent expansion always | 0.3000 | 0.2667 | 0.4111 | 0.2755 | 2918.06 | completed |
| Adjacent expansion trend-only | 0.3667 | 0.3333 | 0.4611 | 0.3349 | 2982.26 | completed |
| Child-parent retrieval same-page parent | 0.4000 | 0.2000 | 0.3056 | 0.3537 | 3064.40 | completed |
| Child-parent retrieval adjacent-child parent | 0.4000 | 0.0667 | 0.1944 | 0.3537 | 3120.68 | completed |
| Child-parent retrieval page-bounded parent | 0.4000 | 0.1333 | 0.2611 | 0.3537 | 3167.88 | completed |
| Sentence-window retrieval window 1 | 0.4000 | 0.0667 | 0.1056 | 0.3537 | 3036.21 | completed |
| Sentence-window retrieval window 2 | 0.4000 | 0.1000 | 0.1833 | 0.3537 | 3514.75 | completed |

MMR means Maximal Marginal Relevance, a diversity-based retrieval/selection method. True MMR rows are shown when `reports/mmr_experiments` exists. DIV01 used exact-overlap diversity filtering and is not MMR.

## Multi-query / terminology expansion / facet decomposition

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Terminology expansion append | 0.4000 | 0.3667 | 0.4889 | 0.3161 | 2766.17 | completed |
| Multi-query retrieval / terminology expansion | 0.4000 | 0.3667 | 0.4889 | 0.3161 | 2757.75 | completed |
| Facet decomposition | 0.4000 | 0.3667 | 0.4889 | 0.3161 | 2837.39 | completed |

Saved Stage A query-normalisation, terminology-expansion, multi-query, and facet experiments did not improve over the selected quota baseline.

## Sufficiency-gated generation

| Method | Factual | Faithfulness | Citation | Temporal | Comparative | Abstention | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2 Cohere retrieval + generation | 0.6344 | 0.8657 | 0.8824 | 0.8824 | 0.9167 | 0.4706 | completed |
| V2 Cohere retrieval + sufficiency-gated generation | 0.7954 | 0.9762 | 0.8824 | 0.8824 | 0.2778 | 1.0000 | completed |

Sufficiency gating improved factual correctness, faithfulness, and abstention correctness. Comparative correctness drops because incomplete comparative questions are now caveated or abstained instead of being treated as full answers.

## Poppler-enabled Unstructured update

- Poppler verification after setup: `True`.
- Selected V2 experiment: `V2_COHERE_ONLY`.
- Unstructured rows below are development-only and do not use held-out data.

### Overall V2 retrieval rows

| Method | CER | All-Hit | Evidence Recall | Macro MRR | Median ms | Status |
| --- | --- | --- | --- | --- | --- | --- |
| V2 baseline final retrieval | 0.3667 | 0.4000 | 0.5056 | 0.3537 | 3159.04 | completed |
| V2 Cohere retrieval | 0.4667 | 0.5000 | 0.6000 | 0.4154 | 12948.27 | completed |
| V2_UNSTRUCTURED_COHERE | — | — | — | — | — | not_run |
| V2_UNSTRUCTURED_ONLY | — | — | — | — | — | not_run |

### Table / numeric questions

| Experiment | CER | Evidence Recall | Macro MRR | Cases |
| --- | --- | --- | --- | --- |
| V2_BASELINE_FINAL | 0.2500 | 0.4083 | 0.2750 | 20 |
| V2_COHERE_ONLY | 0.4000 | 0.5500 | 0.3668 | 20 |
