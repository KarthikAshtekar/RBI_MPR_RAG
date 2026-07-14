# Technique-Wise Comparison

## Dense vs BM25 vs Hybrid

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Single-document Dense | 0.8333 | — | — | 0.6556 | — | completed |
| Single-document BM25 | 0.8333 | — | — | 0.6917 | — | completed |
| Single-document Hybrid RRF | 0.8000 | — | — | 0.7444 | — | completed |
| Single-document Hybrid RRF + reranker | 0.9333 | — | — | 0.8222 | — | completed |
| Hybrid BM25 + Dense retrieval with RRF | 0.3667 | 0.3333 | 0.4611 | 0.3389 | 1978.68 | completed |
| V2 baseline final retrieval | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 2947.24 | completed |
| V2 Cohere retrieval | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12906.78 | completed |

Dense and BM25 both helped in the single-document baseline. Hybrid RRF plus reranking was strongest in the April-only setting. For the temporal setting, dense and BM25 were used together; standalone temporal dense-only/BM25-only metrics were not found.

## RRF and Weighted RRF

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| RRF fusion k=10 | 0.3333 | 0.3000 | 0.4444 | 0.3417 | 4077.97 | completed |
| RRF fusion k=30 | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 4602.42 | completed |
| RRF fusion k=60 | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 4596.76 | completed |
| RRF fusion k=100 | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 4557.07 | completed |
| Weighted RRF D1/B1 reference | 0.3333 | 0.3000 | 0.4278 | 0.3361 | 4493.95 | completed |
| Weighted RRF dense weight 1.5 | 0.3667 | 0.3000 | 0.4389 | 0.3333 | 5236.62 | completed |
| Weighted RRF dense weight 2 | 0.3667 | 0.3000 | 0.4389 | 0.3333 | 4209.47 | completed |
| Weighted RRF BM25 weight 1.5 | 0.2667 | 0.2333 | 0.3111 | 0.2667 | 4146.06 | completed |
| Weighted RRF BM25 weight 2 | 0.2667 | 0.2333 | 0.3111 | 0.2667 | 4550.08 | completed |

The best saved RRF-k run by Macro MRR was RRF_K10, but Stage A selection favored the larger quota configuration because it improved Complete Evidence Recall and All-Reports Hit. Weighted RRF did not beat the selected quota configuration.

## Reranking

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Single-document Hybrid RRF + reranker | 0.9333 | — | — | 0.8222 | — | completed |
| Local cross-encoder reranking | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 3722.89 | completed |
| V2 baseline final retrieval | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 2947.24 | completed |
| V2 Cohere retrieval | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12906.78 | completed |

Local cross-encoder reranking was retained through the baseline. Cohere reranking improved development retrieval metrics but materially increased retrieval latency.

## MMR / diversity selection

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Exact-overlap diversity filter | 0.4000 | 0.3667 | 0.4944 | 0.3509 | 4285.51 | completed |
| MMR / diversity selection | — | — | — | — | — | evaluated_selected |
| MMR baseline: V2 Cohere selected context | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12906.78 | completed |
| True MMR lambda 0.6 | 0.5667 | 0.5333 | 0.6500 | 0.4055 | 15426.97 | completed |
| True MMR lambda 0.7 | 0.5333 | 0.5000 | 0.6333 | 0.4145 | 15347.95 | completed |
| True MMR lambda 0.8 | 0.5000 | 0.4667 | 0.6000 | 0.4164 | 15412.29 | completed |
| Adjacent expansion boundary | 0.3000 | 0.3000 | 0.4556 | 0.3026 | 3070.17 | completed |
| Adjacent expansion always | 0.3000 | 0.2667 | 0.4111 | 0.2755 | 3097.28 | completed |
| Adjacent expansion trend-only | 0.3667 | 0.3333 | 0.4611 | 0.3349 | 3017.50 | completed |
| Child-parent retrieval same-page parent | 0.4000 | 0.2000 | 0.3056 | 0.3537 | 3400.03 | completed |
| Child-parent retrieval adjacent-child parent | 0.4000 | 0.0667 | 0.1944 | 0.3537 | 3534.37 | completed |
| Child-parent retrieval page-bounded parent | 0.4000 | 0.1333 | 0.2611 | 0.3537 | 2955.55 | completed |
| Sentence-window retrieval window 1 | 0.4000 | 0.0667 | 0.1056 | 0.3537 | 2646.14 | completed |
| Sentence-window retrieval window 2 | 0.4000 | 0.1000 | 0.1833 | 0.3537 | 2809.83 | completed |
| MMR lambda 0.6 retrieval + sufficiency-gated generation | — | — | — | — | 25053.17 | completed |
| Final selected generation bake-off strategy | — | — | — | — | 25053.17 | completed_reused |
| Generation bake-off: GEN_MMR07_SUFFICIENCY_V1 | — | — | — | — | 28860.68 | completed |
| Generation bake-off: GEN_MMR08_SUFFICIENCY_V1 | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_CHRONO_ORDER_V1 | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_RERANK_ORDER_V1 | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1 | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1 | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_CITATION_REPAIR_V1 | — | — | — | — | 0.0000 | completed_repaired |

MMR means Maximal Marginal Relevance, a diversity-based retrieval/selection method. True MMR rows are shown when `reports/mmr_experiments` exists. DIV01 used exact-overlap diversity filtering and is not MMR.

## Multi-query / terminology expansion / facet decomposition

| Method | Hit/All-Hit | CER | Evidence Recall | MRR/Macro MRR | Median Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Terminology expansion append | 0.3667 | 0.3333 | 0.4778 | 0.3139 | 4007.51 | completed |
| Multi-query retrieval / terminology expansion | 0.3667 | 0.3333 | 0.4778 | 0.3139 | 4152.32 | completed |
| Facet decomposition | 0.3667 | 0.3333 | 0.4778 | 0.3139 | 4267.16 | completed |

Saved Stage A query-normalisation, terminology-expansion, multi-query, and facet experiments did not improve over the selected quota baseline.

## Sufficiency-gated generation

| Method | Factual | Faithfulness | Citation | Temporal | Comparative | Abstention | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2 Cohere retrieval + generation | 0.6344 | 0.8654 | 0.8824 | 0.8824 | 0.9167 | 0.4706 | completed |
| V2 Cohere retrieval + sufficiency-gated generation | 0.7954 | 0.9762 | 0.8824 | 0.8824 | 0.2778 | 1.0000 | completed |
| MMR lambda 0.6 retrieval + sufficiency-gated generation | 0.8153 | 0.9731 | 0.8824 | 0.8824 | 0.3333 | 1.0000 | completed |
| Final selected generation bake-off strategy | 0.8153 | 0.9731 | 0.8824 | 0.8824 | 0.3333 | 1.0000 | completed_reused |
| Generation bake-off: GEN_V2_COHERE_SUFFICIENCY_V1 | 0.7954 | 0.9762 | 0.8824 | 0.8824 | 0.2778 | 1.0000 | completed_reused |
| Generation bake-off: GEN_MMR07_SUFFICIENCY_V1 | 0.8041 | 0.9730 | 0.8824 | 0.8824 | 0.2778 | 1.0000 | completed |
| Generation bake-off: GEN_MMR08_SUFFICIENCY_V1 | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_CHRONO_ORDER_V1 | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_RERANK_ORDER_V1 | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1 | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1 | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_CITATION_REPAIR_V1 | 0.8153 | 0.9731 | 0.8824 | 0.8824 | 0.3333 | 1.0000 | completed_repaired |

Sufficiency gating improved factual correctness, faithfulness, and abstention correctness. Comparative correctness drops because incomplete comparative questions are now caveated or abstained instead of being treated as full answers.

## Poppler-enabled Unstructured update

- Poppler verification after setup: `True`.
- Selected V2 experiment: `V2_COHERE_ONLY`.
- Unstructured rows below are development-only and do not use held-out data.

### Overall V2 retrieval rows

| Method | CER | All-Hit | Evidence Recall | Macro MRR | Median ms | Status |
| --- | --- | --- | --- | --- | --- | --- |
| V2 baseline final retrieval | 0.3667 | 0.4000 | 0.5056 | 0.3537 | 2947.24 | completed |
| V2 Cohere retrieval | 0.4667 | 0.5000 | 0.6000 | 0.4154 | 12906.78 | completed |
| V2_UNSTRUCTURED_COHERE | — | — | — | — | — | not_run |
| V2_UNSTRUCTURED_ONLY | — | — | — | — | — | not_run |

### Table / numeric questions

| Experiment | CER | Evidence Recall | Macro MRR | Cases |
| --- | --- | --- | --- | --- |
| V2_BASELINE_FINAL | 0.2500 | 0.4083 | 0.2750 | 20 |
| V2_COHERE_ONLY | 0.4000 | 0.5500 | 0.3668 | 20 |
