# RAG Methods Master Comparison

This table is generated from saved artifacts only. Missing metrics are shown as `—` and are stored as JSON null.

Single-document Hit-Rate@4 is not directly comparable to multi-report Complete Evidence Recall because the temporal task requires evidence from multiple report periods and stricter report attribution.

MRR = Mean Reciprocal Rank, a ranking metric. MMR = Maximal Marginal Relevance, a diversity-aware selection technique.

Old Phase 7 held-out results are historical and should not be presented as a fresh V2 benchmark.

| Method | Scope | Split | Hit / All-Reports Hit | CER | Evidence Recall | MRR / Macro MRR | Factual Correctness | Citation Correctness | Temporal Attribution | Median Latency | Mean Tokens | Coverage | Contamination | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-document Dense | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8333 | — | — | 0.6556 | — | — | — | — | — | — | — | completed |
| Single-document Dense + reranker | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.9000 | — | — | 0.8278 | — | — | — | — | — | — | — | completed |
| Single-document BM25 | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8333 | — | — | 0.6917 | — | — | — | — | — | — | — | completed |
| Single-document BM25 + reranker | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8667 | — | — | 0.7667 | — | — | — | — | — | — | — | completed |
| Single-document Hybrid RRF | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8000 | — | — | 0.7444 | — | — | — | — | — | — | — | completed |
| Single-document Hybrid RRF + reranker | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.9333 | — | — | 0.8222 | — | — | — | — | — | — | — | completed |
| Multi-report naive global retrieval | Three-report temporal retrieval | dev | 0.1667 | 0.1000 | 0.2111 | 0.2417 | — | — | — | 920.64 | — | 0.9389 | 0.9167 | completed |
| Multi-report report-aware retrieval | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4611 | 0.3389 | — | — | — | 1601.99 | — | 1.0000 | 0.0000 | completed |
| Dense-only temporal retrieval | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | implemented_not_independently_evaluated |
| BM25-only temporal retrieval | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | implemented_not_independently_evaluated |
| Hybrid BM25 + Dense retrieval with RRF | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4611 | 0.3389 | — | — | — | 1333.17 | 1293.33 | 1.0000 | 0.0000 | completed |
| RRF fusion k=10 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4444 | 0.3417 | — | — | — | 2884.28 | 1289.83 | 1.0000 | 0.0000 | completed |
| RRF fusion k=30 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 2847.66 | 1294.07 | 1.0000 | 0.0000 | completed |
| RRF fusion k=60 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 3145.51 | 1294.07 | 1.0000 | 0.0000 | completed |
| RRF fusion k=100 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 2800.32 | 1294.07 | 1.0000 | 0.0000 | completed |
| Weighted RRF D1/B1 reference | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 2914.22 | 1294.07 | 1.0000 | 0.0000 | completed |
| Weighted RRF dense weight 1.5 | Three-report temporal retrieval | dev | 0.3667 | 0.3000 | 0.4389 | 0.3333 | — | — | — | 3027.17 | 1290.80 | 1.0000 | 0.0000 | completed |
| Weighted RRF dense weight 2 | Three-report temporal retrieval | dev | 0.3667 | 0.3000 | 0.4389 | 0.3333 | — | — | — | 3200.68 | 1290.80 | 1.0000 | 0.0000 | completed |
| Weighted RRF BM25 weight 1.5 | Three-report temporal retrieval | dev | 0.2667 | 0.2333 | 0.3111 | 0.2667 | — | — | — | 2462.29 | 1302.07 | 1.0000 | 0.0000 | completed |
| Weighted RRF BM25 weight 2 | Three-report temporal retrieval | dev | 0.2667 | 0.2333 | 0.3111 | 0.2667 | — | — | — | 2782.74 | 1302.07 | 1.0000 | 0.0000 | completed |
| Final local cross-encoder retrieval baseline | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 3159.04 | 2150.93 | 1.0000 | 0.0000 | completed |
| Local cross-encoder reranking | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 3096.76 | 2150.93 | 1.0000 | 0.0000 | completed |
| Candidate-pool/quota optimised retrieval | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 3729.85 | 2150.93 | 1.0000 | 0.0000 | completed |
| Terminology expansion append | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.4889 | 0.3161 | — | — | — | 2766.17 | 2156.43 | 1.0000 | 0.0000 | completed |
| Multi-query retrieval / terminology expansion | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.4889 | 0.3161 | — | — | — | 2757.75 | 2156.43 | 1.0000 | 0.0000 | completed |
| Facet decomposition | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.4889 | 0.3161 | — | — | — | 2837.39 | 2156.43 | 1.0000 | 0.0000 | completed |
| Exact-overlap diversity filter | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 3010.57 | 2150.93 | 1.0000 | 0.0000 | completed |
| MMR / diversity selection | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | evaluated_selected |
| MMR baseline: V2 Cohere selected context | Three-report temporal retrieval | dev | 0.5000 | 0.4667 | 0.6000 | 0.4154 | — | — | — | 12948.27 | 2096.40 | 1.0000 | 0.0000 | completed |
| True MMR lambda 0.6 | Three-report temporal retrieval | dev | 0.5667 | 0.5333 | 0.6500 | 0.4055 | — | — | — | 15217.14 | 2037.03 | 1.0000 | 0.0000 | completed |
| True MMR lambda 0.7 | Three-report temporal retrieval | dev | 0.5333 | 0.5000 | 0.6333 | 0.4145 | — | — | — | 15279.26 | 2048.40 | 1.0000 | 0.0000 | completed |
| True MMR lambda 0.8 | Three-report temporal retrieval | dev | 0.5000 | 0.4667 | 0.6000 | 0.4164 | — | — | — | 15228.44 | 2047.30 | 1.0000 | 0.0000 | completed |
| Adjacent expansion boundary | Three-report temporal retrieval | dev | 0.3000 | 0.3000 | 0.4556 | 0.3026 | — | — | — | 2952.91 | 2465.57 | 0.9000 | 0.0000 | completed |
| Adjacent expansion always | Three-report temporal retrieval | dev | 0.3000 | 0.2667 | 0.4111 | 0.2755 | — | — | — | 2918.06 | 2480.33 | 0.8333 | 0.0000 | completed |
| Adjacent expansion trend-only | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4611 | 0.3349 | — | — | — | 2982.26 | 2034.77 | 0.9333 | 0.0000 | completed |
| Child-parent retrieval same-page parent | Three-report temporal retrieval | dev | 0.4000 | 0.2000 | 0.3056 | 0.3537 | — | — | — | 3064.40 | 2189.67 | 1.0000 | 0.0000 | completed |
| Child-parent retrieval adjacent-child parent | Three-report temporal retrieval | dev | 0.4000 | 0.0667 | 0.1944 | 0.3537 | — | — | — | 3120.68 | 1999.83 | 1.0000 | 0.0000 | completed |
| Child-parent retrieval page-bounded parent | Three-report temporal retrieval | dev | 0.4000 | 0.1333 | 0.2611 | 0.3537 | — | — | — | 3167.88 | 2191.13 | 1.0000 | 0.0000 | completed |
| Sentence-window retrieval window 1 | Three-report temporal retrieval | dev | 0.4000 | 0.0667 | 0.1056 | 0.3537 | — | — | — | 3036.21 | 265.67 | 1.0000 | 0.0000 | completed |
| Sentence-window retrieval window 2 | Three-report temporal retrieval | dev | 0.4000 | 0.1000 | 0.1833 | 0.3537 | — | — | — | 3514.75 | 430.77 | 1.0000 | 0.0000 | completed |
| Chunk-window retrieval neighbour chunks | Three-report temporal retrieval | dev | 0.4000 | 0.2000 | 0.3056 | 0.3537 | — | — | — | 3280.24 | 2474.97 | 1.0000 | 0.0000 | completed |
| Phase 7 selected final retrieval held-out diagnostic | Three-report temporal retrieval | phase7_heldout_reused_for_old_final_only | 0.3846 | 0.3846 | 0.5513 | 0.2590 | — | — | — | 2495.77 | 2134.85 | 1.0000 | 0.0000 | completed_historical_heldout |
| V2 baseline final retrieval | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 3159.04 | 2150.93 | 1.0000 | 0.0000 | completed |
| V2 Cohere retrieval | Three-report temporal retrieval | dev | 0.5000 | 0.4667 | 0.6000 | 0.4154 | — | — | — | 12948.27 | 2097.20 | 1.0000 | 0.0000 | completed |
| V2_UNSTRUCTURED_COHERE | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | not_run |
| V2_UNSTRUCTURED_ONLY | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | not_run |
| V2 Cohere retrieval + generation | Three-report temporal generation | dev | — | — | — | — | 0.6344 | 0.8824 | 0.8824 | 26759.08 | — | — | — | completed |
| V2 Cohere retrieval + sufficiency-gated generation | Three-report temporal generation | dev | — | — | — | — | 0.7954 | 0.8824 | 0.8824 | 25694.34 | — | — | — | completed |
| MMR lambda 0.6 retrieval + sufficiency-gated generation | Three-report temporal generation | dev | — | — | — | — | 0.8153 | 0.8824 | 0.8824 | 25053.17 | — | — | — | completed |
| Final selected generation bake-off strategy | Three-report temporal generation | dev | — | — | — | — | 0.8153 | 0.8824 | 0.8824 | 25053.17 | 1797.38 | — | — | completed_reused |
| Generation bake-off: GEN_V2_COHERE_SUFFICIENCY_V1 | Three-report temporal generation | dev | — | — | — | — | 0.7954 | 0.8824 | 0.8824 | 25694.34 | 1998.26 | — | — | completed_reused |
| Generation bake-off: GEN_MMR07_SUFFICIENCY_V1 | Three-report temporal generation | dev | — | — | — | — | 0.8041 | 0.8824 | 0.8824 | 28860.68 | 1807.41 | — | — | completed |
| Generation bake-off: GEN_MMR08_SUFFICIENCY_V1 | Three-report temporal generation | dev | — | — | — | — | — | — | — | — | 1806.38 | — | — | skipped |
| Generation bake-off: GEN_MMR06_CHRONO_ORDER_V1 | Three-report temporal generation | dev | — | — | — | — | — | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_RERANK_ORDER_V1 | Three-report temporal generation | dev | — | — | — | — | — | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1 | Three-report temporal generation | dev | — | — | — | — | — | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1 | Three-report temporal generation | dev | — | — | — | — | — | — | — | — | — | — | — | skipped |
| Generation bake-off: GEN_MMR06_CITATION_REPAIR_V1 | Three-report temporal generation | dev | — | — | — | — | 0.8153 | 0.8824 | 0.8824 | 0.0000 | 1797.38 | — | — | completed_repaired |
