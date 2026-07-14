# RAG Methods Master Comparison

This table is generated from saved artifacts only. Missing metrics are shown as `—` and are stored as JSON null.

Single-document Hit-Rate@4 is not directly comparable to multi-report Complete Evidence Recall because the temporal task requires evidence from multiple report periods and stricter report attribution.

| Method | Scope | Split | Hit / All-Reports Hit | CER | Evidence Recall | MRR / Macro MRR | Factual Correctness | Citation Correctness | Temporal Attribution | Median Latency | Mean Tokens | Coverage | Contamination | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-document Dense | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8333 | — | — | 0.6556 | — | — | — | — | — | — | — | completed |
| Single-document Dense + reranker | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.9000 | — | — | 0.8278 | — | — | — | — | — | — | — | completed |
| Single-document BM25 | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8333 | — | — | 0.6917 | — | — | — | — | — | — | — | completed |
| Single-document BM25 + reranker | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8667 | — | — | 0.7667 | — | — | — | — | — | — | — | completed |
| Single-document Hybrid RRF | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.8000 | — | — | 0.7444 | — | — | — | — | — | — | — | completed |
| Single-document Hybrid RRF + reranker | April 2025 only | rbi_mpr_april_2025_dev_v1 | 0.9333 | — | — | 0.8222 | — | — | — | — | — | — | — | completed |
| Multi-report naive global retrieval | Three-report temporal retrieval | dev | 0.1667 | 0.1000 | 0.2111 | 0.2417 | — | — | — | 1052.61 | — | 0.9389 | 0.9167 | completed |
| Multi-report report-aware retrieval | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4611 | 0.3389 | — | — | — | 1817.99 | — | 1.0000 | 0.0000 | completed |
| Dense-only temporal retrieval | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | implemented_not_independently_evaluated |
| BM25-only temporal retrieval | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | implemented_not_independently_evaluated |
| Hybrid BM25 + Dense retrieval with RRF | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4611 | 0.3389 | — | — | — | 1978.68 | 1293.33 | 1.0000 | 0.0000 | completed |
| RRF fusion k=10 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4444 | 0.3417 | — | — | — | 4077.97 | 1289.83 | 1.0000 | 0.0000 | completed |
| RRF fusion k=30 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 4602.42 | 1294.07 | 1.0000 | 0.0000 | completed |
| RRF fusion k=60 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 4596.76 | 1294.07 | 1.0000 | 0.0000 | completed |
| RRF fusion k=100 | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 4557.07 | 1294.07 | 1.0000 | 0.0000 | completed |
| Weighted RRF D1/B1 reference | Three-report temporal retrieval | dev | 0.3333 | 0.3000 | 0.4278 | 0.3361 | — | — | — | 4493.95 | 1294.07 | 1.0000 | 0.0000 | completed |
| Weighted RRF dense weight 1.5 | Three-report temporal retrieval | dev | 0.3667 | 0.3000 | 0.4389 | 0.3333 | — | — | — | 5236.62 | 1290.80 | 1.0000 | 0.0000 | completed |
| Weighted RRF dense weight 2 | Three-report temporal retrieval | dev | 0.3667 | 0.3000 | 0.4389 | 0.3333 | — | — | — | 4209.47 | 1290.80 | 1.0000 | 0.0000 | completed |
| Weighted RRF BM25 weight 1.5 | Three-report temporal retrieval | dev | 0.2667 | 0.2333 | 0.3111 | 0.2667 | — | — | — | 4146.06 | 1302.07 | 1.0000 | 0.0000 | completed |
| Weighted RRF BM25 weight 2 | Three-report temporal retrieval | dev | 0.2667 | 0.2333 | 0.3111 | 0.2667 | — | — | — | 4550.08 | 1302.07 | 1.0000 | 0.0000 | completed |
| Final local cross-encoder retrieval baseline | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 2947.24 | 2150.93 | 1.0000 | 0.0000 | completed |
| Local cross-encoder reranking | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 3722.89 | 2150.93 | 1.0000 | 0.0000 | completed |
| Candidate-pool/quota optimised retrieval | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 4855.38 | 2150.93 | 1.0000 | 0.0000 | completed |
| Terminology expansion append | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4778 | 0.3139 | — | — | — | 4007.51 | 1815.40 | 1.0000 | 0.0000 | completed |
| Multi-query retrieval / terminology expansion | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4778 | 0.3139 | — | — | — | 4152.32 | 1815.40 | 1.0000 | 0.0000 | completed |
| Facet decomposition | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4778 | 0.3139 | — | — | — | 4267.16 | 1815.40 | 1.0000 | 0.0000 | completed |
| Exact-overlap diversity filter | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.4944 | 0.3509 | — | — | — | 4285.51 | 1812.33 | 1.0000 | 0.0000 | completed |
| MMR / diversity selection | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | evaluated_selected |
| MMR baseline: V2 Cohere selected context | Three-report temporal retrieval | dev | 0.5000 | 0.4667 | 0.6000 | 0.4154 | — | — | — | 12906.78 | 2096.20 | 1.0000 | 0.0000 | completed |
| True MMR lambda 0.6 | Three-report temporal retrieval | dev | 0.5667 | 0.5333 | 0.6500 | 0.4055 | — | — | — | 15426.97 | 2037.03 | 1.0000 | 0.0000 | completed |
| True MMR lambda 0.7 | Three-report temporal retrieval | dev | 0.5333 | 0.5000 | 0.6333 | 0.4145 | — | — | — | 15347.95 | 2048.40 | 1.0000 | 0.0000 | completed |
| True MMR lambda 0.8 | Three-report temporal retrieval | dev | 0.5000 | 0.4667 | 0.6000 | 0.4164 | — | — | — | 15412.29 | 2047.23 | 1.0000 | 0.0000 | completed |
| Adjacent expansion boundary | Three-report temporal retrieval | dev | 0.3000 | 0.3000 | 0.4556 | 0.3026 | — | — | — | 3070.17 | 2465.57 | 0.9000 | 0.0000 | completed |
| Adjacent expansion always | Three-report temporal retrieval | dev | 0.3000 | 0.2667 | 0.4111 | 0.2755 | — | — | — | 3097.28 | 2480.33 | 0.8333 | 0.0000 | completed |
| Adjacent expansion trend-only | Three-report temporal retrieval | dev | 0.3667 | 0.3333 | 0.4611 | 0.3349 | — | — | — | 3017.50 | 2034.77 | 0.9333 | 0.0000 | completed |
| Child-parent retrieval same-page parent | Three-report temporal retrieval | dev | 0.4000 | 0.2000 | 0.3056 | 0.3537 | — | — | — | 3400.03 | 2189.67 | 1.0000 | 0.0000 | completed |
| Child-parent retrieval adjacent-child parent | Three-report temporal retrieval | dev | 0.4000 | 0.0667 | 0.1944 | 0.3537 | — | — | — | 3534.37 | 1999.83 | 1.0000 | 0.0000 | completed |
| Child-parent retrieval page-bounded parent | Three-report temporal retrieval | dev | 0.4000 | 0.1333 | 0.2611 | 0.3537 | — | — | — | 2955.55 | 2191.13 | 1.0000 | 0.0000 | completed |
| Sentence-window retrieval window 1 | Three-report temporal retrieval | dev | 0.4000 | 0.0667 | 0.1056 | 0.3537 | — | — | — | 2646.14 | 265.67 | 1.0000 | 0.0000 | completed |
| Sentence-window retrieval window 2 | Three-report temporal retrieval | dev | 0.4000 | 0.1000 | 0.1833 | 0.3537 | — | — | — | 2809.83 | 430.77 | 1.0000 | 0.0000 | completed |
| Chunk-window retrieval neighbour chunks | Three-report temporal retrieval | dev | 0.4000 | 0.2000 | 0.3056 | 0.3537 | — | — | — | 3450.68 | 2474.97 | 1.0000 | 0.0000 | completed |
| Phase 7 selected final retrieval held-out diagnostic | Three-report temporal retrieval | phase7_heldout_reused_for_old_final_only | 0.3846 | 0.3846 | 0.5513 | 0.2590 | — | — | — | 2495.77 | 2134.85 | 1.0000 | 0.0000 | completed_historical_heldout |
| V2 baseline final retrieval | Three-report temporal retrieval | dev | 0.4000 | 0.3667 | 0.5056 | 0.3537 | — | — | — | 2947.24 | 2150.93 | 1.0000 | 0.0000 | completed |
| V2 Cohere retrieval | Three-report temporal retrieval | dev | 0.5000 | 0.4667 | 0.6000 | 0.4154 | — | — | — | 12906.78 | 2096.97 | 1.0000 | 0.0000 | completed |
| V2_UNSTRUCTURED_COHERE | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | not_run |
| V2_UNSTRUCTURED_ONLY | Three-report temporal retrieval | dev | — | — | — | — | — | — | — | — | — | — | — | not_run |
| V2 Cohere retrieval + generation | Three-report temporal generation | dev | — | — | — | — | 0.6344 | 0.8824 | 0.8824 | 26759.08 | — | — | — | completed |
| V2 Cohere retrieval + sufficiency-gated generation | Three-report temporal generation | dev | — | — | — | — | 0.7954 | 0.8824 | 0.8824 | 25694.34 | — | — | — | completed |
