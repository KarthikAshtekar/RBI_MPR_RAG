# Presentation Tables

## 1. Original single-document retrieval

| Method | Hit@4 | MRR | Mean latency |
| --- | --- | --- | --- |
| Single-document Dense | 0.8333 | 0.6556 | 30.67 |
| Single-document Dense + reranker | 0.9000 | 0.8278 | 1075.77 |
| Single-document BM25 | 0.8333 | 0.6917 | 2.1741 |
| Single-document BM25 + reranker | 0.8667 | 0.7667 | 785.48 |
| Single-document Hybrid RRF | 0.8000 | 0.7444 | 25.77 |
| Single-document Hybrid RRF + reranker | 0.9333 | 0.8222 | 1233.94 |

## 2. Multi-report final retrieval

| Method | All-Hit | CER | Evidence Recall | Macro MRR | Median latency |
| --- | --- | --- | --- | --- | --- |
| V2 baseline final retrieval | 0.4000 | 0.3667 | 0.5056 | 0.3537 | 2947.24 |
| V2 Cohere retrieval | 0.5000 | 0.4667 | 0.6000 | 0.4154 | 12906.78 |

## 3. V2 Cohere improvement

| Method | CER | All-Hit | Evidence Recall | Macro MRR | Mean tokens |
| --- | --- | --- | --- | --- | --- |
| V2 baseline final retrieval | 0.3667 | 0.4000 | 0.5056 | 0.3537 | 2150.93 |
| V2 Cohere retrieval | 0.4667 | 0.5000 | 0.6000 | 0.4154 | 2096.97 |

## 4. Generation before vs after sufficiency gate

| Method | Factual | Faithfulness | Abstention | Citation | Temporal | Comparative |
| --- | --- | --- | --- | --- | --- | --- |
| V2 Cohere retrieval + generation | 0.6344 | 0.8654 | 0.4706 | 0.8824 | 0.8824 | 0.9167 |
| V2 Cohere retrieval + sufficiency-gated generation | 0.7954 | 0.9762 | 1.0000 | 0.8824 | 0.8824 | 0.2778 |
| MMR lambda 0.6 retrieval + sufficiency-gated generation | 0.8153 | 0.9731 | 1.0000 | 0.8824 | 0.8824 | 0.3333 |
| Final selected generation bake-off strategy | 0.8153 | 0.9731 | 1.0000 | 0.8824 | 0.8824 | 0.3333 |

## 5. Technique contribution table

| Technique | Purpose | Observed Impact | Trade-off | Final Status |
|---|---|---|---|---|
| BM25 | keyword/numeric retrieval | Useful with dense hybrid | Lexical only | retained |
| Dense embeddings | semantic retrieval | Useful baseline and hybrid component | Can miss exact numeric evidence | retained |
| Hybrid search | combines lexical + semantic | Stronger than either alone in the final architecture | More complex | retained |
| RRF | rank fusion | Stabilised dense+BM25 candidate fusion | Requires k/retention tuning | retained |
| Reranking | improves ordering | Cohere improved dev metrics | Latency increase | retained in V2 |
| MMR | diversity control | True MMR evaluated over saved V2 Cohere outputs | May drop evidence if misapplied | evaluated_selected |
| Multi-query | expands query intent | EXP02 did not beat quota baseline | Added complexity | not selected |
| Sufficiency gate | prevents unsupported answers | Improved abstention/factuality | Lower answer coverage | retained |

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
