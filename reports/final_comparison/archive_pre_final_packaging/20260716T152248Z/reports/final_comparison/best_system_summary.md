# Best Current System Summary

Best evaluated generation system: **V2 Cohere retrieval + sufficiency-gated generation**.

## Architecture

- PyPDFLoader extraction
- Dense vector retrieval
- BM25 retrieval
- Hybrid retrieval using RRF
- Cohere `rerank-v3.5`
- Report-aware context quotas
- Source-labelled contexts
- Groq `llama-3.1-8b-instant` generation
- Evidence sufficiency gate
- Citation validation
- Temporal attribution validation

## Final development retrieval metrics

- CER: 0.4667
- All-Reports Hit: 0.5000
- Evidence Recall: 0.6000
- Macro MRR: 0.4154
- Median latency: 12948.27 ms
- Mean tokens: 2097.20

## Final development generation metrics after sufficiency gate

- Factual correctness: 0.7954
- Faithfulness to context: 0.9762
- Abstention correctness: 1.0000
- Citation correctness: 0.8824
- Citation completeness: 1.0000
- Temporal attribution correctness: 0.8824
- Comparative correctness: 0.2778

## Caveats

- Development-only final V2 generation.
- Old held-out set was not reused as a fresh V2 benchmark.
- Metrics are deterministic heuristics, not human evaluation.
- Poppler is now configured for the project, but Unstructured extraction remains blocked for the current PDFs because non-OCR extraction returned zero elements and OCR/Tesseract is unavailable.
- Cohere improves quality but increases latency.

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
