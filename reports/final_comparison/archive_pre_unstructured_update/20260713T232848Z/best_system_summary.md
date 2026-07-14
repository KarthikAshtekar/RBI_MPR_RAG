# Best Current System Summary

Best current system: **V2 Cohere Retrieval + Sufficiency-Gated Generation**.

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
- Median latency: 12906.78 ms
- Mean tokens: 2096.97

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
- Unstructured extraction remains blocked by missing Poppler.
- Cohere improves quality but increases latency.
