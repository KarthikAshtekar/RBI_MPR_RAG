# Final Project Report

## Executive summary

This project implements **Temporal multi-document RAG for RBI Monetary Policy Reports**. The final development system retrieves and answers questions about policy stance and narrative evolution across April 2025, October 2025, and April 2026 RBI Monetary Policy Reports.

Best current system: **V2 Cohere Retrieval + Sufficiency-Gated Generation**. It is ready for demonstration and interview discussion with known limitations; it is not production-ready.

## Problem statement

The system answers monetary-policy questions that may require evidence from one report, two reports, or all three reports while preserving correct report attribution.

## Why RBI Monetary Policy Reports

The reports are dense, periodic, and evidence-rich. They contain changes in inflation, growth, risks, and policy stance that are better framed as temporal retrieval than generic sentiment analysis.

## Dataset

- April 2025 Monetary Policy Report
- October 2025 Monetary Policy Report
- April 2026 Monetary Policy Report

## Original single-document RAG baseline

The project began with an April 2025-only RAG baseline using PyPDFLoader, chunking, MiniLM embeddings, BM25, RRF, local cross-encoder reranking, and Groq generation. Those Hit-Rate@4 and MRR metrics are preserved separately because they are not directly comparable to multi-report Complete Evidence Recall.

## Why multi-document temporal RAG is harder

Multi-report questions require retrieving complete evidence from the right report periods, avoiding wrong-report contamination, and supporting comparisons across changing narratives.

## Final architecture

- PyPDFLoader extraction
- `sentence-transformers/all-MiniLM-L6-v2` dense embeddings
- BM25 lexical retrieval
- Hybrid RRF candidate fusion
- Cohere `rerank-v3.5` reranking
- Report-aware routing and quotas
- Groq `llama-3.1-8b-instant` generation
- Sufficiency gate before final answer acceptance
- Source-labelled citations and temporal attribution checks

## Techniques evaluated

Dense retrieval, BM25, hybrid search, RRF, weighted RRF, multi-query/terminology expansion, facet decomposition, true MMR, Cohere reranking, Unstructured extraction attempt, and sufficiency-gated generation.

## Final retrieval metrics

| Method | CER | All-Reports Hit | Evidence Recall | Macro MRR | Median latency ms | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| V2 Cohere retrieval | 0.4667 | 0.5000 | 0.6000 | 0.4154 | 12906.78 | 2096.97 |

## Final generation metrics

| Metric | Score |
|---|---:|
| Factual correctness | 0.7954 |
| Faithfulness to context | 0.9762 |
| Abstention correctness | 1.0000 |
| Citation correctness | 0.8824 |
| Citation completeness | 1.0000 |
| Temporal attribution correctness | 0.8824 |
| Comparative correctness | 0.2778 |

## MMR result

MMR decision status: `evaluated_selected`. Selected experiment: `MMR_LAMBDA_06`.

| Experiment | CER | All-Reports Hit | Evidence Recall | Macro MRR | Repeated text ratio |
|---|---:|---:|---:|---:|---:|
| MMR_LAMBDA_06 | 0.5333 | 0.5667 | 0.6500 | 0.4055 | 0.0000 |
| MMR_LAMBDA_07 | 0.5000 | 0.5333 | 0.6333 | 0.4145 | 0.0000 |
| MMR_LAMBDA_08 | 0.4667 | 0.5000 | 0.6000 | 0.4164 | 0.0000 |
| MMR_BASELINE_V2_COHERE | 0.4667 | 0.5000 | 0.6000 | 0.4154 | 0.0000 |

## Failure analysis

- Multi-report evidence completeness remains the hardest retrieval objective.
- Table/numeric and comparative questions are more fragile than single-fact questions.
- Cohere improves retrieval but materially increases latency.
- Unstructured extraction was attempted but remains blocked because non-OCR extraction returned zero usable elements and OCR requires Tesseract.
- Comparative generation remains the weakest generation dimension because incomplete evidence must be caveated or abstained from.

## What is production-ready and what is not

The repository is ready for a local demo and interview explanation. It is not production-ready because it lacks fresh V2 held-out evaluation, human evaluation, robust caching, operational monitoring, and deployment hardening.

## Future work

- Build a fresh V2 evaluation set
- Add human evaluation
- Retry Unstructured with Tesseract/OCR deliberately installed
- Cache Cohere reranker calls
- Improve Streamlit live-query mode
- Package and deploy with monitoring
