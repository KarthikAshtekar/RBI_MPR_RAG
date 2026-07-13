# Temporal Multi-Document RAG for RBI Monetary Policy Reports

Generated: 2026-07-12T21:55:56.798749+00:00

## Corpus status

| Report | Availability | Pages | Chunks | Index status |
|---|---|---:|---:|---|
| April 2025 | available | 116 | 461 | reused |
| October 2025 | missing | n/a | n/a | not_indexed |
| April 2026 | missing | n/a | n/a | not_indexed |

## Router

Offline router accuracy: **100.00%**.

## Retrieval evaluation

Scored cases: 9
Mean report coverage: 1.0
All-reports hit rate: 1.0
Macro report MRR: 0.7777777777777778
Single-report cross-contamination rate: 0.0

Only verified April 2025 single-report cases are scored. Pairwise and trend factual cases await the missing PDFs.

## Generation evaluation

Not executed. GROQ_API_KEY was unavailable; no generation metrics are claimed.

## Limitations

- October 2025 and April 2026 PDFs are missing locally.
- PyPDFLoader can lose table structure and chart semantics.
- This system has no conversational memory or history-aware query rewriting.
- This is an evaluated research baseline, not a production-ready system.
