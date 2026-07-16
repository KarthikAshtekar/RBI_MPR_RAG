# V2 Results for Presentation

Temporal multi-document RAG for RBI Monetary Policy Reports

## Why V2 was attempted

V2 tests whether layout-aware PDF extraction and a stronger reranker improve policy stance and narrative evolution retrieval, especially for comparative and table/numeric evidence.

## Controlled development results

- Current final baseline: CER=0.36666666666666664, Macro MRR=0.3537037037037037
- V2_UNSTRUCTURED_ONLY: skipped - RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
- V2_COHERE_ONLY: CER=0.4666666666666667, Macro MRR=0.41537037037037033
- V2_UNSTRUCTURED_COHERE: skipped - RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.

Selected V2 result: `V2_COHERE_ONLY`.
V2 improved Complete Evidence Recall: True.
V2 improved Macro MRR: True.
Table/numeric selected result: baseline CER=0.25, selected CER=0.4.
Latency trade-off: baseline median=3159.0397500112886 ms, selected median=12948.27224999608 ms.

Scientific caveat: the previous held-out set was already evaluated in Phase 7 and was not used for V2 selection.

Interview-ready explanation: V2 was designed as an attribution experiment. Each parser/reranker arm is evaluated on development data with the same retrieval skeleton, and unavailable or failed arms are recorded rather than filled with fabricated metrics.
