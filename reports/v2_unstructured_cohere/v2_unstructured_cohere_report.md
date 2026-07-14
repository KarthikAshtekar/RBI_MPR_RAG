# V2 Unstructured + Cohere Retrieval Report

Temporal multi-document RAG for RBI Monetary Policy Reports

## Motivation

V2 evaluates whether layout-aware extraction and Cohere reranking improve retrieval for policy stance and narrative evolution questions.

## Current final baseline

Baseline final dev result: CER=0.36666666666666664, Evidence=0.5055555555555555, Macro MRR=0.3537037037037037, Hit=0.4.

## Environment readiness

Unstructured installed: True; Cohere installed: True; Cohere key available: True; Groq key available: True.

## Unstructured extraction implementation

Implemented in `src/rbi_rag/unstructured_extraction.py`; extraction status is recorded in `extraction/unstructured_extraction_manifest.json`.

## Extraction audit

See `extraction/unstructured_extraction_audit.md`.

## Cohere reranker implementation

Implemented in `src/rbi_rag/cohere_reranker.py`; API calls are gated by package/key availability and are not used in unit tests.

## Experiment matrix

V2_BASELINE_FINAL, V2_UNSTRUCTURED_ONLY, V2_COHERE_ONLY, V2_UNSTRUCTURED_COHERE

## Development results

- V2_COHERE_ONLY: CER=0.4666666666666667, Hit=0.5, Evidence=0.6, MRR=0.41537037037037033
- V2_BASELINE_FINAL: CER=0.36666666666666664, Hit=0.4, Evidence=0.5055555555555555, MRR=0.3537037037037037

## Skipped experiments

- V2_UNSTRUCTURED_ONLY: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
- V2_UNSTRUCTURED_COHERE: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.

## Category-level results

Saved in `v2_category_results.*`.

## Paired comparisons

Computed rows: 6. No result is conclusive when intervals cross zero.

## Selected V2 configuration

Selected: `V2_COHERE_ONLY`. Reason: Selected by V2 development eligibility and metric ordering.

## Optional held-out diagnostic status/results

Status: `heldout_diagnostic_prepared_not_run`. This diagnostic reuses a held-out set that was already evaluated in Phase 7. It is useful for comparison but should not be presented as a fresh final benchmark.

## Generation readiness

Status: `not_ready`. Generation was not run.

## Limitations

- Development results are not a fresh held-out benchmark.
- Unstructured may fall back to lighter PDF partitioning if system OCR/layout dependencies are unavailable.
- No fresh V2 held-out benchmark was created.
- The project is not production-ready.

## Next steps

- fix generation evaluator
- run Groq generation evaluation
- add history-aware query rewriting
- build Streamlit interface
- optionally create a new fresh evaluation set for V2
