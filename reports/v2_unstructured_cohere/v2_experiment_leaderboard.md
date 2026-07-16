# V2 Experiment Leaderboard

| Experiment | Parser | Reranker | CER | Hit | Evidence | MRR | Coverage | Contam | Median ms | Tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_COHERE_ONLY | PyPDFLoader | cohere | 0.4666666666666667 | 0.5 | 0.6 | 0.41537037037037033 | 1.0 | 0.0 | 12948.27224999608 | 2097.2 |
| V2_BASELINE_FINAL | PyPDFLoader | local_cross_encoder | 0.36666666666666664 | 0.4 | 0.5055555555555555 | 0.3537037037037037 | 1.0 | 0.0 | 3159.0397500112886 | 2150.9333333333334 |

## Skipped

- V2_UNSTRUCTURED_ONLY: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
- V2_UNSTRUCTURED_COHERE: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
