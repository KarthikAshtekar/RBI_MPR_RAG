# V2 Paired Comparisons

Baseline: `V2_BASELINE_FINAL`. Bootstrap resamples: 2000; seed: 42.

- V2_COHERE_ONLY / complete_evidence_recall: diff=0.1, CI=[-0.03333333333333333, 0.26666666666666666], conclusive=False
- V2_COHERE_ONLY / all_reports_hit: diff=0.1, CI=[-0.03333333333333333, 0.26666666666666666], conclusive=False
- V2_COHERE_ONLY / evidence_recall: diff=0.09444444444444446, CI=[-0.03333333333333333, 0.22777777777777777], conclusive=False
- V2_COHERE_ONLY / macro_report_mrr: diff=0.054411764705882354, CI=[-0.044035947712418304, 0.16013071895424838], conclusive=False
- V2_COHERE_ONLY / median_latency: diff=7394.191549999746, CI=[6008.628538234916, 8929.523223529994], conclusive=True
- V2_COHERE_ONLY / mean_estimated_tokens: diff=-47.411764705882355, CI=[-81.3529411764706, -17.529411764705884], conclusive=True

## Skipped experiment comparisons

- V2_UNSTRUCTURED_ONLY: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
- V2_UNSTRUCTURED_COHERE: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
