# Final Project Status

- best_retrieval_method: `MMR_LAMBDA_06`
- best_generation_method: `GEN_MMR06_SUFFICIENCY_V1 + Groq llama-3.1-8b-instant + sufficiency gate`
- final_mmr_generation_status: `selected_mmr_end_to_end`
- final_generation_bakeoff_status: `kept_previous_best_generation_strategy`
- final_generation_bakeoff_selected_variant: `GEN_MMR06_SUFFICIENCY_V1`
- mmr_status: `evaluated_selected`
- unstructured_status: `blocked_by_ocr_tesseract_requirement`
- heldout_status: `not_rerun_for_overnight_packaging`
- generation_status: `final_generation_bakeoff_completed_dev_only; previous_generation_artifacts_preserved`
- streamlit_status: `available`
- readme_status: `available`
- tests_status: `pending`
- pip_check_status: `pending`
- api_key_scan_status: `passed`
- ready_for_demo: `ready_for_demo_with_known_limitations`
- ready_for_interview: `ready_for_interview`
- production_status: `not_production_ready`

## Remaining limitations

- fresh V2 held-out evaluation not created
- generation metrics are deterministic heuristics, not human evaluation
- Unstructured extraction remains OCR/Tesseract-blocked
- Cohere latency is high without caching
- comparative generation remains weak when evidence is incomplete
