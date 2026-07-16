# Final Project Status

- best_retrieval_method: `MMR_LAMBDA_06`
- best_generation_method: `V2_COHERE_ONLY + Groq llama-3.1-8b-instant + sufficiency gate`
- final_mmr_generation_status: `not_run`
- final_generation_bakeoff_status: `not_run`
- final_generation_bakeoff_selected_variant: `None`
- mmr_status: `evaluated_selected`
- unstructured_status: `blocked_by_ocr_tesseract_requirement`
- heldout_status: `not_rerun_for_overnight_packaging`
- generation_status: `existing_dev_sufficiency_generation_used; not_rerun`
- streamlit_status: `available`
- readme_status: `available`
- tests_status: `passed`
- pip_check_status: `passed`
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
