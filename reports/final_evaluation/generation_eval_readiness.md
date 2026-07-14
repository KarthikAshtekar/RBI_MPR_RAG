# Generation Evaluation Readiness

Status: not_ready

| Check | Passed |
|---|---:|
| heldout_retrieval_completed | True |
| heldout_retrieval_integrity_passed | True |
| final_retrieval_config_unchanged | True |
| groq_api_key_available | True |
| generation_can_use_frozen_retrieval_outputs | False |
| no_retrieval_tuning_after_heldout | True |

Reason: Generation was not run because the checked-in multi-report generation command does not consume configs/final_retrieval_selected.yaml or the Phase 7 frozen retrieval outputs; running it would evaluate the older configs/multi_report.yaml retrieval settings instead of the final ADJ00 configuration.
