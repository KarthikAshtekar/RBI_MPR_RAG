# Final Generation Bake-Off Variant Registry

| Variant | Mode | Retrieval | Prompt | Ordering | Priority |
|---|---|---|---|---|---:|
| GEN_V2_COHERE_SUFFICIENCY_V1 | reuse_v2_sufficiency | V2_COHERE_ONLY | v2_sufficiency_prompt_v1 | default | 0 |
| GEN_MMR06_SUFFICIENCY_V1 | reuse_mmr_generation | MMR_LAMBDA_06 | v2_sufficiency_prompt_v1 | page_order | 0 |
| GEN_MMR07_SUFFICIENCY_V1 | live_generation | MMR_LAMBDA_07 | v2_sufficiency_prompt_v1 | page_order | 1 |
| GEN_MMR08_SUFFICIENCY_V1 | live_generation | MMR_LAMBDA_08 | v2_sufficiency_prompt_v1 | page_order | 2 |
| GEN_MMR06_CHRONO_ORDER_V1 | live_generation | MMR_LAMBDA_06 | v2_sufficiency_prompt_v1 | chrono_order | 3 |
| GEN_MMR06_RERANK_ORDER_V1 | live_generation | MMR_LAMBDA_06 | v2_sufficiency_prompt_v1 | rerank_order | 4 |
| GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1 | live_generation | MMR_LAMBDA_06 | evidence_first_prompt_v1 | page_order | 5 |
| GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1 | live_generation | MMR_LAMBDA_06 | comparative_strict_prompt_v1 | page_order | 6 |
| GEN_MMR06_CITATION_REPAIR_V1 | citation_repair | MMR_LAMBDA_06 | deterministic_citation_repair_v1 | page_order | 7 |
