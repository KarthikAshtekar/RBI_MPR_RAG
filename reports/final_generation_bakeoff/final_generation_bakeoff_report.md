# Final Generation Strategy Bake-Off Report

## Scope

This is a development-only bake-off for Temporal multi-document RAG for RBI Monetary Policy Reports. It does not use held-out data, create a fresh evaluation set, run Unstructured/Tesseract work, or change retrieval models.

## Why this bake-off was needed

The project had a selected retrieval-only MMR configuration and a strong MMR06 sufficiency-gated generation result. The remaining question was whether nearby MMR lambdas, context ordering, prompt variants, or deterministic citation repair improved the end-to-end answer metrics.

## Leaderboard

| Variant | Retrieval | Prompt | Ordering | Status | Factual | Citation | Temporal | Comparative | Abstention | Eligible |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| GEN_MMR06_SUFFICIENCY_V1 | MMR_LAMBDA_06 | v2_sufficiency_prompt_v1 | page_order | completed_reused | 0.8153331682936946 | 0.8823529411764706 | 0.8823529411764706 | 0.3333333333333333 | 1.0 | eligible |
| GEN_V2_COHERE_SUFFICIENCY_V1 | V2_COHERE_ONLY | v2_sufficiency_prompt_v1 | default | completed_reused | 0.7954456988291574 | 0.8823529411764706 | 0.8823529411764706 | 0.2777777777777778 | 1.0 | eligible |
| GEN_MMR07_SUFFICIENCY_V1 | MMR_LAMBDA_07 | v2_sufficiency_prompt_v1 | page_order | completed | 0.804133157291052 | 0.8823529411764706 | 0.8823529411764706 | 0.2777777777777778 | 1.0 | eligible |
| GEN_MMR08_SUFFICIENCY_V1 | MMR_LAMBDA_08 | v2_sufficiency_prompt_v1 | page_order | skipped | None | None | None | None | None | not_eligible |
| GEN_MMR06_CHRONO_ORDER_V1 | MMR_LAMBDA_06 | v2_sufficiency_prompt_v1 | chrono_order | skipped | None | None | None | None | None | not_eligible |
| GEN_MMR06_RERANK_ORDER_V1 | MMR_LAMBDA_06 | v2_sufficiency_prompt_v1 | rerank_order | skipped | None | None | None | None | None | not_eligible |
| GEN_MMR06_EVIDENCE_FIRST_PROMPT_V1 | MMR_LAMBDA_06 | evidence_first_prompt_v1 | page_order | skipped | None | None | None | None | None | not_eligible |
| GEN_MMR06_COMPARATIVE_STRICT_PROMPT_V1 | MMR_LAMBDA_06 | comparative_strict_prompt_v1 | page_order | skipped | None | None | None | None | None | not_eligible |
| GEN_MMR06_CITATION_REPAIR_V1 | MMR_LAMBDA_06 | deterministic_citation_repair_v1 | page_order | completed_repaired | 0.8153331682936946 | 0.8823529411764706 | 0.8823529411764706 | 0.3333333333333333 | 1.0 | eligible |

## Selection decision

Status: `kept_previous_best_generation_strategy`
Selected variant: `GEN_MMR06_SUFFICIENCY_V1`
Selected retrieval method: `MMR_LAMBDA_06`

No eligible bake-off variant beat the current MMR06 sufficiency baseline under the selection policy.

## Category-level findings

See `category_analysis.md` for query type, source structure, numeric-evidence, and sufficiency-status breakdowns.

## Scientific caveats

- Development-only comparison.
- No held-out rerun.
- No fresh V2 evaluation set.
- Metrics are deterministic heuristic evaluation signals, not human evaluation.
- Skipped variants are explicitly marked and must not be claimed as tested.

## Interview-ready explanation

The final selected development system is `GEN_MMR06_SUFFICIENCY_V1` using `MMR_LAMBDA_06` retrieval. It was selected because it ranked best among eligible development-only variants under factual, citation, temporal attribution, abstention, comparative, latency, and simplicity criteria.

## Validation

Validation status: `passed`. API-key scan: `passed`.
