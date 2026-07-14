# Final End-to-End MMR Generation Experiment Report

## Why this experiment was needed

MMR improved retrieval-only development metrics, but generation had not been rerun on MMR-selected contexts. This experiment tests whether that retrieval improvement carries through to answer quality.

## MMR retrieval result summary

CER=0.5333333333333333, All-Reports Hit=0.5666666666666667, Evidence Recall=0.65, Macro MRR=0.4054629629629629.

## Generation setup

Experiment `GEN_MMR06_SUFFICIENCY_V1` used Groq `llama-3.1-8b-instant`, temperature 0, prompt `v2_sufficiency_prompt_v1`, retrieval source `MMR_LAMBDA_06`, and the sufficiency gate.

## Sufficiency classification results

Statuses: {'sufficient': 16, 'partially_sufficient': 14, 'insufficient': 4}
Behaviours: {'answer': 16, 'answer_with_caveat': 14, 'abstain': 4}

## Generation metrics

| Metric | Previous V2 sufficiency | MMR06 sufficiency |
|---|---:|---:|
| factual_correctness | 0.7954456988291574 | 0.8153331682936946 |
| faithfulness_to_context | 0.9761655452382324 | 0.9731204857728691 |
| contextual_relevancy | 0.5294117647058824 | 0.5735294117647058 |
| contextual_recall | 0.4117647058823529 | 0.47058823529411764 |
| citation_correctness | 0.8823529411764706 | 0.8823529411764706 |
| citation_completeness | 1.0 | 1.0 |
| temporal_attribution_correctness | 0.8823529411764706 | 0.8823529411764706 |
| comparative_correctness | 0.2777777777777778 | 0.3333333333333333 |
| abstention_correctness | 1.0 | 1.0 |

## Final selection decision

Decision: `selected_mmr_end_to_end`. Best setting: `GEN_MMR06_SUFFICIENCY_V1`.

## What improved

See `generation_comparison.md` for metric-level deltas.

## What did not improve

The selection decision records whether MMR generation did or did not replace the previous generation setting.

## Remaining limitations

- Development-only evaluation.
- No held-out rerun or fresh evaluation set.
- Deterministic heuristic generation metrics, not human evaluation.
- No Unstructured/Tesseract work.

## Final project claim wording

Use retrieval-only and generation claims separately unless the selection decision is `selected_mmr_end_to_end`.
