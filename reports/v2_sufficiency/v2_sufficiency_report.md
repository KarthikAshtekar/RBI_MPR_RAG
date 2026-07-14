# V2 Sufficiency Gate Report

Temporal multi-document RAG for RBI Monetary Policy Reports

Status: `dev_sufficiency_generation_complete`
Integrity: `passed`

## Classifier results

Sufficiency statuses: {'sufficient': 14, 'partially_sufficient': 16, 'insufficient': 4}
Required generation behaviours: {'answer': 14, 'answer_with_caveat': 16, 'abstain': 4}

## Generation results

Rows: 34; successes: 34; failures: 0.
Prompt version: `v2_sufficiency_prompt_v1`.

## Evaluation metrics

- abstention_correctness: mean=1.0, n=34
- citation_completeness: mean=1.0, n=14
- citation_correctness: mean=0.8823529411764706, n=34
- comparative_correctness: mean=0.2777777777777778, n=18
- contextual_recall: mean=0.4117647058823529, n=34
- contextual_relevancy: mean=0.5294117647058824, n=34
- factual_correctness: mean=0.7954456988291574, n=14
- faithfulness_to_context: mean=0.9761655452382324, n=34
- temporal_attribution_correctness: mean=0.8823529411764706, n=34

## Old vs new comparison

Old incomplete-retrieval abstention rate: 0.1
New incomplete-retrieval abstention rate: 1.0
Old unsupported-answer rate when retrieval incomplete: 0.9
New unsupported-answer rate when retrieval incomplete: 0.0
Old complete-retrieval factual correctness: 0.6942262207299802
New complete-retrieval factual correctness: 0.7954456988291574

## Limitations

- Sufficiency classification is deterministic and label-assisted on development data only.
- Insufficient cases use a hard abstention gate, so answer coverage can drop.
- Metrics remain deterministic heuristics, not human judgement.
- Held-out data was not used.

## Exact next phase

Create a fresh V2 evaluation set, run final retrieval plus generation on that fresh set, then build the Streamlit interface.
