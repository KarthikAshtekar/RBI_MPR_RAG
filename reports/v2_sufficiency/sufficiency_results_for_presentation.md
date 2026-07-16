# Sufficiency Results for Presentation

Temporal multi-document RAG for RBI Monetary Policy Reports

## Why sufficiency gating was added

The previous V2 dev generation answered even when retrieval evidence was incomplete, so safety needed to move from retrieval tuning to evidence sufficiency detection.

## Previous weakness

Old incomplete-retrieval abstention rate: 0.1

## New behaviour

Sufficiency statuses: {'sufficient': 14, 'partially_sufficient': 16, 'insufficient': 4}
Required generation behaviours: {'answer': 14, 'answer_with_caveat': 16, 'abstain': 4}

## Metric changes

| Metric | Old | New | Delta |
|---|---:|---:|---:|
| abstention_correctness | 0.47058823529411764 | 1.0 | 0.5294117647058824 |
| citation_completeness | 1.0 | 1.0 | 0.0 |
| citation_correctness | 0.8823529411764706 | 0.8823529411764706 | 0.0 |
| comparative_correctness | 0.9166666666666666 | 0.2777777777777778 | -0.6388888888888888 |
| contextual_recall | 0.4117647058823529 | 0.4117647058823529 | 0.0 |
| contextual_relevancy | 0.5294117647058824 | 0.5294117647058824 | 0.0 |
| factual_correctness | 0.6343914458612334 | 0.7954456988291574 | 0.16105425296792397 |
| faithfulness_to_context | 0.8657197071510555 | 0.9761655452382324 | 0.1104458380871769 |
| temporal_attribution_correctness | 0.8823529411764706 | 0.8823529411764706 | 0.0 |

## Examples of safer abstention

- `apr26_dev_001` changed from answered to abstained.
- `apr26_dev_003` changed from answered to abstained.
- `apr26_dev_004` changed from answered to abstained.

## Examples where answer quality remained strong

- `oct_dev_001` retained/new factual score 1.0.
- `oct_dev_002` retained/new factual score 0.8571428571428571.
- `oct_dev_004` retained/new factual score 0.75.

## Trade-off

The gate improves safety by reducing unsupported answers, but answer coverage can drop because insufficient cases abstain.

## Scientific caveat

This is development-only. Held-out retrieval and held-out generation were not run because the prior held-out set has already been used.

The system is not production-ready.
