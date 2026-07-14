# True MMR Experiment Report

This report evaluates Maximal Marginal Relevance as a final context-selection layer over saved V2_COHERE_ONLY development reranker outputs.

MRR is Mean Reciprocal Rank, a metric. MMR is Maximal Marginal Relevance, a selection technique.

## Formula

`MMR(document) = lambda * relevance_score - (1 - lambda) * max_similarity_to_selected_documents`

## Results

- MMR_LAMBDA_06: CER=0.5333333333333333, Hit=0.5666666666666667, MRR=0.4054629629629629
- MMR_LAMBDA_07: CER=0.5, Hit=0.5333333333333333, MRR=0.41453703703703704
- MMR_LAMBDA_08: CER=0.4666666666666667, Hit=0.5, MRR=0.4163888888888889
- MMR_BASELINE_V2_COHERE: CER=0.4666666666666667, Hit=0.5, MRR=0.41537037037037033

## Decision

Status: `evaluated_selected`
Selected: `MMR_LAMBDA_06`
Reason: MMR improved development retrieval quality while preserving coverage and contamination.

## Integrity

Validation: `passed`

Held-out evaluation was not run. Generation was not run.
