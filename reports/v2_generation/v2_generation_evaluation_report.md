# V2 Generation Evaluation Report

Temporal multi-document RAG for RBI Monetary Policy Reports

## Environment readiness

Groq key available: True. Cohere key available: True.

## Retrieval input validation

Status: passed. The evaluator used `V2_COHERE_ONLY` saved development retrieval outputs.

## Context-building method

Contexts are built from `selected_chunks_by_report` in saved V2 raw retrieval rows, grouped by report period in chronological order.

## Prompt template/version

`v2_source_labelled_context_v1`

## Model/provider settings

Provider/model: Groq / llama-3.1-8b-instant; temperature=0.0.

## Generation run status

Status: `dev_generation_complete`. Rows=34; successes=34; failures=0.

## Generation metrics

- abstention_correctness: mean=0.47058823529411764, n=34
- citation_completeness: mean=1.0, n=30
- citation_correctness: mean=0.8823529411764706, n=34
- comparative_correctness: mean=0.9166666666666666, n=18
- contextual_recall: mean=0.4117647058823529, n=34
- contextual_relevancy: mean=0.5294117647058824, n=34
- factual_correctness: mean=0.6343914458612334, n=32
- faithfulness_to_context: mean=0.8654068160371632, n=34
- temporal_attribution_correctness: mean=0.8823529411764706, n=34

## Metric coverage

- abstention_correctness: coverage=1.0, failed=0, not_applicable=0
- citation_completeness: coverage=1.0, failed=0, not_applicable=4
- citation_correctness: coverage=1.0, failed=0, not_applicable=0
- comparative_correctness: coverage=1.0, failed=0, not_applicable=16
- contextual_recall: coverage=1.0, failed=0, not_applicable=0
- contextual_relevancy: coverage=1.0, failed=0, not_applicable=0
- factual_correctness: coverage=1.0, failed=0, not_applicable=2
- faithfulness_to_context: coverage=1.0, failed=0, not_applicable=0
- temporal_attribution_correctness: coverage=1.0, failed=0, not_applicable=0

## Judge/evaluator failures

Failed metric evaluations: 0. Metrics marked not applicable were excluded from their averages.

## Retrieval-to-generation analysis

Complete retrieval mean factual score: 0.6942262207299802.
Incomplete retrieval abstention rate: 0.0.
Table/numeric mean factual score: 0.6549121147111432.
Macro MRR to factual-score correlation: 0.33853171948154576.

## Citation validation

Citation correctness mean: 0.8823529411764706.
Citation completeness mean: 1.0.

## Temporal attribution validation

Temporal attribution correctness mean: 0.8823529411764706.

## Comparative correctness

Comparative correctness mean: 0.9166666666666666.

## Abstention behaviour

Abstention correctness mean: 0.47058823529411764.
Incomplete retrieval abstention rate: 0.0.

## Table/numeric analysis

Table/numeric question count: 21.
Table/numeric mean factual score: 0.6549121147111432.

## Limitations

- Metrics are deterministic heuristics, not an external human or LLM judge.
- Development generation only; held-out generation was not run.
- The project is not production-ready.

## Exact next phase

Review generation failures and citation quality, then decide whether to run a labelled post-final held-out diagnostic or create a fresh V2 evaluation set.
