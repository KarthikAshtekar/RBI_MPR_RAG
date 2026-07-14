# V2 Generation Results for Presentation

Temporal multi-document RAG for RBI Monetary Policy Reports

## Why generation was previously blocked

Generation was blocked because the evaluator was not wired to consume the frozen V2 selected retrieval config and saved source-labelled retrieval outputs.

## How it was fixed

The evaluator now builds source-labelled contexts from saved `V2_COHERE_ONLY` retrieval outputs without rerunning retrieval.

## Selected retrieval config

`V2_COHERE_ONLY`

## Retrieval development metrics

- Complete Evidence Recall: 0.4666666666666667
- All-Reports Hit: 0.5
- Evidence Recall: 0.6
- Macro Report MRR: 0.41537037037037033
- Median retrieval latency ms: 12906.782750000275
- Mean estimated tokens: 2096.9666666666667

## Generation development metrics

- abstention_correctness: mean=0.47058823529411764, n=34
- citation_completeness: mean=1.0, n=30
- citation_correctness: mean=0.8823529411764706, n=34
- comparative_correctness: mean=0.9166666666666666, n=18
- contextual_recall: mean=0.4117647058823529, n=34
- contextual_relevancy: mean=0.5294117647058824, n=34
- factual_correctness: mean=0.6343914458612334, n=32
- faithfulness_to_context: mean=0.8654068160371632, n=34
- temporal_attribution_correctness: mean=0.8823529411764706, n=34

Citation correctness: 0.8823529411764706
Temporal attribution correctness: 0.8823529411764706
Comparative correctness: 0.9166666666666666
Table/numeric mean factual score: 0.6549121147111432

## Example good answers

- `oct_dev_002`: generated with 3 parsed citations.
- `oct_dev_003`: generated with 1 parsed citations.

## Example failure modes

- `unsupported_dev_001`: missing parsed citations
- `unsupported_dev_002`: missing parsed citations

Latency/cost caveat: generation uses Groq API calls and should be treated separately from retrieval latency.

Scientific caveat: development generation only; held-out generation was not run; any V2 held-out diagnostic is not a fresh benchmark unless separately created and labelled.
