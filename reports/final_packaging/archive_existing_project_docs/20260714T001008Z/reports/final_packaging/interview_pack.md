# Interview Pack

## What problem did you solve?

I built a temporal multi-document RAG system over RBI Monetary Policy Reports so users can ask questions that require correctly attributed evidence across report periods.

## Why multi-document temporal RAG?

RBI policy interpretation changes over time. The value is in comparing policy stance and narrative evolution across reports, not just retrieving one paragraph.

## Why did multi-document scores look lower than single-document scores?

The metrics are stricter. Single-document Hit-Rate@4 needs one relevant chunk; multi-report CER needs all required evidence from the right reports.

## Why BM25 + dense embeddings?

BM25 helps exact policy terms and numbers, while dense retrieval helps semantic phrasing. RBI reports need both.

## Why RRF?

RRF fuses dense and BM25 rankings without requiring score calibration.

## Difference between MRR and MMR?

MRR is Mean Reciprocal Rank, an evaluation metric. MMR is Maximal Marginal Relevance, a diversity-aware selection method.

## Did MMR help?

MMR status: evaluated_selected. Selected system after MMR: MMR_LAMBDA_06.

## Why Cohere reranking?

Cohere reranking improved development retrieval quality by better ordering the hybrid candidate set.

## Why did Cohere increase latency?

It adds remote reranking calls over candidate documents, so latency increases materially.

## Why evidence sufficiency gate?

It prevents the generator from giving unsupported answers when retrieved evidence is incomplete.

## Why not just trust the LLM?

The LLM can answer fluently without complete evidence. The gate and citations force source-grounded behavior.

## How did you prevent wrong-report attribution?

The pipeline uses report IDs, report-aware routing/quotas, source-labelled contexts, and contamination checks.

## How did you evaluate?

Retrieval used CER, All-Reports Hit, Evidence Recall, Macro MRR, coverage, contamination, latency, and context-size metrics. Generation used deterministic heuristic checks over saved outputs.

## What were the final metrics?

Retrieval: CER 0.4667, All-Reports Hit 0.5000, Evidence Recall 0.6000, Macro MRR 0.4154. Generation factual correctness 0.7954, faithfulness 0.9762, abstention 1.0000.

## What failed?

Unstructured extraction did not produce usable non-OCR elements and needs Tesseract/OCR. Comparative generation also remains weak when evidence is incomplete.

## What would you improve next?

Create a fresh V2 evaluation set, add human evaluation, cache Cohere calls, retry Unstructured with OCR deliberately installed, and improve live Streamlit mode.

## Is it production-ready?

No. It is demo/interview-ready with known limitations, not production-ready.

## How would you scale/deploy it?

Precompute indexes and reranker cache, add a service API, add observability and evaluation monitoring, secure key management, and deploy the Streamlit/API layer behind auth.
