# Interview Explanation

1. **Why did multi-document performance look lower than single-document?**  
   The multi-document task is stricter: the system must retrieve evidence from the correct report periods, often across multiple reports, while avoiding contamination.

2. **Why are Hit Rate and CER not directly comparable?**  
   Single-document Hit-Rate@4 checks whether one relevant chunk appears in the top 4. Complete Evidence Recall requires all required evidence across report periods.

3. **Why use BM25 + dense hybrid search?**  
   BM25 helps exact terms and numeric wording; dense retrieval helps semantic matches. RBI reports need both.

4. **Why use RRF?**  
   Reciprocal Rank Fusion combines dense and BM25 rankings without requiring score calibration.

5. **What is MMR and did it help?**  
   MMR is Maximal Marginal Relevance, a diversity selection method. The final comparison distinguishes true MMR experiment artifacts from DIV01, which is only an exact-overlap diversity filter.

6. **What is MRR and how is it different from MMR?**  
   MRR is Mean Reciprocal Rank, a metric measuring how early relevant evidence appears. MMR is a retrieval/selection technique. They are unrelated except for similar abbreviations.

7. **Why use Cohere reranking?**  
   Cohere reranking improved development retrieval metrics by better ordering candidate chunks after hybrid retrieval.

8. **Why did Cohere improve results but slow latency?**  
   It adds remote API reranking calls over candidate documents, so quality improves at a material latency cost.

9. **Why add sufficiency gating?**  
   The model answered even when retrieval was incomplete. The gate makes the system abstain or caveat unsupported answers.

10. **What is the final architecture?**  
   PyPDFLoader, dense + BM25 hybrid retrieval, RRF, Cohere reranking, report-aware quotas, source-labelled contexts, Groq generation, sufficiency gate, and citation/temporal validation.

11. **What are the limitations?**  
   Current V2 results are development-only, metrics are deterministic heuristics, Unstructured extraction depends on Poppler/layout tooling, and Cohere adds latency.

12. **What would you improve next?**  
   Create a fresh V2 evaluation set, run final retrieval plus generation on it, then build the Streamlit interface.
