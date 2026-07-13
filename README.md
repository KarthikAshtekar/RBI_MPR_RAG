# RBI Monetary Policy Report RAG

Reproducible frozen single-report baseline for the RBI **April 2025 Monetary Policy Report**, plus a registry-driven temporal multi-report extension. The Streamlit interface remains outside this phase.

The repository now also contains a Phase 2–4 architecture for **Temporal multi-document RAG for RBI Monetary Policy Reports**. It is temporal because it routes and compares dated source reports; it is not chat-history-aware RAG and has no conversational memory or query rewriting.

## Final baseline pipeline

Generation always uses:

```text
dense top-15 + BM25 top-15
        -> Reciprocal Rank Fusion (k=60, top-15 candidates)
        -> cross-encoder reranking
        -> final top-4 context
        -> Llama 3.1 8B Instant generation
```

Retrieval evaluation also retains dense, dense+rereanker, BM25, BM25+reranker, hybrid RRF, and hybrid RRF+reranker for comparison. A reranker cannot recover a chunk absent from its candidate pool, but it can promote a relevant result from ranks 5–15 into the final four; it can therefore improve both MRR and Hit-Rate@4.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Only generation commands need `GROQ_API_KEY`. No key is stored in YAML.

## Reproducible commands

```powershell
rbi-rag --config configs/baseline.yaml build-index
rbi-rag --config configs/baseline.yaml retrieval-eval
rbi-rag --config configs/baseline.yaml generation-eval
rbi-rag --config configs/baseline.yaml report
python -m compileall src scripts
python -m pytest
```

Multi-report commands use `configs/multi_report.yaml`:

```powershell
rbi-rag --config configs/multi_report.yaml validate-reports
rbi-rag --config configs/multi_report.yaml build-index
rbi-rag --config configs/multi_report.yaml inspect-index
rbi-rag --config configs/multi_report.yaml route-query --query "Compare April and October 2025 inflation projections."
rbi-rag --config configs/multi_report.yaml retrieve --query "How did policy stance evolve across reports?"
rbi-rag --config configs/multi_report.yaml retrieval-eval
rbi-rag --config configs/multi_report.yaml report
```

## Multi-report architecture

`configs/reports.yaml` is the versioned report registry. Available reports are chunked with the frozen 1,000/300 settings and indexed in a separate shared Chroma collection with `report_id` filtering. The original single-report index remains separate. BM25 is rebuilt deterministically in memory once per report, preventing cross-report leakage without unsafe serialization.

The offline router recognizes explicit periods, pairwise comparisons, trends, latest/earliest/previous language, global unspecified queries, and unsupported periods. Single-report retrieval filters both dense and sparse branches. Comparative retrieval runs dense, BM25, RRF, and reranking independently within every required report, then retains three chunks per report for pairwise comparisons and two per report for trends. Deduplication occurs only after report quotas are satisfied.

Comparative contexts use explicit `<SOURCE>` labels with report period, page, and chunk ID. The generator is instructed to distinguish report facts from cautious synthesis about policy stance and narrative evolution. Structured citations are validated against the chunks actually supplied.

Multi-report evaluation adds report coverage, all-reports hit, per-report hit/MRR, macro report MRR, evidence recall, balance diagnostics, and single-report contamination. Pairwise and trend factual cases are not scored until their PDFs and page labels can be inspected.

To add another MPR, place the local PDF under `data/raw/`, add its dated entry to `configs/reports.yaml`, run `validate-reports`, then rebuild the multi-report index. Only the changed report is replaced. Add manually verified router and factual cases before reporting comparative performance.

The equivalent entry scripts are under `scripts/`.

## Evaluation integrity

- Every metric attempt uses a new metric instance.
- Failed judge calls store `score: null`, error type, error text, and attempt count.
- Generation checkpoints are atomic and written after each metric.
- Resume skips successful metrics and reruns only failed or missing metrics.
- A mismatched PDF/configuration fingerprint cannot resume an old checkpoint.
- Means, medians, and standard deviations use successful cases only and are always shown with coverage.
- Raw retrieval results are saved before summaries.
- The 30 source questions are explicitly a development set; no held-out score is claimed.

## Outputs

Current machine-generated outputs live in `reports/current/`:

- `retrieval_raw_results.json`
- `retrieval_question_results.csv`
- `retrieval_pipeline_summary.csv`
- `retrieval_summary.json`
- `generation_evaluation.json` after a credentialed run
- `baseline_report.md`, generated only from saved outputs

Superseded reports and preliminary outputs are preserved in `reports/archived/`, with provenance in `archive_manifest.json`. The original notebook remains unchanged.

## Limitations

The development set has only 30 questions and page-level relevance labels. It is useful for iteration, not an unbiased held-out estimate. Hosted generation can vary even at temperature zero. Generation metrics are not validated unless every reported average includes its measured coverage.

PyPDFLoader may flatten tables and omit chart semantics. October 2025 and April 2026 are currently registered but missing locally, so current multi-report factual scoring covers April 2025 only. The system is not production-ready and generation evaluation remains unexecuted without `GROQ_API_KEY`.
