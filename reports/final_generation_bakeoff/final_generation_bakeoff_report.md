# Final Generation Strategy Bake-Off Report

## Scope

This is a development-only bake-off for Temporal multi-document RAG for RBI Monetary Policy Reports. It does not use held-out data, create a fresh evaluation set, run Unstructured/Tesseract work, or change retrieval models.

## Why this bake-off was needed

The project had a selected retrieval-only MMR configuration and a strong MMR06 sufficiency-gated generation result. The remaining question was whether nearby MMR lambdas, context ordering, prompt variants, or deterministic citation repair improved the end-to-end answer metrics.

## Leaderboard

| Variant | Retrieval | Prompt | Ordering | Status | Factual | Citation | Temporal | Comparative | Abstention | Eligible |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|

## Selection decision

Status: `None`
Selected variant: `None`
Selected retrieval method: `None`



## Category-level findings

See `category_analysis.md` for query type, source structure, numeric-evidence, and sufficiency-status breakdowns.

## Scientific caveats

- Development-only comparison.
- No held-out rerun.
- No fresh V2 evaluation set.
- Metrics are deterministic heuristic evaluation signals, not human evaluation.
- Skipped variants are explicitly marked and must not be claimed as tested.

## Interview-ready explanation

The final selected development system is `None` using `None` retrieval. It was selected because it ranked best among eligible development-only variants under factual, citation, temporal attribution, abstention, comparative, latency, and simplicity criteria.

## Validation

Validation status: `failed`. API-key scan: `passed`.
