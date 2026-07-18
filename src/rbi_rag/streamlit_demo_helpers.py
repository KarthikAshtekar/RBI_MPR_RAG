from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .security import contains_key_material


ROOT = Path(__file__).resolve().parents[2]

FINAL_DEV_METRICS = {
    "retrieval": {
        "complete_evidence_recall": 0.5333,
        "all_reports_hit": 0.5667,
        "evidence_recall": 0.6500,
        "macro_mrr": 0.4055,
    },
    "generation": {
        "factual_correctness": 0.7954456988291574,
        "faithfulness_to_context": 0.9761655452382324,
        "contextual_relevancy": 0.5294117647058824,
        "contextual_recall": 0.4117647058823529,
        "abstention_correctness": 1.0000,
        "citation_correctness": 0.8824,
        "citation_completeness": 1.0000,
        "temporal_attribution_correctness": 0.8824,
        "comparative_correctness": 0.2778,
    },
}

SELECTED_RETRIEVAL = "MMR_LAMBDA_06"
SELECTED_GENERATION = "V2_COHERE_ONLY + sufficiency gate"
SELECTED_MODEL = "Groq llama-3.1-8b-instant"
SELECTED_PROMPT = "v2_sufficiency_prompt_v1"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _metric_from_eval(summary: dict[str, Any], name: str) -> float | None:
    value = ((summary.get("metrics") or {}).get(name) or {}).get("mean_score")
    return float(value) if isinstance(value, (int, float)) else None


def load_final_metrics(root: Path = ROOT) -> dict[str, Any]:
    retrieval_summary = read_json(root / "reports/mmr_experiments/experiments/MMR_LAMBDA_06/summary.json", {})
    generation_summary = read_json(
        root / "reports/final_generation_bakeoff/experiments/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json",
        {},
    )
    generation_source = "reports/final_generation_bakeoff/experiments/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json"
    if not generation_summary.get("metrics"):
        generation_summary = read_json(root / "reports/v2_sufficiency/dev_sufficiency_eval_summary.json", {})
        generation_source = "reports/v2_sufficiency/dev_sufficiency_eval_summary.json"
    fallback_used: list[str] = []
    retrieval = {
        "complete_evidence_recall": retrieval_summary.get("complete_evidence_recall"),
        "all_reports_hit": retrieval_summary.get("all_reports_hit"),
        "evidence_recall": retrieval_summary.get("evidence_recall"),
        "macro_mrr": retrieval_summary.get("macro_report_mrr"),
    }
    generation = {
        key: _metric_from_eval(generation_summary, key)
        for key in FINAL_DEV_METRICS["generation"]
    }
    for key, value in list(retrieval.items()):
        if value is None:
            retrieval[key] = FINAL_DEV_METRICS["retrieval"][key]
            fallback_used.append(f"retrieval.{key}")
    for key, value in list(generation.items()):
        if value is None:
            generation[key] = FINAL_DEV_METRICS["generation"][key]
            fallback_used.append(f"generation.{key}")
    return {
        "retrieval": retrieval,
        "generation": generation,
        "fallback_used": fallback_used,
        "source_note": "development evaluation results",
        "generation_source": generation_source,
    }


def load_saved_examples(root: Path = ROOT) -> list[dict[str, Any]]:
    preferred = root / "reports/final_generation_bakeoff/experiments/GEN_MMR06_SUFFICIENCY_V1/raw_results.json"
    fallback = root / "reports/v2_sufficiency/dev_generation_sufficiency_raw_results.json"
    rows = read_json(preferred, [])
    if rows:
        return rows
    return read_json(fallback, [])


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def find_demo_answer(query: str, examples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not examples:
        return None
    query_norm = normalize_query(query)
    if query_norm:
        for row in examples:
            if query_norm == normalize_query(str(row.get("original_query", ""))):
                return row
        for row in examples:
            row_query = normalize_query(str(row.get("original_query", "")))
            if query_norm in row_query or row_query in query_norm:
                return row
    return examples[0]


def status_label(status: str | None) -> tuple[str, str]:
    mapping = {
        "sufficient": ("Sufficient evidence", "success"),
        "partially_sufficient": ("Partially sufficient", "warning"),
        "insufficient": ("Insufficient evidence", "error"),
    }
    return mapping.get(str(status or "").lower(), ("Sufficiency unknown", "info"))


def group_citations(citations: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for citation in citations or []:
        period = citation.get("report_period") or citation.get("report_id") or "Unknown report"
        grouped[str(period)].append(citation)
    return dict(sorted(grouped.items()))


SOURCE_RE = re.compile(
    r"\[SOURCE: (?P<label>.+?) \| page (?P<page>\d+) \| chunk (?P<chunk>[^\]]+)\]\s*(?P<text>.*?)(?=\n\[SOURCE:|\n## |\Z)",
    re.S,
)


def extract_context_snippets(source_labelled_context: str, max_snippets: int = 10) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for match in SOURCE_RE.finditer(source_labelled_context or ""):
        label = match.group("label")
        grouped[label].append(
            {
                "report_period": label,
                "page": match.group("page"),
                "chunk_id": match.group("chunk"),
                "text": " ".join(match.group("text").split())[:900],
            }
        )
        if sum(len(items) for items in grouped.values()) >= max_snippets:
            break
    return dict(grouped)


def compact_method_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    rows = read_json(root / "reports/final_comparison/rag_methods_master_comparison.json", [])
    keep = {
        "V2 Cohere retrieval",
        "True MMR lambda 0.6",
        "Final selected generation bake-off strategy",
        "V2 Cohere retrieval + sufficiency-gated generation",
    }
    output = []
    for row in rows:
        if row.get("method") in keep:
            output.append(
                {
                    "Method": row.get("method"),
                    "Scope": row.get("retrieval_or_generation"),
                    "CER": pct(row.get("complete_evidence_recall")) if row.get("complete_evidence_recall") is not None else "",
                    "Evidence Recall": pct(row.get("evidence_recall")) if row.get("evidence_recall") is not None else "",
                    "Factual": pct(row.get("factual_correctness")) if row.get("factual_correctness") is not None else "",
                    "Citation": pct(row.get("citation_correctness")) if row.get("citation_correctness") is not None else "",
                    "Status": row.get("status"),
                }
            )
    return output


def key_availability_status() -> dict[str, bool]:
    return {
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "cohere": bool(os.getenv("COHERE_API_KEY")),
    }


def caveats() -> list[str]:
    return [
        "Development-only evaluation results; not a held-out or production benchmark.",
        "Generation metrics are deterministic heuristic checks, not human evaluation.",
        "No fresh held-out final benchmark has been created for this V2 system.",
        "PyPDFLoader retained because it produced a valid evaluated corpus; Unstructured was attempted but blocked by OCR/Tesseract requirement.",
        "Cohere reranking improves retrieval quality but adds latency.",
    ]


def mrr_mmr_explanation() -> str:
    return "MRR = Mean Reciprocal Rank, a metric. MMR = Maximal Marginal Relevance, a selection method."


def production_status_text() -> str:
    return "Not production-ready; demo/interview-ready with known limitations."


def contains_production_overclaim(text: str) -> bool:
    lowered = (text or "").lower()
    return "production-ready" in lowered and "not production-ready" not in lowered
