from __future__ import annotations

import json

from rbi_rag.streamlit_demo_helpers import (
    caveats,
    contains_key_material,
    contains_production_overclaim,
    extract_context_snippets,
    find_demo_answer,
    group_citations,
    load_final_metrics,
    load_saved_examples,
    mrr_mmr_explanation,
    pct,
    score,
    status_label,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_metric_formatting():
    assert pct(0.5333) == "53.33%"
    assert pct(None) == "n/a"
    assert score(0.4054629) == "0.4055"


def test_final_metric_loading_uses_artifacts_and_fallback(tmp_path):
    write_json(
        tmp_path / "reports/mmr_experiments/experiments/MMR_LAMBDA_06/summary.json",
        {
            "complete_evidence_recall": 0.5,
            "all_reports_hit": 0.6,
            "evidence_recall": 0.7,
            "macro_report_mrr": 0.8,
        },
    )
    write_json(
        tmp_path / "reports/final_generation_bakeoff/experiments/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json",
        {"metrics": {"factual_correctness": {"mean_score": 0.9}}},
    )
    metrics = load_final_metrics(tmp_path)
    assert metrics["retrieval"]["macro_mrr"] == 0.8
    assert metrics["generation"]["factual_correctness"] == 0.9
    assert metrics["generation"]["citation_correctness"] == 0.8824
    assert "generation.citation_correctness" in metrics["fallback_used"]


def test_saved_example_loading_prefers_final_bakeoff(tmp_path):
    write_json(
        tmp_path / "reports/final_generation_bakeoff/experiments/GEN_MMR06_SUFFICIENCY_V1/raw_results.json",
        [{"question_id": "preferred", "original_query": "Preferred question"}],
    )
    write_json(
        tmp_path / "reports/v2_sufficiency/dev_generation_sufficiency_raw_results.json",
        [{"question_id": "fallback", "original_query": "Fallback question"}],
    )
    rows = load_saved_examples(tmp_path)
    assert rows[0]["question_id"] == "preferred"


def test_find_demo_answer_matches_exact_then_fuzzy():
    rows = [
        {"question_id": "a", "original_query": "Compare inflation outlook"},
        {"question_id": "b", "original_query": "Food inflation trend"},
    ]
    assert find_demo_answer("Food inflation trend", rows)["question_id"] == "b"
    assert find_demo_answer("inflation outlook", rows)["question_id"] == "a"


def test_citation_grouping_and_context_extraction():
    citations = [
        {"report_period": "April 2025", "page": 10, "chunk_id": "c1"},
        {"report_period": "April 2025", "page": 11, "chunk_id": "c2"},
        {"report_period": "October 2025", "page": 9, "chunk_id": "c3"},
    ]
    grouped = group_citations(citations)
    assert list(grouped) == ["April 2025", "October 2025"]
    assert len(grouped["April 2025"]) == 2
    context = "[SOURCE: April 2025 MPR | page 10 | chunk c1]\nInflation text.\n## Next"
    snippets = extract_context_snippets(context)
    assert snippets["April 2025 MPR"][0]["text"] == "Inflation text."


def test_sufficiency_badge_mapping():
    assert status_label("sufficient") == ("Sufficient evidence", "success")
    assert status_label("partially_sufficient") == ("Partially sufficient", "warning")
    assert status_label("insufficient") == ("Insufficient evidence", "error")
    assert status_label("other") == ("Sufficiency unknown", "info")


def test_caveats_include_required_wording():
    text = "\n".join(caveats())
    assert "Development-only evaluation results" in text
    assert "PyPDFLoader retained because it produced a valid evaluated corpus" in text
    assert "Unstructured was attempted but blocked by OCR/Tesseract requirement" in text


def test_no_api_key_serialization(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-test-key")
    assert contains_key_material({"safe": "value"}) is False
    assert contains_key_material({"leak": "secret-test-key"}) is True


def test_no_overclaim_text_and_mrr_mmr_explanation():
    assert contains_production_overclaim("This is production-ready") is True
    assert contains_production_overclaim("Not production-ready; demo/interview-ready") is False
    explanation = mrr_mmr_explanation()
    assert "MRR = Mean Reciprocal Rank" in explanation
    assert "MMR = Maximal Marginal Relevance" in explanation
