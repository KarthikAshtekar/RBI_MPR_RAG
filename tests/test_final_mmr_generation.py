from __future__ import annotations

import json

from rbi_rag.final_mmr_generation import (
    EXPERIMENT_ID,
    RETRIEVAL_ID,
    build_context_for_mmr_row,
    validate_final_mmr_generation,
    write_generation_comparison,
    write_selection_decision,
)


def sample_mmr_row() -> dict:
    return {
        "question_id": "q1",
        "split": "dev",
        "query_type": "single_report",
        "required_report_ids": ["rbi_mpr_2025_10"],
        "original_query": "What happened to food inflation?",
        "normalised_query": "what happened to food inflation",
        "configuration_checksum": "abc",
        "retrieval_complete_evidence_recall": True,
        "retrieval_evidence_recall": 1.0,
        "retrieval_all_reports_hit": True,
        "retrieval_macro_mrr": 1.0,
        "report_coverage": 1.0,
        "single_report_contamination": False,
        "repeated_text_ratio": 0.0,
        "selected_chunks_by_report": {
            "rbi_mpr_2025_10": [
                {
                    "chunk_id": "rbi_mpr_2025_10_p010_c000",
                    "report_id": "rbi_mpr_2025_10",
                    "report_period": "October 2025",
                    "page": 10,
                    "text": "Food inflation turned negative in June and July 2025.",
                }
            ],
            "rbi_mpr_2026_04": [
                {
                    "chunk_id": "rbi_mpr_2026_04_p010_c000",
                    "report_id": "rbi_mpr_2026_04",
                    "report_period": "April 2026",
                    "page": 10,
                    "text": "This non-required report must not appear.",
                }
            ],
        },
        "mmr_trace": [
            {
                "chunk_id": "rbi_mpr_2025_10_p010_c000",
                "report_id": "rbi_mpr_2025_10",
                "page_number": 10,
                "reranker_score": 0.9,
                "normalised_relevance_score": 1.0,
                "max_similarity_to_selected": 0.0,
                "mmr_score": 0.6,
                "selected": True,
                "selection_rank": 1,
                "rejection_reason": None,
            }
        ],
    }


def test_mmr_context_construction_preserves_source_schema_and_required_reports():
    context = build_context_for_mmr_row(sample_mmr_row())
    assert context["retrieval_experiment_id"] == RETRIEVAL_ID
    assert context["context_ordering"] == "page_order"
    assert context["selected_chunk_ids"] == ["rbi_mpr_2025_10_p010_c000"]
    assert "April 2026" not in context["source_labelled_context"]
    assert "[SOURCE: October 2025 MPR | page 10 | chunk rbi_mpr_2025_10_p010_c000]" in context["source_labelled_context"]
    assert context["retrieval_complete_evidence_recall"] is True


def test_mmr_context_rerank_order_uses_mmr_selection_rank():
    row = sample_mmr_row()
    row["selected_chunks_by_report"]["rbi_mpr_2025_10"].append(
        {
            "chunk_id": "rbi_mpr_2025_10_p009_c000",
            "report_id": "rbi_mpr_2025_10",
            "report_period": "October 2025",
            "page": 9,
            "text": "Second selected chunk.",
        }
    )
    row["mmr_trace"].append(
        {
            "chunk_id": "rbi_mpr_2025_10_p009_c000",
            "report_id": "rbi_mpr_2025_10",
            "page_number": 9,
            "reranker_score": 0.8,
            "normalised_relevance_score": 0.8,
            "max_similarity_to_selected": 0.0,
            "mmr_score": 0.5,
            "selected": True,
            "selection_rank": 2,
            "rejection_reason": None,
        }
    )
    context = build_context_for_mmr_row(row, ordering="rerank_order")
    assert context["selected_chunk_ids"] == ["rbi_mpr_2025_10_p010_c000", "rbi_mpr_2025_10_p009_c000"]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_generation_comparison_and_selection_decision(tmp_path):
    write_json(
        tmp_path / "reports/v2_sufficiency/dev_sufficiency_eval_summary.json",
        {"metrics": {name: {"mean_score": 0.5, "successful_count": 1} for name in [
            "factual_correctness",
            "faithfulness_to_context",
            "contextual_relevancy",
            "contextual_recall",
            "abstention_correctness",
            "citation_correctness",
            "citation_completeness",
            "temporal_attribution_correctness",
            "comparative_correctness",
        ]}},
    )
    write_json(tmp_path / "reports/v2_sufficiency/dev_generation_sufficiency_summary.json", {"median_generation_latency_ms": 10})
    write_json(
        tmp_path / "reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json",
        {"metrics": {
            "factual_correctness": {"mean_score": 0.6, "successful_count": 1},
            "faithfulness_to_context": {"mean_score": 0.5, "successful_count": 1},
            "contextual_relevancy": {"mean_score": 0.5, "successful_count": 1},
            "contextual_recall": {"mean_score": 0.5, "successful_count": 1},
            "abstention_correctness": {"mean_score": 0.5, "successful_count": 1},
            "citation_correctness": {"mean_score": 0.5, "successful_count": 1},
            "citation_completeness": {"mean_score": 0.5, "successful_count": 1},
            "temporal_attribution_correctness": {"mean_score": 0.7, "successful_count": 1},
            "comparative_correctness": {"mean_score": 0.5, "successful_count": 1},
        }},
    )
    write_json(tmp_path / "reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/summary.json", {"median_generation_latency_ms": 12})
    write_json(tmp_path / "reports/mmr_experiments/experiments/MMR_LAMBDA_06/summary.json", {"experiment_id": "MMR_LAMBDA_06", "complete_evidence_recall": 0.5})
    write_json(tmp_path / "reports/mmr_experiments/experiments/MMR_BASELINE_V2_COHERE/summary.json", {"experiment_id": "MMR_BASELINE_V2_COHERE", "complete_evidence_recall": 0.4})
    rows, delta = write_generation_comparison(tmp_path)
    decision = write_selection_decision(tmp_path, rows)
    assert delta["factual_correctness"] == 0.09999999999999998
    assert decision["status"] == "selected_mmr_end_to_end"


def test_final_mmr_validation_catches_bad_citation(tmp_path):
    out = tmp_path / "reports/final_mmr_generation"
    write_json(
        out / "mmr06_source_labelled_contexts.json",
        [
            {
                "question_id": "q1",
                "split": "dev",
                "retrieval_experiment_id": RETRIEVAL_ID,
                "required_report_ids": ["r1"],
                "selected_chunk_ids": ["c1"],
                "context_blocks": [{"report_id": "r1", "chunk_id": "c1"}],
            }
        ],
    )
    write_json(
        out / EXPERIMENT_ID / "raw_results.json",
        [
            {
                "question_id": "q1",
                "retrieval_experiment_id": RETRIEVAL_ID,
                "prompt_version": "v2_sufficiency_prompt_v1",
                "model_name": "llama-3.1-8b-instant",
                "temperature": 0.0,
                "required_generation_behavior": "answer",
                "citations": [{"chunk_id": "bad"}],
            }
        ],
    )
    write_json(out / EXPERIMENT_ID / "eval_raw_results.json", [{"question_id": "q1", "metrics": {}}])
    write_json(out / EXPERIMENT_ID / "metric_coverage.json", {"factual_correctness": {"coverage": 1.0}})
    (tmp_path / "reports/final_comparison").mkdir(parents=True)
    (tmp_path / "reports/final_comparison/rag_methods_master_comparison.md").write_text(
        "MRR = Mean Reciprocal Rank. MMR = Maximal Marginal Relevance. fresh V2 benchmark",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("not production-ready", encoding="utf-8")
    validation = validate_final_mmr_generation(tmp_path)
    assert validation["status"] == "failed"
    assert any("citation_not_in_supplied_context" in issue for issue in validation["issues"])
