from __future__ import annotations

import json

from rbi_rag.final_generation_bakeoff import (
    METRIC_NAMES,
    PREVIOUS_BEST_VARIANT,
    build_context_from_retrieval_row,
    build_variant_prompt,
    citation_repair,
    eligibility_status,
    evaluate_variant,
    run_generation_cases_for_variant,
    write_context_artifacts,
    write_generation_summary,
    write_selection_decision,
    write_sufficiency_artifacts,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sample_retrieval_row(experiment_id: str = "MMR_LAMBDA_06") -> dict:
    return {
        "question_id": "dev_q1",
        "split": "dev",
        "query_type": "single_report",
        "required_report_ids": ["rbi_mpr_2025_10"],
        "original_query": "What was the food inflation trend?",
        "normalised_query": "food inflation trend",
        "experiment_id": experiment_id,
        "configuration_checksum": "abc",
        "complete_evidence_recall": True,
        "evidence_recall": 1.0,
        "all_reports_hit": True,
        "macro_report_mrr": 1.0,
        "report_coverage": 1.0,
        "single_report_contamination": 0.0,
        "repeated_text_ratio": 0.0,
        "selected_chunks_by_report": {
            "rbi_mpr_2025_10": [
                {
                    "chunk_id": "rbi_mpr_2025_10_p010_c001",
                    "report_id": "rbi_mpr_2025_10",
                    "report_period": "October 2025",
                    "page": 10,
                    "text": "Food inflation softened in October 2025.",
                }
            ]
        },
        "mmr_trace": [
            {"chunk_id": "rbi_mpr_2025_10_p010_c001", "selection_rank": 1, "reranker_score": 0.9}
        ],
    }


def test_bakeoff_context_schema_preserves_retrieval_and_ordering():
    context = build_context_from_retrieval_row(
        sample_retrieval_row("MMR_LAMBDA_07"),
        retrieval_experiment_id="MMR_LAMBDA_07",
        ordering="rerank_order",
    )
    assert context["retrieval_experiment_id"] == "MMR_LAMBDA_07"
    assert context["context_ordering"] == "rerank_order"
    assert context["selected_chunk_ids"] == ["rbi_mpr_2025_10_p010_c001"]
    assert context["retrieval_complete_evidence_recall"] is True
    assert "October 2025 MPR | page 10" in context["source_labelled_context"]


def test_bakeoff_prompt_variants_add_constraints():
    prompt = build_variant_prompt(
        prompt_version="comparative_strict_prompt_v1",
        question="Compare reports",
        context="context",
        query_type="pairwise_comparison",
        required_periods_value=["April 2025", "October 2025"],
        sufficiency={
            "sufficiency_status": "sufficient",
            "required_generation_behavior": "answer",
            "sufficiency_reasons": [],
        },
    )
    assert "mini-conclusion" in prompt
    assert "Do not make cross-report claims" in prompt


class FakeGenerator:
    def invoke(self, prompt: str) -> str:
        assert "Use only the supplied source-labelled context" in prompt
        return (
            "Answer:\nFood inflation softened in October 2025 "
            "(rbi_mpr_2025_10_p010_c001).\n\n"
            "Citations:\n- [October 2025, page 10, rbi_mpr_2025_10_p010_c001]"
        )


def test_bakeoff_generation_runner_uses_fake_generator_without_api(tmp_path):
    context = build_context_from_retrieval_row(sample_retrieval_row(), retrieval_experiment_id="MMR_LAMBDA_06", ordering="page_order")
    classifications = [
        {
            "question_id": "dev_q1",
            "sufficiency_status": "sufficient",
            "sufficiency_reasons": [],
            "required_generation_behavior": "answer",
        }
    ]
    rows = run_generation_cases_for_variant(
        [context],
        {"dev_q1": {"question": "What was the trend?", "expected_answer": "Food inflation softened in October 2025."}},
        classifications,
        variant_id="GEN_MMR06_SUFFICIENCY_V1",
        prompt_version="v2_sufficiency_prompt_v1",
        generator=FakeGenerator(),
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert rows[0]["generation_success"] is True
    assert rows[0]["generation_experiment_id"] == "GEN_MMR06_SUFFICIENCY_V1"
    assert rows[0]["citations"][0]["valid_supplied_chunk"] is True


def test_bakeoff_citation_repair_removes_invalid_citation(tmp_path):
    context = build_context_from_retrieval_row(sample_retrieval_row(), retrieval_experiment_id="MMR_LAMBDA_06", ordering="page_order")
    write_context_artifacts(tmp_path, "GEN_MMR06_CITATION_REPAIR_V1", [context])
    eval_path = tmp_path / "data/evaluation/multi_report_dev.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(
        json.dumps({
            "question_id": "dev_q1",
            "question": "What was the food inflation trend?",
            "expected_answer": "Food inflation softened in October 2025.",
        }) + "\n",
        encoding="utf-8",
    )
    write_json(tmp_path / "reports/mmr_experiments/experiments/MMR_LAMBDA_06/raw_results.json", [sample_retrieval_row()])
    write_json(
        tmp_path / "reports/final_generation_bakeoff/experiments/GEN_MMR06_SUFFICIENCY_V1/raw_results.json",
        [
            {
                "question_id": "dev_q1",
                "split": "dev",
                "generation_success": True,
                "generated_answer": "Answer cites rbi_mpr_2025_10_p999_c999 and rbi_mpr_2025_10_p010_c001.",
                "citations": [
                    {"chunk_id": "rbi_mpr_2025_10_p999_c999"},
                    {"chunk_id": "rbi_mpr_2025_10_p010_c001"},
                ],
            }
        ],
    )
    result = citation_repair(tmp_path)
    repaired = json.loads(
        (tmp_path / "reports/final_generation_bakeoff/experiments/GEN_MMR06_CITATION_REPAIR_V1/raw_results.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "completed_repaired"
    assert "rbi_mpr_2025_10_p999_c999" not in repaired[0]["generated_answer"]
    assert repaired[0]["citations"][0]["chunk_id"] == "rbi_mpr_2025_10_p010_c001"


def test_bakeoff_eligibility_and_selection_policy_keeps_baseline_when_best():
    baseline = {
        "factual_correctness": 0.8,
        "citation_correctness": 0.9,
        "temporal_attribution_correctness": 0.9,
        "abstention_correctness": 1.0,
    }
    row = {
        "status": "completed",
        "success_rate": 1.0,
        "factual_correctness": 0.79,
        "citation_correctness": 0.89,
        "temporal_attribution_correctness": 0.9,
        "abstention_correctness": 1.0,
    }
    status, reasons = eligibility_status(row, baseline)
    assert status == "not_eligible"
    assert "citation_not_preserved_without_material_factual_gain" in reasons


def test_bakeoff_evaluate_variant_writes_nulls_for_failed_metrics(tmp_path):
    row = sample_retrieval_row()
    context = build_context_from_retrieval_row(row, retrieval_experiment_id="MMR_LAMBDA_06", ordering="page_order")
    write_json(tmp_path / "reports/mmr_experiments/experiments/MMR_LAMBDA_06/raw_results.json", [row])
    out = tmp_path / "reports/final_generation_bakeoff/experiments" / PREVIOUS_BEST_VARIANT
    write_json(out / "context_records.json", [context])
    write_sufficiency_artifacts(
        tmp_path,
        PREVIOUS_BEST_VARIANT,
        [row],
        [context],
    )
    gen = {
        "question_id": "dev_q1",
        "split": "dev",
        "query_type": "single_report",
        "required_report_ids": ["rbi_mpr_2025_10"],
        "retrieval_experiment_id": "MMR_LAMBDA_06",
        "generated_answer": "",
        "generation_success": False,
        "generation_error_type": "FakeError",
        "generation_error_message": "redacted",
        "citations": [],
        "expected_answer": "Food inflation softened in October 2025.",
    }
    write_json(out / "raw_results.json", [gen])
    summary = evaluate_variant(tmp_path, PREVIOUS_BEST_VARIANT)
    eval_rows = json.loads((out / "eval_raw_results.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    for name in METRIC_NAMES:
        metric = eval_rows[0]["metrics"][name]
        assert metric["success"] is False
        assert metric["score"] is None


def test_bakeoff_selection_decision_prefers_eligible_highest_rank(tmp_path):
    rows = [
        {
            "variant_id": PREVIOUS_BEST_VARIANT,
            "eligibility": "eligible",
            "retrieval_experiment_id": "MMR_LAMBDA_06",
            "factual_correctness": 0.8,
            "citation_correctness": 0.9,
            "temporal_attribution_correctness": 0.9,
            "abstention_correctness": 1.0,
            "comparative_correctness": 0.3,
            "contextual_recall": 0.5,
            "median_generation_latency_ms": 10,
        },
        {
            "variant_id": "GEN_MMR07_SUFFICIENCY_V1",
            "eligibility": "eligible",
            "retrieval_experiment_id": "MMR_LAMBDA_07",
            "factual_correctness": 0.85,
            "citation_correctness": 0.9,
            "temporal_attribution_correctness": 0.9,
            "abstention_correctness": 1.0,
            "comparative_correctness": 0.3,
            "contextual_recall": 0.5,
            "median_generation_latency_ms": 12,
        },
    ]
    decision = write_selection_decision(tmp_path, rows)
    assert decision["status"] == "selected_bakeoff_generation_strategy"
    assert decision["selected_variant_id"] == "GEN_MMR07_SUFFICIENCY_V1"
