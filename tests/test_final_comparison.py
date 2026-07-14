import json

from rbi_rag.final_comparison import build_master_rows, generate_final_comparison, metric_lookup


def test_final_comparison_extracts_metrics_and_preserves_nulls():
    rows = build_master_rows()
    single_dense = metric_lookup(rows, "Single-document Dense")
    assert single_dense["hit_rate"] is not None
    assert single_dense["mrr"] is not None
    mmr = metric_lookup(rows, "MMR / diversity selection")
    assert mmr["status"] in {"not_run", "blocked", "evaluated", "evaluated_not_selected", "evaluated_selected"}


def test_mrr_and_mmr_terminology_distinction():
    rows = build_master_rows()
    mmr = metric_lookup(rows, "MMR / diversity selection")
    assert "Maximal Marginal Relevance" in mmr["notes"]
    v2 = metric_lookup(rows, "V2 Cohere retrieval")
    assert v2["macro_mrr"] is not None
    assert v2["mmr_enabled"] is False


def test_generation_comparison_rows_present_without_rerun_markers():
    rows = build_master_rows()
    old = metric_lookup(rows, "V2 Cohere retrieval + generation")
    new = metric_lookup(rows, "V2 Cohere retrieval + sufficiency-gated generation")
    assert old["factual_correctness"] is not None
    assert new["abstention_correctness"] == 1.0
    assert new["sufficiency_gate_enabled"] is True


def test_generate_final_comparison_artifacts_and_validation(tmp_path):
    # Uses repository saved artifacts as input but writes only under the requested report directory.
    result = generate_final_comparison()
    assert result["validation_status"] == "passed"
    validation = json.loads(open("reports/final_comparison/comparison_artifact_validation.json", encoding="utf-8").read())
    assert validation["mrr_and_mmr_distinguished"] is True
    assert validation["retrieval_rerun"] is False
    assert validation["generation_rerun"] is False


def test_final_comparison_maps_completed_unstructured_v2_rows(tmp_path):
    out = tmp_path / "reports/v2_unstructured_cohere"
    out.mkdir(parents=True)
    (out / "v2_experiment_leaderboard.json").write_text(json.dumps({
        "completed": [
            {
                "experiment_id": "V2_UNSTRUCTURED_ONLY",
                "parser_name": "Unstructured",
                "reranker_provider": "local_cross_encoder",
                "reranker_model": "cross-encoder/test",
                "complete_evidence_recall": 0.4,
                "all_reports_hit": 0.5,
                "evidence_recall": 0.6,
                "macro_report_mrr": 0.7,
                "report_coverage": 1.0,
                "single_report_contamination": 0.0,
                "median_latency_ms": 10,
                "mean_estimated_tokens": 100,
            },
            {
                "experiment_id": "V2_UNSTRUCTURED_COHERE",
                "parser_name": "Unstructured",
                "reranker_provider": "cohere",
                "reranker_model": "rerank-v3.5",
                "complete_evidence_recall": 0.5,
                "all_reports_hit": 0.6,
                "evidence_recall": 0.7,
                "macro_report_mrr": 0.8,
                "report_coverage": 1.0,
                "single_report_contamination": 0.0,
                "median_latency_ms": 20,
                "mean_estimated_tokens": 110,
            },
        ],
        "skipped": [],
    }), encoding="utf-8")
    rows = build_master_rows(tmp_path)
    assert metric_lookup(rows, "V2 Unstructured retrieval")["complete_evidence_recall"] == 0.4
    assert metric_lookup(rows, "V2 Unstructured + Cohere retrieval")["macro_mrr"] == 0.8


def test_final_comparison_maps_true_mmr_rows(tmp_path):
    out = tmp_path / "reports/mmr_experiments"
    out.mkdir(parents=True)
    (out / "mmr_selection_decision.json").write_text(json.dumps({
        "status": "evaluated_not_selected",
        "selected_experiment_id": "V2_COHERE_ONLY",
        "reason": "MMR did not improve evidence completeness.",
    }), encoding="utf-8")
    (out / "mmr_leaderboard.json").write_text(json.dumps({
        "completed": [
            {
                "experiment_id": "MMR_LAMBDA_07",
                "mmr_enabled": True,
                "mmr_lambda": 0.7,
                "complete_evidence_recall": 0.5,
                "all_reports_hit": 0.6,
                "evidence_recall": 0.7,
                "macro_report_mrr": 0.8,
                "report_coverage": 1.0,
                "single_report_contamination": 0.0,
                "median_latency_ms": 10,
                "mean_estimated_tokens": 100,
            }
        ],
        "skipped": [],
    }), encoding="utf-8")
    rows = build_master_rows(tmp_path)
    overview = metric_lookup(rows, "MMR / diversity selection")
    mmr = metric_lookup(rows, "True MMR lambda 0.7")
    assert overview["status"] == "evaluated_not_selected"
    assert mmr["mmr_enabled"] is True
    assert mmr["macro_mrr"] == 0.8
