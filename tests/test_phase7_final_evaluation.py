from pathlib import Path

from rbi_rag.final_evaluation import (
    assert_config_not_mutated,
    build_dev_vs_heldout_comparison,
    contains_groq_secret,
    decide_generation_readiness,
    deterministic_generation_metrics,
    ensure_one_time_guard,
    metric_result,
    recompute_retrieval_metrics,
    summarise_metric_coverage,
    temporal_attribution_failure_wrong_report,
    validate_citation_references,
    validate_final_retrieval_config,
    validate_final_status,
    validate_generation_integrity,
    validate_heldout_dataset_manifest,
    validate_heldout_raw_schema,
)


def retrieval_row(qid="q1", split="test", report_id="r1", query_type="single_report"):
    return {
        "question_id": qid,
        "split": split,
        "query_type": query_type,
        "required_report_ids": [report_id],
        "original_query": "What was stated?",
        "normalised_query": "What was stated?",
        "expanded_queries": [],
        "facet_queries": [],
        "retrieved_dense_candidates_by_report": {report_id: []},
        "retrieved_bm25_candidates_by_report": {report_id: []},
        "candidate_union_by_report": {report_id: []},
        "rrf_candidates_by_report": {report_id: []},
        "reranker_input_by_report": {report_id: []},
        "reranker_output_by_report": {report_id: []},
        "selected_chunks_by_report": {
            report_id: [{
                "chunk_id": f"{report_id}_c1",
                "report_id": report_id,
                "report_period": "Period",
                "page": 2,
                "text": "The exact evidence is here.",
            }]
        },
        "all_selected_chunks": [{
            "chunk_id": f"{report_id}_c1",
            "report_id": report_id,
            "report_period": "Period",
            "page": 2,
            "text": "The exact evidence is here.",
        }],
        "selected_pages": {report_id: [2]},
        "accepted_pages": {report_id: [2]},
        "expected_evidence": {report_id: ["exact evidence"]},
        "loss_stage": {report_id: "evidence_found"},
        "loss_stage_by_report": {report_id: "evidence_found"},
        "report_coverage": 1.0,
        "all_reports_hit": True,
        "evidence_recall": 1.0,
        "complete_evidence_recall": True,
        "macro_report_mrr": 1.0,
        "single_report_contamination": False,
        "latency_by_stage": {},
        "routing_latency_ms": 1.0,
        "query_transformation_latency_ms": 1.0,
        "dense_latency_ms": 1.0,
        "bm25_latency_ms": 1.0,
        "candidate_union_latency_ms": 1.0,
        "fusion_latency_ms": 1.0,
        "reranking_latency_ms": 1.0,
        "selection_latency_ms": 1.0,
        "deduplication_latency_ms": 1.0,
        "context_construction_latency_ms": 1.0,
        "total_retrieval_latency_ms": 10.0,
        "total_latency": 10.0,
        "selected_character_count": 27,
        "estimated_token_count": 7,
        "selected_chunk_count": 1,
        "unique_page_count": 1,
        "repeated_text_ratio": 0.0,
        "warnings": [],
        "per_report": {
            report_id: {
                "dense_candidate_ids": [],
                "dense_candidate_pages": [],
                "bm25_candidate_ids": [],
                "bm25_candidate_pages": [],
                "candidate_union_ids": [],
                "candidate_union_pages": [],
                "rrf_candidate_ids": [],
                "rrf_candidate_pages": [],
                "reranker_input_ids": [],
                "reranker_input_pages": [],
                "reranker_output_ids": [],
                "reranker_output_pages": [],
                "selected_chunk_ids_after_dedup": [f"{report_id}_c1"],
                "selected_pages": [2],
                "accepted_pages": [2],
                "expected_evidence": ["exact evidence"],
                "loss_stage": "evidence_found",
            }
        },
    }


def generation_row(split="test"):
    return {
        "question_id": "g1",
        "split": split,
        "query_type": "single_report",
        "required_report_ids": ["r1"],
        "query_plan": {},
        "retrieved_contexts": ["ctx"],
        "selected_chunk_ids": ["c1"],
        "selected_pages": [2],
        "report_coverage": 1.0,
        "expected_answer": "answer",
        "generated_answer": "answer",
        "citations": [{"chunk_id": "c1", "report_id": "r1", "page": 2}],
        "citation_chunk_ids": ["c1"],
        "citation_pages": [2],
        "prompt_version": "comparative_v1",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0,
        "generation_latency_ms": 1.0,
        "evaluation_metric_results": {
            "citation_correctness": metric_result(1.0, success=True),
            "judge_failed": metric_result(None, success=False, error_type="RuntimeError"),
        },
    }


def test_final_retrieval_config_and_heldout_dataset_validate():
    assert validate_final_retrieval_config()["status"] == "passed"
    dataset = validate_heldout_dataset_manifest()
    assert dataset["matches"]
    assert dataset["actual_verified_scored_count"] == dataset["manifest_verified_scored_count"]


def test_one_time_guard_and_config_mutation_check(tmp_path):
    ensure_one_time_guard(tmp_path, confirm=True)
    (tmp_path / "heldout_retrieval_raw_results.json").write_text("[]", encoding="utf-8")
    try:
        ensure_one_time_guard(tmp_path, confirm=True)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("guard should refuse a completed held-out output")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("a: 1\n", encoding="utf-8")
    digest = "bad"
    assert not assert_config_not_mutated(cfg, digest)["matches"]


def test_heldout_raw_schema_and_metric_recomputation():
    row = retrieval_row()
    assert validate_heldout_raw_schema([row]) == []
    metrics = recompute_retrieval_metrics([row])
    assert metrics["complete_evidence_recall"] == 1.0
    assert metrics["all_reports_hit"] == 1.0
    assert metrics["macro_report_mrr"] == 1.0


def test_dev_vs_heldout_comparison_reports_direction():
    dev = retrieval_row(qid="dev", report_id="r1")
    heldout = retrieval_row(qid="test", report_id="r1")
    heldout["total_retrieval_latency_ms"] = 20.0
    rows = build_dev_vs_heldout_comparison([dev], [heldout])
    latency = next(row for row in rows if row["metric"] == "mean_latency_ms")
    assert latency["absolute_difference_heldout_minus_dev"] == 10.0


def test_generation_readiness_gate_requires_all_conditions():
    readiness = decide_generation_readiness(
        heldout_completed=True,
        heldout_integrity={"status": "passed"},
        config_not_mutated=True,
        groq_available=True,
        frozen_generation_supported=False,
        retrieval_tuning_after_heldout=False,
    )
    assert readiness["status"] == "not_ready"


def test_generation_metric_coverage_failed_scores_and_citations():
    row = generation_row()
    assert validate_citation_references(row)
    coverage = summarise_metric_coverage([row])
    assert coverage["judge_failed"]["failed_evaluations"] == 1
    assert row["evaluation_metric_results"]["judge_failed"]["score"] is None
    assert validate_generation_integrity([row], expected_split="test")["status"] == "passed"


def test_generation_metadata_and_wrong_report_temporal_failure():
    row = generation_row()
    assert deterministic_generation_metrics(row)["citation_correctness"]["score"] == 1.0
    row["citations"] = [{"chunk_id": "c1", "report_id": "wrong", "page": 2}]
    assert temporal_attribution_failure_wrong_report(row)
    bad = generation_row(split="dev")
    assert validate_generation_integrity([bad], expected_split="test")["status"] == "failed"


def test_final_status_values_and_secret_detection():
    assert validate_final_status("heldout_retrieval_complete_generation_not_run")
    assert not validate_final_status("done")
    assert contains_groq_secret({"token": "gsk_test"})
    assert contains_groq_secret({"env": "GROQ_API_KEY=redacted"})

