from __future__ import annotations

import json

from langchain_core.documents import Document

from rbi_rag.mmr_selection import (
    Candidate,
    build_mmr_row,
    mmr_select,
    paired_bootstrap_diff,
    quota_for_row,
    validate_raw_rows,
)


class FakeSimilarity:
    provider = "fake"
    model_name = "fake"
    error = None

    def similarity_matrix(self, texts):
        return [
            [1.0, 0.95, 0.05],
            [0.95, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ][: len(texts)]


def test_mmr_select_uses_relevance_minus_similarity():
    candidates = [
        Candidate("a", "r1", 1, "same topic", 0.9, 1),
        Candidate("b", "r1", 2, "same topic duplicate", 0.85, 2),
        Candidate("c", "r1", 3, "different evidence", 0.84, 3),
    ]
    selected, trace = mmr_select(
        candidates,
        quota=2,
        lambda_value=0.6,
        similarity_matrix=[
            [1.0, 0.95, 0.05],
            [0.95, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ],
    )
    assert [item.chunk_id for item in selected] == ["a", "c"]
    assert all("mmr_score" in row for row in trace)
    assert any(row["chunk_id"] == "b" and row["selected"] is False for row in trace)


def test_build_mmr_row_preserves_required_reports_and_fields():
    docs = {
        "r1_c1": Document(page_content="inflation declined to 4 percent", metadata={"chunk_id": "r1_c1", "report_id": "r1", "report_period": "April", "page": 1}),
        "r1_c2": Document(page_content="inflation declined to 4 percent again", metadata={"chunk_id": "r1_c2", "report_id": "r1", "report_period": "April", "page": 2}),
        "r1_c3": Document(page_content="growth risk discussion", metadata={"chunk_id": "r1_c3", "report_id": "r1", "report_period": "April", "page": 3}),
    }
    source_row = {
        "question_id": "q1",
        "split": "dev",
        "query_type": "single_report",
        "required_report_ids": ["r1"],
        "original_query": "What happened to inflation?",
        "normalised_query": "What happened to inflation?",
        "accepted_pages": {"r1": [1]},
        "expected_evidence": {"r1": ["inflation declined to 4 percent"]},
        "selected_chunks_by_report": {"r1": [{"chunk_id": "r1_c1"}]},
        "reranker_output_by_report": {
            "r1": [
                {"chunk_id": "r1_c1", "page": 1, "score": 0.9, "rank": 1},
                {"chunk_id": "r1_c2", "page": 2, "score": 0.8, "rank": 2},
                {"chunk_id": "r1_c3", "page": 3, "score": 0.7, "rank": 3},
            ]
        },
        "latency_by_stage": {"rerank": 1.0},
        "total_latency_ms": 10.0,
    }
    row, trace = build_mmr_row(
        source_row,
        experiment_id="MMR_LAMBDA_06",
        mmr_enabled=True,
        mmr_lambda=0.6,
        chunk_lookup=docs,
        similarity_engine=FakeSimilarity(),
    )
    assert row["mmr_enabled"] is True
    assert row["mmr_lambda"] == 0.6
    assert row["retrieval_complete_evidence_recall"] is True
    assert set(row["selected_chunks_by_report"]) == {"r1"}
    assert trace
    assert validate_raw_rows(row and [row], expected_config={"mmr_enabled": True, "mmr_lambda": 0.6}) == []


def test_quota_for_query_types():
    assert quota_for_row({"query_type": "single_report", "required_report_ids": ["r1"]}) == 6
    assert quota_for_row({"query_type": "pairwise_comparison", "required_report_ids": ["r1", "r2"]}) == 5
    assert quota_for_row({"query_type": "trend", "required_report_ids": ["r1", "r2", "r3"]}) == 4


def test_paired_bootstrap_is_deterministic():
    low1, high1 = paired_bootstrap_diff([0, 1, 1], [1, 1, 1], resamples=100, seed=42)
    low2, high2 = paired_bootstrap_diff([0, 1, 1], [1, 1, 1], resamples=100, seed=42)
    assert (low1, high1) == (low2, high2)
    assert high1 >= low1


def test_mmr_raw_validation_catches_heldout_and_missing_trace():
    row = {
        "question_id": "q_bad",
        "split": "heldout",
        "query_type": "single_report",
        "required_report_ids": ["r1"],
        "original_query": "x",
        "normalised_query": "x",
        "selected_chunks_by_report": {"r1": []},
        "selected_pages": {"r1": []},
        "accepted_pages": {"r1": []},
        "expected_evidence": {"r1": []},
        "retrieval_complete_evidence_recall": None,
        "retrieval_evidence_recall": None,
        "retrieval_all_reports_hit": None,
        "retrieval_macro_mrr": 0.0,
        "report_coverage": 1.0,
        "single_report_contamination": False,
        "loss_stage": {},
        "latency_by_stage": {},
        "total_latency_ms": 1.0,
        "estimated_token_count": 0,
        "unique_page_count": 0,
        "repeated_text_ratio": 0.0,
        "mmr_enabled": True,
        "mmr_lambda": 0.7,
        "mmr_selected_chunk_ids": [],
        "mmr_rejected_chunk_ids": [],
        "mmr_trace": [],
    }
    issues = validate_raw_rows([json.loads(json.dumps(row))], expected_config={"mmr_enabled": True, "mmr_lambda": 0.7})
    assert any("heldout_row_present" in issue for issue in issues)
    assert any("missing_mmr_trace" in issue for issue in issues)
