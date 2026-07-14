import json
from pathlib import Path

from rbi_rag.evidence_sufficiency import classify_evidence_sufficiency
from rbi_rag.v2_generation_contexts import build_context_for_row
from rbi_rag.v2_generation_evaluation import evaluate_generation_rows
from rbi_rag.v2_sufficiency import (
    PROMPT_VERSION,
    apply_sufficiency_postprocessing,
    build_sufficiency_prompt,
    comparison_outputs,
    run_sufficiency_generation_cases,
    validate_v2_sufficiency_artifacts,
    write_comparison_outputs,
    write_eval_outputs,
    write_generation_outputs,
    write_pre_sufficiency_checksums,
    write_status,
)


def _chunk(report_id="rbi_mpr_2025_04", period="April 2025", page=5, text="Inflation was 4.5 per cent."):
    return {
        "chunk_id": f"{report_id}_p{page:03d}_c001",
        "report_id": report_id,
        "report_period": period,
        "page": page,
        "text": text,
    }


def _row(*, complete=True, evidence=1.0, query_type="single_report", required=None, chunks=None, qid="dev_q1"):
    required = required or ["rbi_mpr_2025_04"]
    chunks = chunks if chunks is not None else {"rbi_mpr_2025_04": [_chunk()]}
    return {
        "question_id": qid,
        "split": "dev",
        "query_type": query_type,
        "required_report_ids": required,
        "original_query": "What was inflation?",
        "normalised_query": "what was inflation",
        "experiment_id": "V2_COHERE_ONLY",
        "configuration_checksum": "abc",
        "reranker_provider": "cohere",
        "reranker_model": "rerank-v3.5",
        "selected_chunks_by_report": chunks,
        "all_selected_chunks": [chunk for values in chunks.values() for chunk in values],
        "selected_pages": {report_id: [chunk.get("page") for chunk in values] for report_id, values in chunks.items()},
        "dense_candidates_by_report": {report_id: [] for report_id in required},
        "bm25_candidates_by_report": {report_id: [] for report_id in required},
        "complete_evidence_recall": complete,
        "evidence_recall": evidence,
        "all_reports_hit": bool(evidence),
        "report_coverage": 1.0 if all(chunks.get(report_id) for report_id in required) else 0.5,
        "table_or_numeric_question": True,
        "source_information_type": ["table"],
        "source_structure": "table",
    }


def _case():
    return {
        "question_id": "dev_q1",
        "question": "What was inflation?",
        "expected_answer": "Inflation was 4.5 per cent.",
        "source_information_type": ["table"],
    }


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def test_sufficiency_classifier_sufficient_case():
    row = _row()
    context = build_context_for_row(row)
    result = classify_evidence_sufficiency(row, context, _case())
    assert result["sufficiency_status"] == "sufficient"
    assert result["required_generation_behavior"] == "answer"
    assert result["sufficiency_reasons"] == []


def test_sufficiency_classifier_partial_and_incomplete_evidence():
    row = _row(complete=False, evidence=0.5)
    context = build_context_for_row(row)
    result = classify_evidence_sufficiency(row, context, _case())
    assert result["sufficiency_status"] == "partially_sufficient"
    assert result["required_generation_behavior"] == "answer_with_caveat"
    assert "incomplete_evidence" in result["sufficiency_reasons"]


def test_sufficiency_classifier_insufficient_missing_report():
    row = _row(
        complete=False,
        evidence=0.0,
        query_type="pairwise_comparison",
        required=["rbi_mpr_2025_04", "rbi_mpr_2025_10"],
        chunks={"rbi_mpr_2025_04": [_chunk()]},
    )
    context = build_context_for_row(row)
    result = classify_evidence_sufficiency(row, context, _case())
    assert result["sufficiency_status"] == "insufficient"
    assert result["required_generation_behavior"] == "abstain"
    assert "missing_required_report" in result["sufficiency_reasons"]
    assert "missing_comparative_counterpart" in result["sufficiency_reasons"]


def test_sufficiency_classifier_unsupported_period_abstains():
    row = _row(complete=None, evidence=None, query_type="unsupported_period", required=[], chunks={})
    context = build_context_for_row(row)
    result = classify_evidence_sufficiency(row, context, {})
    assert result["sufficiency_status"] == "insufficient"
    assert result["required_generation_behavior"] == "abstain"
    assert "unsupported_period" in result["sufficiency_reasons"]


def test_numeric_evidence_heuristic_marks_missing_numeric_evidence():
    row = _row(complete=False, evidence=0.7, chunks={"rbi_mpr_2025_04": [_chunk(text="Inflation discussion without values.")]})
    context = build_context_for_row(row)
    result = classify_evidence_sufficiency(row, context, _case())
    assert result["numeric_evidence_present"] is False
    assert "missing_numeric_evidence" in result["sufficiency_reasons"]


def test_prompt_behaviour_for_each_sufficiency_status():
    context = build_context_for_row(_row())
    sufficient = classify_evidence_sufficiency(_row(), context, _case())
    partial = classify_evidence_sufficiency(_row(complete=False, evidence=0.5), context, _case())
    insufficient = classify_evidence_sufficiency(_row(complete=False, evidence=0.0, chunks={}), context, _case())
    prompt_s = build_sufficiency_prompt("Q", "ctx", "single_report", ["April 2025"], sufficient)
    prompt_p = build_sufficiency_prompt("Q", "ctx", "single_report", ["April 2025"], partial)
    prompt_i = build_sufficiency_prompt("Q", "ctx", "single_report", ["April 2025"], insufficient)
    assert "Answer from the supplied context" in prompt_s
    assert "Answer only the parts supported" in prompt_p
    assert "Do not answer substantively" in prompt_i
    assert PROMPT_VERSION == "v2_sufficiency_prompt_v1"


def test_partial_answer_postprocessing_adds_measurable_caveat():
    row = _row(complete=False, evidence=0.5)
    context = build_context_for_row(row)
    classification = classify_evidence_sufficiency(row, context, _case())
    answer = "Answer:\nInflation was 4.5 per cent.\n\nCitations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]"
    processed = apply_sufficiency_postprocessing(answer, classification)
    assert "partially insufficient" in processed
    assert "cannot be determined" in processed


def test_sufficiency_generation_uses_contexts_and_enforces_citations():
    row = _row()
    context = build_context_for_row(row)
    classification = classify_evidence_sufficiency(row, context, _case())
    generator = FakeGenerator("Answer:\nInflation was 4.5 per cent.\n\nCitations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]")
    rows = run_sufficiency_generation_cases([context], {"dev_q1": _case()}, [classification], generator=generator, checkpoint_path=None)
    assert len(generator.prompts) == 1
    assert rows[0]["prompt_version"] == PROMPT_VERSION
    assert rows[0]["citations"][0]["valid_supplied_chunk"] is True


def test_insufficient_generation_is_hard_gated_without_external_call():
    row = _row(complete=False, evidence=0.0, query_type="unsupported_period", required=[], chunks={})
    context = build_context_for_row(row)
    classification = classify_evidence_sufficiency(row, context, {})
    generator = FakeGenerator("should not be called")
    rows = run_sufficiency_generation_cases([context], {"dev_q1": {}}, [classification], generator=generator, checkpoint_path=None)
    assert generator.prompts == []
    assert rows[0]["model_provider"] == "deterministic_sufficiency_gate"
    assert "supplied context is insufficient" in rows[0]["generated_answer"]


def test_metric_comparison_generation():
    row = _row()
    context = build_context_for_row(row)
    old_gen = [{
        "question_id": "dev_q1",
        "split": "dev",
        "query_type": "single_report",
        "required_report_ids": ["rbi_mpr_2025_04"],
        "generation_success": True,
        "generated_answer": "Answer:\nInflation was 4.5 per cent.\n\nCitations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]",
        "citations": [{"chunk_id": "rbi_mpr_2025_04_p005_c001", "valid_supplied_chunk": True}],
        "expected_answer": "Inflation was 4.5 per cent.",
    }]
    old_eval, *_ = evaluate_generation_rows(old_gen, [row], [context])
    new_gen = [{
        **old_gen[0],
        "generated_answer": "Answer:\nThe supplied context is insufficient.\n\nCitations:\n",
        "citations": [],
        "sufficiency_status": "insufficient",
        "required_generation_behavior": "abstain",
    }]
    new_eval, *_ = evaluate_generation_rows(new_gen, [row], [context])
    rows, summary = comparison_outputs([row], old_gen, old_eval, new_gen, new_eval)
    assert rows[0]["old_answered"] is True
    assert rows[0]["new_abstained"] is True
    assert "factual_correctness" in summary["metric_deltas"]


def test_sufficiency_artifact_validation_no_overwrite_and_no_key_serialization(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-test-value")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/v2_selected_retrieval.yaml").write_text("id: V2_COHERE_ONLY\n", encoding="utf-8")
    (tmp_path / "reports/v2_unstructured_cohere").mkdir(parents=True)
    (tmp_path / "reports/v2_generation").mkdir(parents=True)
    (tmp_path / "data/evaluation").mkdir(parents=True)
    (tmp_path / "reports/v2_generation/dev_generation_raw_results.json").write_text("[]\n", encoding="utf-8")
    write_pre_sufficiency_checksums(tmp_path)
    before = (tmp_path / "reports/v2_generation/dev_generation_raw_results.json").read_text(encoding="utf-8")
    out = tmp_path / "reports/v2_sufficiency"
    row = _row()
    context = build_context_for_row(row)
    classification = classify_evidence_sufficiency(row, context, _case())
    generation_rows = run_sufficiency_generation_cases(
        [context],
        {"dev_q1": _case()},
        [classification],
        generator=FakeGenerator("Answer:\nInflation was 4.5 per cent.\n\nCitations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]"),
        checkpoint_path=None,
    )
    gen_summary = write_generation_outputs(tmp_path, generation_rows)
    eval_rows, eval_summary, coverage, failures = evaluate_generation_rows(generation_rows, [row], [context])
    write_eval_outputs(tmp_path, eval_rows, eval_summary, coverage, failures)
    comp_rows, comp_summary = comparison_outputs([row], generation_rows, eval_rows, generation_rows, eval_rows)
    write_comparison_outputs(tmp_path, comp_rows, comp_summary)
    write_status(tmp_path, gen_summary)
    (out / "dev_sufficiency_classification.json").write_text(json.dumps([classification]), encoding="utf-8")
    (out / "sufficiency_results_for_presentation.md").write_text("no secrets\n", encoding="utf-8")
    assert (tmp_path / "reports/v2_generation/dev_generation_raw_results.json").read_text(encoding="utf-8") == before
    integrity = validate_v2_sufficiency_artifacts(tmp_path)
    assert integrity["status"] == "passed"
    assert integrity["old_v2_generation_artifacts_overwritten"] is False
