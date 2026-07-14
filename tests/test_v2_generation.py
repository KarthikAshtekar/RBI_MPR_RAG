import json
from pathlib import Path

import yaml

from rbi_rag.v2_generation_contexts import build_context_for_row, validate_selected_v2_config
from rbi_rag.v2_generation_evaluation import (
    PROMPT_VERSION,
    evaluate_generation_rows,
    generate_reports,
    parse_citations,
    retrieval_generation_analysis,
    run_generation_cases,
    summarise_metric_coverage,
    validate_generation_artifacts,
    write_analysis_outputs,
    write_eval_outputs,
    write_generation_outputs,
    write_pre_generation_checksums,
    write_status,
)


def _chunk(report_id="rbi_mpr_2025_04", period="April 2025", page=5, chunk_id=None, text="Inflation projection was 4.5 per cent."):
    chunk_id = chunk_id or f"{report_id}_p{page:03d}_c001"
    return {
        "chunk_id": chunk_id,
        "report_id": report_id,
        "report_period": period,
        "page": page,
        "text": text,
    }


def _retrieval_row(question_id="dev_q1"):
    first = _chunk("rbi_mpr_2025_04", "April 2025", 5, "rbi_mpr_2025_04_p005_c001")
    second = _chunk("rbi_mpr_2025_10", "October 2025", 6, "rbi_mpr_2025_10_p006_c001", "Growth projection changed in October 2025.")
    return {
        "question_id": question_id,
        "split": "dev",
        "query_type": "pairwise_comparison",
        "required_report_ids": ["rbi_mpr_2025_10", "rbi_mpr_2025_04"],
        "original_query": "Compare inflation and growth.",
        "normalised_query": "compare inflation and growth",
        "experiment_id": "V2_COHERE_ONLY",
        "configuration_checksum": "abc123",
        "reranker_provider": "cohere",
        "reranker_model": "rerank-v3.5",
        "dense_candidates_by_report": {
            "rbi_mpr_2025_04": [{"chunk_id": first["chunk_id"]}],
            "rbi_mpr_2025_10": [],
        },
        "bm25_candidates_by_report": {
            "rbi_mpr_2025_04": [],
            "rbi_mpr_2025_10": [{"chunk_id": second["chunk_id"]}],
        },
        "selected_chunks_by_report": {
            "rbi_mpr_2025_10": [second],
            "rbi_mpr_2025_04": [first],
        },
        "all_selected_chunks": [second, first],
        "selected_pages": {"rbi_mpr_2025_04": [5], "rbi_mpr_2025_10": [6]},
        "complete_evidence_recall": True,
        "evidence_recall": 1.0,
        "all_reports_hit": True,
        "macro_report_mrr": 0.75,
        "table_or_numeric_question": True,
    }


def _case():
    return {
        "question_id": "dev_q1",
        "question": "Compare inflation and growth.",
        "expected_answer": "Inflation projection was 4.5 per cent and growth projection changed in October 2025.",
        "source_information_type": ["table", "narrative"],
    }


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.answer


class BrokenGenerator:
    def __init__(self):
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        raise RuntimeError("fake failure without secrets")


class TwoAttemptGenerator:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_context_builder_uses_saved_chunks_only_and_preserves_source_labels():
    context = build_context_for_row(_retrieval_row())
    assert context["retrieval_experiment_id"] == "V2_COHERE_ONLY"
    assert context["selected_chunk_ids"] == ["rbi_mpr_2025_04_p005_c001", "rbi_mpr_2025_10_p006_c001"]
    assert "## April 2025" in context["source_labelled_context"]
    assert "## October 2025" in context["source_labelled_context"]
    assert "[SOURCE: April 2025 MPR | page 5 | chunk rbi_mpr_2025_04_p005_c001]" in context["source_labelled_context"]
    assert context["context_blocks"][0]["retriever_source"] == "dense"
    assert context["context_blocks"][1]["retriever_source"] == "bm25"


def test_selected_config_validation_rejects_old_or_wrong_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "v2_selected_retrieval.yaml").write_text(
        yaml.safe_dump({"id": "V2_COHERE_ONLY", "reranker": {"provider": "cohere"}, "parser": {"provider": "pypdfloader"}}),
        encoding="utf-8",
    )
    assert validate_selected_v2_config(tmp_path)["status"] == "passed"
    (config_dir / "v2_selected_retrieval.yaml").write_text(
        yaml.safe_dump({"id": "V2_BASELINE_FINAL", "reranker": {"provider": "local"}, "parser": {"provider": "pypdfloader"}}),
        encoding="utf-8",
    )
    result = validate_selected_v2_config(tmp_path)
    assert result["status"] == "failed"
    assert any("selected_config_id_is_not_V2_COHERE_ONLY" in issue for issue in result["issues"])


def test_generation_prompt_metadata_and_citation_subset():
    context = build_context_for_row(_retrieval_row())
    answer = (
        "Answer:\nApril 2025 says inflation projection was 4.5 per cent. October 2025 says growth changed.\n\n"
        "Citations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]\n"
        "- [October 2025, page 6, rbi_mpr_2025_10_p006_c001]"
    )
    generator = FakeGenerator(answer)
    rows = run_generation_cases([context], {"dev_q1": _case()}, generator=generator, model_name="fake-model", temperature=0.0, checkpoint_path=None)
    assert rows[0]["prompt_version"] == PROMPT_VERSION
    assert rows[0]["model_provider"] == "Groq"
    assert rows[0]["model_name"] == "fake-model"
    assert rows[0]["temperature"] == 0.0
    assert rows[0]["generation_success"] is True
    assert len(rows[0]["citations"]) == 2
    assert "Use only the supplied source-labelled context" in generator.prompts[0]
    bad = parse_citations("rbi_mpr_2026_04_p001_c001", context)
    assert bad == [{"chunk_id": "rbi_mpr_2026_04_p001_c001", "valid_supplied_chunk": False}]


def test_generation_reruns_invalid_citation_before_accepting():
    context = build_context_for_row(_retrieval_row())
    generator = TwoAttemptGenerator([
        "Answer:\nBad citation.\n\nCitations:\n- [October 2025, page 18, rbi_mpr_2025_10_p018_c000]",
        "Answer:\nValid citation.\n\nCitations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]",
    ])
    rows = run_generation_cases([context], {"dev_q1": _case()}, generator=generator, checkpoint_path=None, max_retries=2)
    assert len(generator.prompts) == 2
    assert "previous answer cited chunk IDs that were not supplied" in generator.prompts[1]
    assert rows[0]["generation_success"] is True
    assert rows[0]["generation_error_type"] is None
    assert rows[0]["generation_retry_warnings"][0]["type"] == "InvalidCitation"
    assert rows[0]["citations"][0]["chunk_id"] == "rbi_mpr_2025_04_p005_c001"


def test_generation_failure_records_error_and_null_metric_scores():
    context = build_context_for_row(_retrieval_row())
    generator = BrokenGenerator()
    rows = run_generation_cases([context], {"dev_q1": _case()}, generator=generator, checkpoint_path=None, max_retries=2)
    assert rows[0]["generation_success"] is False
    assert rows[0]["generation_error_type"] == "RuntimeError"
    assert generator.calls == 2
    eval_rows, _summary, coverage, failures = evaluate_generation_rows(rows, [_retrieval_row()], [context])
    assert failures[0]["stage"] == "generation"
    for metric in eval_rows[0]["metrics"].values():
        assert metric["score"] is None
    assert coverage["factual_correctness"]["failed_evaluations"] == 1


def test_temporal_citation_completeness_and_retrieval_to_generation_analysis():
    context = build_context_for_row(_retrieval_row())
    generator = FakeGenerator(
        "Answer:\nApril 2025 says inflation projection was 4.5 per cent.\n\n"
        "Citations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]"
    )
    rows = run_generation_cases([context], {"dev_q1": _case()}, generator=generator, checkpoint_path=None)
    eval_rows, summary, coverage, failures = evaluate_generation_rows(rows, [_retrieval_row()], [context])
    metrics = eval_rows[0]["metrics"]
    assert metrics["citation_correctness"]["score"] == 1.0
    assert metrics["citation_completeness"]["score"] == 0.5
    assert metrics["temporal_attribution_correctness"]["score"] == 1.0
    assert coverage["citation_completeness"]["successful_evaluations"] == 1
    assert summary["metrics"]["citation_correctness"]["mean_score"] == 1.0
    analysis_rows, analysis = retrieval_generation_analysis([_retrieval_row()], rows, eval_rows)
    assert analysis_rows[0]["question_id"] == "dev_q1"
    assert analysis["complete_retrieval_count"] == 1
    assert analysis["table_numeric_count"] == 1
    assert failures == []


def test_metric_coverage_counts_not_applicable_comparative_metric():
    retrieval = _retrieval_row("dev_q2")
    retrieval["query_type"] = "single_report"
    retrieval["required_report_ids"] = ["rbi_mpr_2025_04"]
    retrieval["selected_chunks_by_report"] = {"rbi_mpr_2025_04": [_chunk()]}
    context = build_context_for_row(retrieval)
    rows = run_generation_cases(
        [context],
        {"dev_q2": {"question": "What inflation projection?", "expected_answer": "Inflation projection was 4.5 per cent."}},
        generator=FakeGenerator("Answer:\nInflation projection was 4.5 per cent.\n\nCitations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]"),
        checkpoint_path=None,
    )
    eval_rows, _summary, coverage, _failures = evaluate_generation_rows(rows, [retrieval], [context])
    assert eval_rows[0]["metrics"]["comparative_correctness"]["applicable"] is False
    assert coverage["comparative_correctness"]["not_applicable"] == 1


def test_v2_generation_artifact_validation_reports_and_no_heldout(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/v2_selected_retrieval.yaml").write_text(
        yaml.safe_dump({"id": "V2_COHERE_ONLY", "reranker": {"provider": "cohere"}, "parser": {"provider": "pypdfloader"}}),
        encoding="utf-8",
    )
    exp_dir = tmp_path / "reports/v2_unstructured_cohere/experiments/V2_COHERE_ONLY"
    exp_dir.mkdir(parents=True)
    retrieval = _retrieval_row()
    for name in [
        "config_snapshot.yaml",
        "environment.json",
        "index_manifest.json",
        "question_results.csv",
        "report_level_results.csv",
        "summary.md",
        "stage_diagnostics.csv",
    ]:
        (exp_dir / name).write_text("placeholder\n", encoding="utf-8")
    (exp_dir / "raw_results.json").write_text(json.dumps([retrieval]), encoding="utf-8")
    (exp_dir / "integrity.json").write_text(json.dumps({"status": "valid"}), encoding="utf-8")
    (exp_dir / "summary.json").write_text(json.dumps({"experiment_id": "V2_COHERE_ONLY"}), encoding="utf-8")
    write_pre_generation_checksums(tmp_path)
    context = build_context_for_row(retrieval)
    (tmp_path / "reports/v2_generation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports/v2_generation/v2_generation_contexts.json").write_text(json.dumps([context]), encoding="utf-8")
    answer = (
        "Answer:\nApril 2025 says inflation projection was 4.5 per cent. October 2025 says growth changed.\n\n"
        "Citations:\n- [April 2025, page 5, rbi_mpr_2025_04_p005_c001]\n"
        "- [October 2025, page 6, rbi_mpr_2025_10_p006_c001]"
    )
    generation_rows = run_generation_cases([context], {"dev_q1": _case()}, generator=FakeGenerator(answer), checkpoint_path=None)
    generation_summary = write_generation_outputs(tmp_path, generation_rows)
    eval_rows, eval_summary, coverage, failures = evaluate_generation_rows(generation_rows, [retrieval], [context])
    write_eval_outputs(tmp_path, eval_rows, eval_summary, coverage, failures)
    analysis_rows, analysis = retrieval_generation_analysis([retrieval], generation_rows, eval_rows)
    write_analysis_outputs(tmp_path, analysis_rows, analysis)
    (tmp_path / "reports/v2_generation/environment_readiness.json").write_text(json.dumps({"groq_api_key_available": False, "cohere_api_key_available": True}), encoding="utf-8")
    (tmp_path / "reports/v2_generation/v2_retrieval_input_validation.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    write_status(tmp_path, generation_summary, {"generation_may_proceed": True})
    integrity = validate_generation_artifacts(tmp_path)
    assert integrity["status"] == "passed"
    assert integrity["heldout_generation_run"] is False
    paths = generate_reports(tmp_path)
    assert Path(paths["presentation_summary_path"]).exists()
    assert Path(paths["final_report_path"]).exists()
    assert "development generation only" in Path(paths["final_report_path"]).read_text(encoding="utf-8").lower()
