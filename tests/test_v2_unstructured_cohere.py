import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.documents import Document

from rbi_rag.cohere_reranker import CohereRerankConfig, CohereRerankError, CohereReranker
from rbi_rag.env_loading import load_project_dotenv
from rbi_rag.unstructured_extraction import (
    chunk_unstructured_elements,
    element_to_record,
    map_content_type,
    unstructured_available,
)
from rbi_rag.poppler_setup import (
    find_poppler_bin,
    prepend_path,
    write_poppler_helper,
    apply_poppler_path_from_helper,
    setup_poppler,
)
from rbi_rag.v2_experiments import (
    CONTROLLED_EXPERIMENTS,
    _unstructured_index_manifest,
    contains_api_key_material,
    env_key_available,
    is_table_or_numeric,
    paired_bootstrap_diff,
    report_pair,
    select_v2,
    source_structure,
    validate_unstructured_extraction_records,
    validate_v2_raw_rows,
    validate_v2_registry,
    v2_category_results,
)


@dataclass
class FakeMetadata:
    page_number: int
    text_as_html: str | None = None


class FakeNarrativeText:
    def __init__(self, text, page=3):
        self.text = text
        self.metadata = FakeMetadata(page)


class FakeTable:
    def __init__(self, text, page=4, html="<table><tr><td>1</td></tr></table>"):
        self.text = text
        self.metadata = FakeMetadata(page, html)


def test_unstructured_element_schema_metadata_and_page_preservation():
    record = element_to_record(
        FakeNarrativeText("Policy stance text", page=7),
        report_id="rbi_mpr_2025_10",
        report_period="October 2025",
        source_file="Oct_2025_RBI_MPR.pdf",
        extraction_strategy="auto",
        parser_version="test",
    )
    assert record.parser_name == "unstructured"
    assert record.page_number == 7
    assert record.report_id == "rbi_mpr_2025_10"
    assert record.content_type == "narrative_text"
    assert record.text_length == len("Policy stance text")


def test_content_type_mapping_and_table_metadata_preservation():
    assert map_content_type("Title") == "title"
    assert map_content_type("FigureCaption") == "figure_caption"
    assert map_content_type("DoesNotExist") == "unknown"
    record = element_to_record(
        FakeTable("Table text", page=8),
        report_id="r",
        report_period="Period",
        source_file="report.pdf",
        extraction_strategy="auto",
        parser_version="test",
    )
    assert record.content_type == "table"
    assert record.table_html.startswith("<table>")
    assert record.table_text == "Table text"


def test_element_aware_chunking_does_not_merge_table_with_narrative():
    first = element_to_record(FakeNarrativeText("short narrative", page=1), report_id="r", report_period="P", source_file="r.pdf", extraction_strategy="auto")
    table = element_to_record(FakeTable("table value", page=1), report_id="r", report_period="P", source_file="r.pdf", extraction_strategy="auto")
    second = element_to_record(FakeNarrativeText("more narrative", page=1), report_id="r", report_period="P", source_file="r.pdf", extraction_strategy="auto")
    chunks = chunk_unstructured_elements([first, table, second], max_chars=1000)
    assert len(chunks) == 3
    assert chunks[1].metadata["content_type"] == "table"
    assert "table_html" in chunks[1].metadata


class FakeCohereClient:
    def __init__(self):
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}


def test_cohere_reranker_request_and_response_parsing():
    client = FakeCohereClient()
    reranker = CohereReranker(CohereRerankConfig(model="rerank-test", top_n=2), client=client, api_key="secret", sleep=lambda _: None)
    docs = [Document(page_content="a", metadata={"chunk_id": "a"}), Document(page_content="b", metadata={"chunk_id": "b"})]
    ranked, meta = reranker.rerank("query", docs)
    assert client.calls[0]["model"] == "rerank-test"
    assert client.calls[0]["query"] == "query"
    assert client.calls[0]["documents"] == ["a", "b"]
    assert [doc.metadata["chunk_id"] for doc, _ in ranked] == ["b", "a"]
    assert meta.reranker_api_success is True


def test_dotenv_loading_and_boolean_key_detection(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    (tmp_path / ".env").write_text("COHERE_API_KEY=test-value\n", encoding="utf-8")
    assert load_project_dotenv(tmp_path)
    assert env_key_available("COHERE_API_KEY", tmp_path)
    payload = {"cohere_api_key_available": True}
    assert not contains_api_key_material(payload)
    assert contains_api_key_material({"value": "COHERE_API_KEY"})


class BrokenCohereClient:
    def __init__(self):
        self.calls = 0

    def rerank(self, **_):
        self.calls += 1
        raise RuntimeError("secret rate limit")


def test_cohere_error_handling_retry_bounds_and_key_redaction():
    client = BrokenCohereClient()
    reranker = CohereReranker(CohereRerankConfig(max_retries=2), client=client, api_key="secret", sleep=lambda _: None)
    try:
        reranker.rerank("query", [Document(page_content="x")])
    except CohereRerankError as exc:
        assert exc.metadata.reranker_api_attempts == 2
        assert "[redacted]" in exc.metadata.reranker_error_message
        assert "secret" not in exc.metadata.reranker_error_message
    else:
        raise AssertionError("expected CohereRerankError")
    assert client.calls == 2


def test_optional_dependency_detection_functions_are_boolean():
    assert isinstance(unstructured_available(), bool)


def test_v2_registry_validation_and_no_final_config_overwrite():
    registry = yaml.safe_load(Path("configs/v2_unstructured_cohere_experiments.yaml").read_text())
    assert tuple(registry) == CONTROLLED_EXPERIMENTS
    assert validate_v2_registry(registry) == []
    assert Path("configs/final_retrieval_selected.yaml").exists()
    assert Path("configs/v2_unstructured_cohere_experiments.yaml") != Path("configs/final_retrieval_selected.yaml")


def minimal_v2_row():
    return {
        "question_id": "q",
        "split": "dev",
        "query_type": "single_report",
        "required_report_ids": ["r"],
        "original_query": "What projection was stated?",
        "normalised_query": "What projection was stated?",
        "expanded_queries": [],
        "facet_queries": [],
        "parser_name": "PyPDFLoader",
        "reranker_provider": "local_cross_encoder",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "retrieved_dense_candidates_by_report": {"r": []},
        "retrieved_bm25_candidates_by_report": {"r": []},
        "dense_candidates_by_report": {"r": []},
        "bm25_candidates_by_report": {"r": []},
        "candidate_union_by_report": {"r": []},
        "rrf_candidates_by_report": {"r": []},
        "reranker_input_by_report": {"r": []},
        "reranker_output_by_report": {"r": []},
        "selected_chunks_by_report": {"r": [{"chunk_id": "c", "report_id": "r", "page": 1, "text": "evidence"}]},
        "all_selected_chunks": [{"chunk_id": "c", "report_id": "r", "page": 1, "text": "evidence"}],
        "selected_pages": {"r": [1]},
        "accepted_pages": {"r": [1]},
        "expected_evidence": {"r": ["evidence"]},
        "loss_stage": {"r": "evidence_found"},
        "report_coverage": 1.0,
        "all_reports_hit": True,
        "evidence_recall": 1.0,
        "complete_evidence_recall": True,
        "macro_report_mrr": 1.0,
        "single_report_contamination": False,
        "latency_by_stage": {
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
        },
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
        "total_latency_ms": 10.0,
        "total_latency": 10.0,
        "selected_character_count": 8,
        "estimated_token_count": 2,
        "selected_chunk_count": 1,
        "unique_page_count": 1,
        "repeated_text_ratio": 0.0,
        "warnings": [],
        "errors": [],
        "per_report": {
            "r": {
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
                "selected_chunk_ids_after_dedup": ["c"],
                "selected_pages": [1],
                "accepted_pages": [1],
                "expected_evidence": ["evidence"],
                "loss_stage": "evidence_found",
            }
        },
    }


def test_v2_raw_schema_category_helpers_and_bootstrap():
    row = minimal_v2_row()
    assert validate_v2_raw_rows([row]) == []
    case = {"question": "What 6.5 per cent projection?", "source_information_type": ["table"], "expected_answer": "6.5 per cent"}
    assert source_structure(case) == "table"
    assert is_table_or_numeric(case)
    assert report_pair(["rbi_mpr_2025_04", "rbi_mpr_2025_10"]) == "April 2025 vs October 2025"
    low, high = paired_bootstrap_diff([1.0, 0.0, -1.0], resamples=50, seed=42)
    assert low <= high


def test_v2_category_aggregation_selection_policy_and_no_key_serialization(tmp_path):
    row = minimal_v2_row()
    row.update({"question_structure": "single_facet", "source_structure": "narrative", "topic": "inflation", "report_pair": "April", "table_or_numeric_question": True})
    categories = v2_category_results({"V2_BASELINE_FINAL": [row]})
    assert any(item["category_type"] == "table_or_numeric_questions" for item in categories)
    registry = yaml.safe_load(Path("configs/v2_unstructured_cohere_experiments.yaml").read_text())
    selected = select_v2(tmp_path, [{
        "experiment_id": "V2_BASELINE_FINAL",
        "complete_evidence_recall": 1.0,
        "all_reports_hit": 1.0,
        "evidence_recall": 1.0,
        "macro_report_mrr": 1.0,
        "report_coverage": 1.0,
        "single_report_contamination": 0.0,
        "median_latency_ms": 1.0,
        "mean_estimated_tokens": 10.0,
    }], registry)
    assert selected["selected_experiment_id"] == "V2_BASELINE_FINAL"
    assert (tmp_path / "configs/v2_selected_retrieval.yaml").exists()
    assert not env_key_available("DEFINITELY_NOT_A_REAL_KEY", tmp_path)


def test_poppler_detection_path_configuration_and_helper(tmp_path, monkeypatch):
    fake_bin = tmp_path / "poppler" / "Library" / "bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "pdfinfo.exe").write_text("", encoding="utf-8")
    (fake_bin / "pdftoppm.exe").write_text("", encoding="utf-8")
    assert find_poppler_bin([tmp_path / "poppler"]) == fake_bin
    env = prepend_path({"PATH": "C:\\Other"}, fake_bin)
    assert env["PATH"].split(";")[0] == str(fake_bin)
    write_poppler_helper(tmp_path, fake_bin)
    monkeypatch.setenv("PATH", "")
    assert apply_poppler_path_from_helper(tmp_path)
    assert str(fake_bin) in os.environ["PATH"]


def test_poppler_install_attempt_logging_when_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local_app_data"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user_profile"))
    payload = setup_poppler(tmp_path)
    assert payload["status"] == "blocked"
    install_log = tmp_path / "reports/v2_unstructured_cohere/poppler_retry/poppler_install_attempts.json"
    assert install_log.exists()
    assert "blocked_reason" in install_log.read_text(encoding="utf-8")


def test_unstructured_extraction_rejection_rules():
    assert "r:zero_elements" in validate_unstructured_extraction_records({"r": []})
    issues = validate_unstructured_extraction_records({
        "r": [
            {
                "report_id": "r",
                "source_file": "r.pdf",
                "page_number": None,
                "text_length": 10,
            },
            {
                "report_id": "r",
                "source_file": "r.pdf",
                "page_number": None,
                "text_length": 5,
            },
        ]
    })
    assert "r:page_numbers_missing_for_most_elements" in issues


def test_unstructured_index_manifest_includes_page_and_table_counts(tmp_path):
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"pdf")

    class FakeReport:
        report_id = "r"
        pdf_path = pdf
        available = True

    class FakeRegistry:
        def enabled(self):
            return [FakeReport()]

    docs = [
        Document(page_content="table text", metadata={"chunk_id": "c1", "report_id": "r", "page_number": 2, "content_type": "table", "chunk_char_count": 10})
    ]
    records = {"r": [{"page_number": 2, "content_type": "table"}]}
    manifest = _unstructured_index_manifest({"r": docs}, records, FakeRegistry(), "embed")
    assert manifest["pages_extracted_per_report"]["r"] == 1
    assert manifest["table_like_element_counts"]["r"] == 1
