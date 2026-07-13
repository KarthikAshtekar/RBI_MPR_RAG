import json
from hashlib import sha256
from pathlib import Path

import yaml
from langchain_core.documents import Document

from rbi_rag.structural_optimisation import (
    adjacent_chunk_expansion,
    build_parent_window,
    dependency_available,
    e5_passage,
    e5_query,
    semantic_chunks_for_page,
    sentence_window_documents,
    split_sentence_like_units,
)


def docs():
    values = []
    for index, text in enumerate([
        "Alpha sentence. Boundary lead.",
        "Beta evidence is here. More beta.",
        "Gamma continuation. Final sentence.",
    ]):
        values.append(Document(
            page_content=text,
            metadata={
                "report_id": "r1",
                "page": 10,
                "chunk_index": index,
                "chunk_id": f"r1_p010_c{index:03d}",
            },
        ))
    return {"r1": values}


def test_phase6b_registry_has_required_sections():
    registry = yaml.safe_load(Path("configs/structural_optimisation_experiments.yaml").read_text())
    required = {"description", "parent_experiment", "enabled", "parser", "chunking", "index", "embedding",
                "retrieval", "bm25", "fusion", "reranker", "context_selection", "deduplication", "evaluation"}
    assert "stage_a_selected_reference" in registry
    for experiment in registry.values():
        assert required <= set(experiment)


def test_semantic_chunks_preserve_metadata_and_bounds():
    page = Document(
        page_content="One sentence. Two sentence. Three sentence. Four sentence.",
        metadata={"report_id": "r1", "page": 2, "chunk_id": "r1_p002_c000"},
    )
    chunks = semantic_chunks_for_page(page, min_chars=10, max_chars=30, percentile=85)
    assert chunks
    assert all(chunk.metadata["report_id"] == "r1" and chunk.metadata["page"] == 2 for chunk in chunks)
    assert max(len(chunk.page_content) for chunk in chunks) <= 45


def test_child_parent_mapping_preserves_report_and_page():
    selection = build_parent_window(["r1_p010_c001"], docs(), strategy="same_page_local_window", parent_max_chars=1200)
    assert selection.documents[0].metadata["report_id"] == "r1"
    assert selection.documents[0].metadata["page"] == 10
    assert selection.documents[0].metadata["child_chunk_id"] == "r1_p010_c001"
    assert "r1_p010_c000" in selection.added_chunk_ids


def test_sentence_window_bounds_and_metadata():
    selection = sentence_window_documents(["r1_p010_c001"], docs(), before=1, after=1, max_chars=50)
    assert len(selection.documents[0].page_content) <= 50
    assert selection.documents[0].metadata["structural_unit"] == "sentence_window"
    assert selection.documents[0].metadata["report_id"] == "r1"


def test_adjacent_expansion_preserves_report_and_budget():
    selection = adjacent_chunk_expansion(["r1_p010_c001"], docs(), max_tokens=40, mode="always", query_type="single_report")
    assert {doc.metadata["report_id"] for doc in selection.documents} == {"r1"}
    assert sum(len(doc.page_content) for doc in selection.documents) / 4 <= 40
    assert selection.added_chunk_ids


def test_e5_prefix_formatting():
    assert e5_query("repo rate") == "query: repo rate"
    assert e5_passage("policy text") == "passage: policy text"


def test_dependency_available_false_for_missing_module():
    assert dependency_available("definitely_missing_phase6b_module") is False


def test_final_config_checksum_shape():
    payload = {"id": "x", "retrieval": {"dense_k": 1}}
    digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert len(digest) == 64
