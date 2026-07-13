from pathlib import Path
from types import SimpleNamespace
from langchain_core.documents import Document
from rbi_rag.multi_config import MultiReportConfig
from rbi_rag.multi_retrieval import MultiReportRetriever, deduplicate_preserving_reports
from rbi_rag.report_bm25 import BM25ByReport
from rbi_rag.report_registry import ReportRegistry
from rbi_rag.temporal_router import TemporalQueryRouter


def doc(report, chunk, page=1, text=None):
    period = {"rbi_mpr_2025_04":"April 2025","rbi_mpr_2025_10":"October 2025","rbi_mpr_2026_04":"April 2026"}[report]
    return Document(page_content=text or f"{report} inflation evidence {chunk}", metadata={
        "report_id": report, "report_period": period, "chunk_id": chunk, "page": page,
    })


class Store:
    def __init__(self, chunks): self.chunks = chunks; self.filters = []
    def similarity_search_with_relevance_scores(self, query, k, filter):
        self.filters.append(filter)
        return [(d, .9 - i*.01) for i,d in enumerate(self.chunks[filter["report_id"]][:k])]


class Reranker:
    def predict(self, pairs): return list(range(len(pairs), 0, -1))


def setup_retriever():
    ids = ["rbi_mpr_2025_04","rbi_mpr_2025_10","rbi_mpr_2026_04"]
    chunks = {rid: [doc(rid, f"{rid}_c{i}", i+1) for i in range(6)] for rid in ids}
    config = MultiReportConfig.from_yaml(Path("configs/multi_report.yaml"))
    registry = ReportRegistry.from_yaml(Path("configs/reports.yaml"))
    store = Store(chunks)
    return MultiReportRetriever(store, chunks, registry, config, Reranker()), store


def test_shared_dense_filter_and_single_report_zero_contamination():
    retriever, store = setup_retriever(); router = TemporalQueryRouter(retriever.registry)
    result = retriever.retrieve_from_query_plan(router.route("Inflation in April 2025"))
    assert store.filters == [{"report_id":"rbi_mpr_2025_04"}]
    assert {d.metadata["report_id"] for d in result["final_selected_chunks"]} == {"rbi_mpr_2025_04"}


def test_report_specific_bm25_isolation():
    retriever, _ = setup_retriever()
    values = retriever.bm25.search("rbi_mpr_2025_10", "inflation", 4)
    assert all(d.metadata["report_id"] == "rbi_mpr_2025_10" for d,_ in values)


def test_pairwise_reranking_quota_and_balanced_context():
    retriever, _ = setup_retriever(); router = TemporalQueryRouter(retriever.registry)
    result = retriever.retrieve_from_query_plan(router.route("Compare April and October 2025 inflation"))
    assert result["final_chunk_quota_by_report"] == {"rbi_mpr_2025_04":3,"rbi_mpr_2025_10":3}
    assert len(result["rrf_results_by_report"]["rbi_mpr_2025_04"]) <= 15


def test_trend_quota_is_two_per_report():
    retriever, _ = setup_retriever(); router = TemporalQueryRouter(retriever.registry)
    result = retriever.retrieve_from_query_plan(router.route("Inflation trend across all reports"))
    assert set(result["final_chunk_quota_by_report"].values()) == {2}


def test_deduplication_preserves_required_report_coverage():
    a = doc("rbi_mpr_2025_04", "a", text="same")
    b = doc("rbi_mpr_2025_10", "b", text="same")
    result = deduplicate_preserving_reports([(a,1),(a,.9),(b,.8)], {a.metadata["report_id"],b.metadata["report_id"]})
    assert {d.metadata["report_id"] for d,_ in result} == {"rbi_mpr_2025_04","rbi_mpr_2025_10"}

