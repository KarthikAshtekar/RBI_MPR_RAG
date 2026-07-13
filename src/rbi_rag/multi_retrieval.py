from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import time

from sentence_transformers import CrossEncoder

from .fusion import reciprocal_rank_fusion
from .multi_config import MultiReportConfig
from .report_bm25 import BM25ByReport
from .report_registry import ReportRegistry
from .reranking import rerank
from .schemas import QueryPlan


def _serialize(items):
    return [{"chunk_id": doc.metadata["chunk_id"], "page": doc.metadata["page"],
             "score": score} for doc, score in items]


class MultiReportRetriever:
    def __init__(self, store, chunks_by_report, registry: ReportRegistry,
                 config: MultiReportConfig, cross_encoder=None):
        self.store, self.registry, self.config = store, registry, config
        self.bm25 = BM25ByReport(chunks_by_report)
        self.cross_encoder = cross_encoder or CrossEncoder(config.reranker_model)

    def dense_search(self, query: str, report_id: str):
        values = self.store.similarity_search_with_relevance_scores(
            query, k=self.config.dense_k, filter={"report_id": report_id}
        )
        for document, _ in values:
            document.metadata.setdefault("section", None)
        return values

    def _retrieve_report(self, query: str, report_id: str, quota: int):
        latency = {}
        started = time.perf_counter(); dense = self.dense_search(query, report_id)
        latency["dense_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); sparse = self.bm25.search(report_id, query, self.config.bm25_k)
        latency["bm25_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        fused = reciprocal_rank_fusion([dense, sparse], rrf_k=self.config.rrf_k,
                                       limit=self.config.reranker_k)
        latency["rrf_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); ranked = rerank(self.cross_encoder, query, fused, quota)
        latency["rerank_ms"] = (time.perf_counter() - started) * 1000
        return dense, sparse, fused, ranked, latency

    def retrieve_single_report(self, query: str, report_id: str, plan: QueryPlan | None = None):
        if report_id not in self.bm25._retrievers:
            return self._empty(plan, [report_id], f"Report is not indexed: {report_id}")
        dense, sparse, fused, ranked, latency = self._retrieve_report(
            query, report_id, self.config.final_single
        )
        return self._result(plan, [report_id], {report_id: dense}, {report_id: sparse},
                            {report_id: fused}, {report_id: ranked}, ranked,
                            {report_id: len(ranked)}, [], {report_id: latency})

    def retrieve_comparative(self, query: str, report_ids: list[str] | tuple[str, ...],
                             plan: QueryPlan | None = None):
        quota = self.config.final_trend if plan and plan.query_type == "trend_all_reports" \
            else self.config.final_comparative
        dense_by, sparse_by, fused_by, ranked_by, latency_by, warnings = {}, {}, {}, {}, {}, []
        for report_id in self._chronological(report_ids):
            if report_id not in self.bm25._retrievers:
                warnings.append(f"Report is not indexed: {report_id}"); continue
            dense, sparse, fused, ranked, latency = self._retrieve_report(query, report_id, quota)
            dense_by[report_id], sparse_by[report_id] = dense, sparse
            fused_by[report_id], ranked_by[report_id], latency_by[report_id] = fused, ranked, latency
        final = [item for report_id in self._chronological(report_ids)
                 for item in ranked_by.get(report_id, [])]
        final = deduplicate_preserving_reports(final, set(ranked_by))
        counts = Counter(doc.metadata["report_id"] for doc, _ in final)
        return self._result(plan, list(report_ids), dense_by, sparse_by, fused_by, ranked_by,
                            final, dict(counts), warnings, latency_by)

    def retrieve_from_query_plan(self, plan: QueryPlan):
        if plan.query_type == "unsupported_period":
            return self._empty(plan, list(plan.report_ids), plan.routing_reason)
        if plan.query_type in ("single_report", "latest_report") and len(plan.report_ids) == 1:
            return self.retrieve_single_report(plan.original_query, plan.report_ids[0], plan)
        return self.retrieve_comparative(plan.original_query, plan.report_ids, plan)

    def _chronological(self, report_ids):
        by_id = self.registry.by_id()
        return sorted(report_ids, key=lambda value: by_id[value].report_date)

    @staticmethod
    def _result(plan, report_ids, dense, sparse, fused, ranked, final, quota, warnings, latency):
        return {
            "query_plan": asdict(plan) if plan else None, "report_ids_searched": report_ids,
            "dense_results_by_report": {k: _serialize(v) for k, v in dense.items()},
            "bm25_results_by_report": {k: _serialize(v) for k, v in sparse.items()},
            "rrf_results_by_report": {k: _serialize(v) for k, v in fused.items()},
            "reranked_results_by_report": {k: _serialize(v) for k, v in ranked.items()},
            "final_selected_chunks": [doc for doc, _ in final],
            "final_chunk_quota_by_report": quota, "missing_report_warnings": warnings,
            "retrieval_latency_by_stage": latency,
        }

    @classmethod
    def _empty(cls, plan, report_ids, warning):
        return cls._result(plan, report_ids, {}, {}, {}, {}, [], {}, [warning], {})


def deduplicate_preserving_reports(items, required_reports: set[str]):
    selected, seen = [], set()
    for document, score in items:
        key = (document.metadata["report_id"], " ".join(document.page_content.split()).lower())
        if key in seen:
            continue
        seen.add(key); selected.append((document, score))
    represented = {doc.metadata["report_id"] for doc, _ in selected}
    for report_id in required_reports - represented:
        candidate = next((item for item in items if item[0].metadata["report_id"] == report_id), None)
        if candidate:
            selected.append(candidate)
    return selected

