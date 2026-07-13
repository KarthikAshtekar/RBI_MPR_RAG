from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import time
from langchain_community.retrievers import BM25Retriever

from .fusion import reciprocal_rank_fusion
from .reranking import rerank


class NaiveGlobalRetriever:
    """Evaluation-only global baseline with no report filtering or quotas."""
    def __init__(self, store, chunks_by_report, registry, config, cross_encoder):
        self.store,self.registry,self.config,self.cross_encoder=store,registry,config,cross_encoder
        chunks=[chunk for rid in sorted(chunks_by_report) for chunk in chunks_by_report[rid]]
        self.bm25=BM25Retriever.from_documents(chunks)

    def retrieve_from_query_plan(self, plan):
        if plan.query_type == "unsupported_period":
            return {"query_plan":asdict(plan),"report_ids_searched":[],"dense_results_by_report":{},
                    "bm25_results_by_report":{},"rrf_results_by_report":{},"reranked_results_by_report":{},
                    "final_selected_chunks":[],"final_chunk_quota_by_report":{},
                    "missing_report_warnings":[plan.routing_reason],"retrieval_latency_by_stage":{}}
        report_count=max(1,len(plan.report_ids))
        budget=self.config.final_single if report_count==1 else \
            (self.config.final_trend*report_count if plan.query_type=="trend_all_reports" else self.config.final_comparative*report_count)
        started=time.perf_counter()
        dense=self.store.similarity_search_with_relevance_scores(plan.original_query,k=self.config.dense_k)
        dense_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter(); self.bm25.k=self.config.bm25_k
        sparse=[(doc,None) for doc in self.bm25.invoke(plan.original_query)]
        bm25_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter(); fused=reciprocal_rank_fusion([dense,sparse],rrf_k=self.config.rrf_k,limit=max(self.config.reranker_k,budget))
        rrf_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter(); ranked=rerank(self.cross_encoder,plan.original_query,fused,budget)
        rerank_ms=(time.perf_counter()-started)*1000
        def group(items):
            result={}
            for doc,score in items: result.setdefault(doc.metadata["report_id"],[]).append({"chunk_id":doc.metadata["chunk_id"],"page":doc.metadata["page"],"score":score})
            return result
        counts=Counter(doc.metadata["report_id"] for doc,_ in ranked)
        return {"query_plan":asdict(plan),"report_ids_searched":[r.report_id for r in self.registry.available()],
                "dense_results_by_report":group(dense),"bm25_results_by_report":group(sparse),
                "rrf_results_by_report":group(fused),"reranked_results_by_report":group(ranked),
                "final_selected_chunks":[doc for doc,_ in ranked],"final_chunk_quota_by_report":dict(counts),
                "missing_report_warnings":[],"retrieval_latency_by_stage":{"__global__":{"dense_ms":dense_ms,"bm25_ms":bm25_ms,"rrf_ms":rrf_ms,"rerank_ms":rerank_ms}}}

