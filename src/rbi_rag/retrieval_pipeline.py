from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from .config import RAGConfig
from .dense_retrieval import dense_search
from .fusion import reciprocal_rank_fusion
from .reranking import rerank
from .sparse_retrieval import bm25_search


class RetrievalPipeline:
    def __init__(self, chunks, vector_store, config: RAGConfig):
        self.config = config
        self.vector_store = vector_store
        self.bm25 = BM25Retriever.from_documents(list(chunks))
        self.cross_encoder = CrossEncoder(config.reranker_model)

    def dense_candidates(self, query: str):
        return dense_search(self.vector_store, query, self.config.dense_k)

    def sparse_candidates(self, query: str):
        return bm25_search(self.bm25, query, self.config.bm25_k)

    def dense(self, query: str):
        return self.dense_candidates(query)[: self.config.final_k]

    def dense_reranked(self, query: str):
        return rerank(
            self.cross_encoder, query,
            self.dense_candidates(query)[: self.config.reranker_candidate_k],
            self.config.final_k,
        )

    def bm25_only(self, query: str):
        return self.sparse_candidates(query)[: self.config.final_k]

    def bm25_reranked(self, query: str):
        return rerank(
            self.cross_encoder, query,
            self.sparse_candidates(query)[: self.config.reranker_candidate_k],
            self.config.final_k,
        )

    def hybrid_candidates(self, query: str):
        return reciprocal_rank_fusion(
            [self.sparse_candidates(query), self.dense_candidates(query)],
            rrf_k=self.config.fusion_rrf_k,
            limit=self.config.reranker_candidate_k,
        )

    def hybrid_rrf(self, query: str):
        return self.hybrid_candidates(query)[: self.config.final_k]

    def hybrid_reranked(self, query: str):
        return rerank(
            self.cross_encoder, query, self.hybrid_candidates(query), self.config.final_k
        )

    def strategies(self):
        return {
            "dense": self.dense,
            "dense_reranked": self.dense_reranked,
            "bm25": self.bm25_only,
            "bm25_reranked": self.bm25_reranked,
            "hybrid_rrf": self.hybrid_rrf,
            "hybrid_reranked": self.hybrid_reranked,
        }

