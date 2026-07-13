from types import SimpleNamespace
from rbi_rag.config import RAGConfig
from rbi_rag.retrieval_pipeline import RetrievalPipeline


class Doc:
    def __init__(self, chunk_id, text="text"):
        self.metadata = {"chunk_id": chunk_id, "page": 1, "page_number": 1}
        self.page_content = text


def test_single_report_hybrid_pipeline_smoke_without_network():
    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline.config = RAGConfig()
    pipeline.sparse_candidates = lambda _: [(Doc("s"), None)]
    pipeline.dense_candidates = lambda _: [(Doc("d"), .8)]
    pipeline.cross_encoder = SimpleNamespace(predict=lambda pairs: [1.0] * len(pairs))
    result = pipeline.hybrid_reranked("question")
    assert len(result) == 2
    assert {doc.metadata["chunk_id"] for doc, _ in result} == {"s", "d"}

