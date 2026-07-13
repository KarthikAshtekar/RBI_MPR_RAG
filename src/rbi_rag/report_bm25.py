from __future__ import annotations

from langchain_community.retrievers import BM25Retriever


class BM25ByReport:
    """Deterministically rebuilt in memory; avoids unsafe pickle serialization."""

    def __init__(self, chunks_by_report: dict[str, list]):
        self._retrievers = {
            report_id: BM25Retriever.from_documents(chunks)
            for report_id, chunks in sorted(chunks_by_report.items()) if chunks
        }

    def get_bm25_retriever(self, report_id: str):
        if report_id not in self._retrievers:
            raise KeyError(f"no BM25 index for report {report_id}")
        return self._retrievers[report_id]

    def search(self, report_id: str, query: str, k: int):
        retriever = self.get_bm25_retriever(report_id)
        retriever.k = k
        documents = retriever.invoke(query)
        assert all(doc.metadata["report_id"] == report_id for doc in documents)
        return [(document, None) for document in documents]

