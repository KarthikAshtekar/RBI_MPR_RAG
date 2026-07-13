from .config import RAGConfig
from .indexing import build_or_open_index
from .ingestion import load_and_chunk_pdf
from .retrieval_pipeline import RetrievalPipeline


def build_retrieval_suite(config: RAGConfig):
    config.validate()
    _, chunks = load_and_chunk_pdf(config)
    store, _ = build_or_open_index(chunks, config)
    return RetrievalPipeline(chunks, store, config)

