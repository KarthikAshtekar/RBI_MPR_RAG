from __future__ import annotations

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from .chunking import split_pages
from .config import RAGConfig


def load_and_chunk_pdf(config: RAGConfig):
    if not config.pdf_path.is_file():
        raise FileNotFoundError(f"RBI report not found: {config.pdf_path}")
    pages = PyPDFLoader(str(config.pdf_path)).load()
    chunks = split_pages(
        pages, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    page_chunk_counts: dict[int, int] = {}
    for global_index, chunk in enumerate(chunks):
        page = int(chunk.metadata.get("page", -1)) + 1
        local_index = page_chunk_counts.get(page, 0)
        page_chunk_counts[page] = local_index + 1
        chunk_id = f"{config.report_id}_p{page:03d}_c{local_index:02d}"
        chunk.metadata = {
            **chunk.metadata,
            "report_id": config.report_id,
            "report_period": config.report_period,
            "report_date": config.report_date,
            "source_file": config.pdf_path.name,
            "page": page,
            "page_number": page,
            "chunk_id": chunk_id,
            "chunk_index": global_index,
            "page_chunk_index": local_index,
            "ingestion_version": config.ingestion_version,
        }
    return pages, chunks

