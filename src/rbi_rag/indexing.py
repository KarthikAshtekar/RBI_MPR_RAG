from __future__ import annotations

import json
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import RAGConfig, file_sha256

logger = logging.getLogger(__name__)


def build_or_open_index(chunks, config: RAGConfig):
    digest = file_sha256(config.pdf_path)
    identity = {
        "pdf_sha256": digest,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "ingestion_version": config.ingestion_version,
    }
    collection = f"{config.report_id}-{config.fingerprint(digest)}"
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(config.chroma_path),
    )
    manifest_path = config.chroma_path / f"{collection}.manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing != identity or store._collection.count() != len(chunks):
        logger.info("Building Chroma collection %s with %d chunks", collection, len(chunks))
        store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    else:
        logger.info("Reusing unchanged Chroma collection %s", collection)
    return store, collection
