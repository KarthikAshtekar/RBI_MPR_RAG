from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .chunking import split_pages
from .config import file_sha256
from .multi_config import MultiReportConfig
from .report_registry import ReportRegistry, ReportSpec


def report_fingerprint(report: ReportSpec, checksum: str, config: MultiReportConfig) -> str:
    value = "|".join((report.report_id, checksum, str(config.chunk_size),
                      str(config.chunk_overlap), config.embedding_model, config.ingestion_version))
    return sha256(value.encode()).hexdigest()[:16]


def ingest_report(report: ReportSpec, config: MultiReportConfig):
    if not report.available:
        raise FileNotFoundError(f"missing report PDF: {report.pdf_path}")
    pages = PyPDFLoader(str(report.pdf_path)).load()
    title_text = " ".join(page.page_content for page in pages[:4]).lower()
    if "monetary policy report" not in title_text or report.report_period.lower() not in title_text:
        raise ValueError(f"PDF identity does not match registry entry {report.report_id}")
    chunks = split_pages(pages, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    per_page: dict[int, int] = {}
    for chunk in chunks:
        page_index = int(chunk.metadata["page"])
        page = page_index + 1
        chunk_index = per_page.get(page, 0); per_page[page] = chunk_index + 1
        chunk.metadata = {
            "report_id": report.report_id, "report_period": report.report_period,
            "report_date": report.report_date.isoformat(), "report_year": report.report_year,
            "report_month": report.report_month, "report_type": report.report_type,
            "source_file": report.pdf_path.name, "page": page, "page_index": page_index,
            "chunk_index": chunk_index,
            "chunk_id": f"{report.report_id}_p{page:03d}_c{chunk_index:03d}",
            "section": None,
        }
    return pages, chunks


def _safe_for_chroma(chunks):
    return [Document(page_content=chunk.page_content,
                     metadata={k: v for k, v in chunk.metadata.items() if v is not None})
            for chunk in chunks]


def open_shared_index(config: MultiReportConfig):
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    return Chroma(collection_name=config.collection_name, embedding_function=embeddings,
                  persist_directory=str(config.persistence_path))


def build_multi_report_index(config: MultiReportConfig, registry: ReportRegistry):
    store = open_shared_index(config)
    previous = json.loads(config.manifest_path.read_text(encoding="utf-8")) \
        if config.manifest_path.exists() else {"reports": {}}
    manifest = {
        "schema_version": 1, "registry_version": registry.version,
        "collection_name": config.collection_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "reports": {},
    }
    chunk_sets = {}
    for report in registry.enabled():
        if not report.available:
            manifest["reports"][report.report_id] = {
                "status": "missing", "pdf_path": str(report.pdf_path), "indexing_status": "not_indexed"
            }
            continue
        checksum = file_sha256(report.pdf_path)
        fingerprint = report_fingerprint(report, checksum, config)
        old = previous.get("reports", {}).get(report.report_id, {})
        pages, chunks = ingest_report(report, config)
        unchanged = old.get("ingestion_fingerprint") == fingerprint
        if not unchanged:
            existing = store._collection.get(where={"report_id": report.report_id}, include=[])
            if existing.get("ids"):
                store.delete(ids=existing["ids"])
            safe_chunks = _safe_for_chroma(chunks)
            store.add_documents(safe_chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        chunk_sets[report.report_id] = chunks
        manifest["reports"][report.report_id] = {
            "status": "available", "pdf_path": str(report.pdf_path), "sha256": checksum,
            "page_count": len(pages), "chunk_count": len(chunks),
            "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ingestion_fingerprint": fingerprint,
            "indexing_status": "reused" if unchanged else "indexed",
            "registry_metadata": {
                "report_id": report.report_id, "report_period": report.report_period,
                "report_date": report.report_date.isoformat(), "report_year": report.report_year,
                "report_month": report.report_month, "report_type": report.report_type,
                "enabled": report.enabled,
            },
            "validation_status": "validated",
            "extraction_status": "extracted",
        }
    all_ids = [chunk.metadata["chunk_id"] for chunks in chunk_sets.values() for chunk in chunks]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate chunk IDs detected in multi-report corpus")
    manifest["total_chunk_count"] = len(all_ids)
    manifest["duplicate_chunk_ids"] = 0
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return store, chunk_sets, manifest


def inspect_shared_index(store, registry: ReportRegistry):
    counts = {}
    for report in registry.enabled():
        counts[report.report_id] = len(store._collection.get(
            where={"report_id": report.report_id}, include=[]).get("ids", []))
    return {"total_chunks": store._collection.count(), "chunks_by_report": counts}
