from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from hashlib import sha256
from math import ceil

from langchain_core.documents import Document


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class StructuralSelection:
    documents: list[Document]
    added_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    expansion_reason: str


def split_sentence_like_units(text: str) -> list[str]:
    units = [" ".join(part.split()) for part in SENTENCE_RE.split(text) if part.strip()]
    return units or [" ".join(text.split())] if text.strip() else []


def semantic_chunks_for_page(
    page: Document,
    *,
    min_chars: int,
    max_chars: int,
    percentile: int,
) -> list[Document]:
    """Deterministic sentence-like semantic fallback used for offline tests.

    The production Phase 6B runner records full semantic re-indexing as skipped
    unless a proper embedding-backed semantic index is available. This helper
    preserves metadata and enforces min/max-ish chunk sizes for controlled tests.
    """
    units = split_sentence_like_units(page.page_content)
    chunks: list[Document] = []
    current: list[str] = []
    for unit in units:
        candidate = " ".join(current + [unit]).strip()
        should_split = current and (len(candidate) > max_chars or (len(candidate) >= min_chars and percentile <= 85))
        if should_split:
            chunks.append(_make_child(page, " ".join(current), len(chunks), "semantic"))
            current = [unit]
        else:
            current.append(unit)
    if current:
        chunks.append(_make_child(page, " ".join(current), len(chunks), "semantic"))
    return chunks


def _make_child(source: Document, text: str, index: int, prefix: str) -> Document:
    metadata = dict(source.metadata)
    base = metadata.get("chunk_id") or f"{metadata.get('report_id', 'report')}_p{metadata.get('page', 0):03d}"
    metadata["chunk_id"] = f"{base}_{prefix}{index:03d}"
    metadata["parent_chunk_id"] = base
    metadata["structural_unit"] = prefix
    return Document(page_content=text, metadata=metadata)


def chunk_lookup(chunks_by_report: dict[str, list[Document]]):
    return {
        chunk.metadata["chunk_id"]: chunk
        for chunks in chunks_by_report.values()
        for chunk in chunks
    }


def ordered_report_chunks(chunks_by_report: dict[str, list[Document]]):
    return {
        report_id: sorted(
            chunks,
            key=lambda chunk: (int(chunk.metadata.get("page", 0)), int(chunk.metadata.get("chunk_index", 0))),
        )
        for report_id, chunks in chunks_by_report.items()
    }


def build_parent_window(
    selected_ids: list[str],
    chunks_by_report: dict[str, list[Document]],
    *,
    strategy: str,
    parent_max_chars: int,
) -> StructuralSelection:
    lookup = chunk_lookup(chunks_by_report)
    ordered = ordered_report_chunks(chunks_by_report)
    selected: list[Document] = []
    added: list[str] = []
    seen: set[str] = set()
    for chunk_id in selected_ids:
        if chunk_id not in lookup:
            continue
        source = lookup[chunk_id]
        report_id = source.metadata["report_id"]
        candidates = _window_candidates(source, ordered[report_id], strategy)
        text_parts: list[str] = []
        source_ids: list[str] = []
        for candidate in candidates:
            if sum(len(part) for part in text_parts) + len(candidate.page_content) > parent_max_chars and text_parts:
                break
            text_parts.append(candidate.page_content)
            source_ids.append(candidate.metadata["chunk_id"])
        parent_id = f"{chunk_id}_parent_{sha256('|'.join(source_ids).encode()).hexdigest()[:8]}"
        if parent_id in seen:
            continue
        metadata = dict(source.metadata)
        metadata.update({
            "chunk_id": parent_id,
            "child_chunk_id": chunk_id,
            "parent_source_chunk_ids": source_ids,
            "structural_unit": "parent_window",
            "parent_strategy": strategy,
        })
        selected.append(Document(page_content="\n".join(text_parts), metadata=metadata))
        seen.add(parent_id)
        added.extend(source_id for source_id in source_ids if source_id != chunk_id)
    return StructuralSelection(selected, sorted(set(added)), [], strategy)


def _window_candidates(source: Document, ordered: list[Document], strategy: str) -> list[Document]:
    index = next((i for i, chunk in enumerate(ordered) if chunk.metadata["chunk_id"] == source.metadata["chunk_id"]), None)
    if index is None:
        return [source]
    if strategy == "page_bounded_parent":
        return [chunk for chunk in ordered if chunk.metadata.get("page") == source.metadata.get("page")]
    if strategy == "adjacent_child_window":
        return ordered[max(0, index - 1): index + 2]
    return [
        chunk for chunk in ordered[max(0, index - 1): index + 2]
        if chunk.metadata.get("page") == source.metadata.get("page")
    ]


def sentence_window_documents(
    selected_ids: list[str],
    chunks_by_report: dict[str, list[Document]],
    *,
    before: int,
    after: int,
    max_chars: int,
) -> StructuralSelection:
    lookup = chunk_lookup(chunks_by_report)
    selected: list[Document] = []
    for chunk_id in selected_ids:
        source = lookup.get(chunk_id)
        if not source:
            continue
        units = split_sentence_like_units(source.page_content)
        if not units:
            continue
        center = max(0, min(len(units) - 1, len(units) // 2))
        window = units[max(0, center - before): min(len(units), center + after + 1)]
        text = " ".join(window)[:max_chars]
        metadata = dict(source.metadata)
        metadata.update({
            "chunk_id": f"{chunk_id}_sentence_window_{before}_{after}",
            "child_chunk_id": chunk_id,
            "structural_unit": "sentence_window",
            "window_before": before,
            "window_after": after,
        })
        selected.append(Document(page_content=text, metadata=metadata))
    return StructuralSelection(selected, [], [], f"sentence_window_{before}_{after}")


def adjacent_chunk_expansion(
    selected_ids: list[str],
    chunks_by_report: dict[str, list[Document]],
    *,
    max_tokens: int,
    mode: str,
    query_type: str,
) -> StructuralSelection:
    lookup = chunk_lookup(chunks_by_report)
    ordered = ordered_report_chunks(chunks_by_report)
    selected: list[Document] = []
    added: list[str] = []
    seen: set[str] = set()
    for chunk_id in selected_ids:
        source = lookup.get(chunk_id)
        if not source:
            continue
        report_id = source.metadata["report_id"]
        report_chunks = ordered[report_id]
        index = next((i for i, chunk in enumerate(report_chunks) if chunk.metadata["chunk_id"] == chunk_id), None)
        candidates = [source]
        if index is not None and _should_expand(source, mode, query_type):
            neighbours = report_chunks[max(0, index - 1): index] + report_chunks[index + 1:index + 2]
            candidates.extend(chunk for chunk in neighbours if chunk.metadata.get("page") == source.metadata.get("page"))
        for candidate in candidates:
            cid = candidate.metadata["chunk_id"]
            if cid in seen:
                continue
            current_chars = sum(len(doc.page_content) for doc in selected)
            if ceil((current_chars + len(candidate.page_content)) / 4) > max_tokens and selected:
                continue
            selected.append(candidate)
            seen.add(cid)
            if cid != chunk_id:
                added.append(cid)
    return StructuralSelection(selected, sorted(set(added)), [], mode)


def _should_expand(source: Document, mode: str, query_type: str) -> bool:
    if mode == "none":
        return False
    if mode == "multi_facet_or_trend":
        return query_type == "trend_all_reports"
    if mode == "boundary":
        text = source.page_content.strip()
        return not text.endswith((".", "!", "?")) or len(text) < 900
    return True


def e5_query(text: str) -> str:
    return f"query: {text}"


def e5_passage(text: str) -> str:
    return f"passage: {text}"


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
