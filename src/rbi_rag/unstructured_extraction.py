from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from importlib import metadata
import importlib.util
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document


CONTENT_TYPE_MAP = {
    "Title": "title",
    "NarrativeText": "narrative_text",
    "ListItem": "list_item",
    "Table": "table",
    "FigureCaption": "figure_caption",
    "Header": "header_footer",
    "Footer": "header_footer",
    "Footnote": "footnote",
    "Text": "mixed",
}


@dataclass(frozen=True)
class UnstructuredElementRecord:
    parser_name: str
    parser_version: str | None
    report_id: str
    report_period: str
    source_file: str
    page_number: int | None
    element_id: str
    element_type: str
    content_type: str
    text: str
    text_length: int
    extraction_strategy: str
    table_html: str | None = None
    table_text: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unstructured_available() -> bool:
    return importlib.util.find_spec("unstructured") is not None


def unstructured_version() -> str | None:
    try:
        return metadata.version("unstructured")
    except metadata.PackageNotFoundError:
        return None


def map_content_type(element_type: str | None) -> str:
    if not element_type:
        return "unknown"
    if "NarrativeText" in element_type:
        return "narrative_text"
    if "ListItem" in element_type:
        return "list_item"
    if "FigureCaption" in element_type:
        return "figure_caption"
    if "Table" in element_type:
        return "table"
    return CONTENT_TYPE_MAP.get(element_type, "unknown")


def _metadata_value(meta: Any, name: str, default: Any = None) -> Any:
    if meta is None:
        return default
    if isinstance(meta, dict):
        return meta.get(name, default)
    return getattr(meta, name, default)


def _element_id(element: Any, report_id: str, page_number: int | None, text: str) -> str:
    explicit = getattr(element, "id", None) or getattr(element, "element_id", None)
    if explicit:
        return str(explicit)
    digest = sha256(f"{report_id}|{page_number}|{text}".encode("utf-8")).hexdigest()[:16]
    page = "unknown" if page_number is None else f"{int(page_number):03d}"
    return f"{report_id}_p{page}_e{digest}"


def element_to_record(
    element: Any,
    *,
    report_id: str,
    report_period: str,
    source_file: str,
    extraction_strategy: str,
    parser_version: str | None = None,
) -> UnstructuredElementRecord:
    text = str(getattr(element, "text", None) or element or "").strip()
    meta = getattr(element, "metadata", None)
    page_number = _metadata_value(meta, "page_number")
    element_type = type(element).__name__
    content_type = map_content_type(element_type)
    table_html = _metadata_value(meta, "text_as_html")
    table_text = text if content_type in {"table", "table_text"} else None
    return UnstructuredElementRecord(
        parser_name="unstructured",
        parser_version=parser_version if parser_version is not None else unstructured_version(),
        report_id=report_id,
        report_period=report_period,
        source_file=source_file,
        page_number=int(page_number) if page_number is not None else None,
        element_id=_element_id(element, report_id, page_number, text),
        element_type=element_type,
        content_type=content_type,
        text=text,
        text_length=len(text),
        extraction_strategy=extraction_strategy,
        table_html=table_html,
        table_text=table_text,
    )


def extract_pdf_elements(
    pdf_path: Path,
    *,
    report_id: str,
    report_period: str,
    strategy: str = "auto",
    infer_table_structure: bool = True,
) -> list[UnstructuredElementRecord]:
    if not unstructured_available():
        raise RuntimeError("unstructured is not installed; install optional V2 dependencies from requirements-v2.txt")
    from unstructured.partition.pdf import partition_pdf

    raw_elements = partition_pdf(
        filename=str(pdf_path),
        strategy=strategy,
        infer_table_structure=infer_table_structure,
    )
    return [
        element_to_record(
            element,
            report_id=report_id,
            report_period=report_period,
            source_file=pdf_path.name,
            extraction_strategy=strategy,
        )
        for element in raw_elements
        if str(getattr(element, "text", None) or element or "").strip()
    ]


def chunk_unstructured_elements(
    elements: Iterable[UnstructuredElementRecord | dict[str, Any]],
    *,
    max_chars: int = 1000,
    merge_below_chars: int = 250,
) -> list[Document]:
    normalised = [item.to_dict() if isinstance(item, UnstructuredElementRecord) else dict(item) for item in elements]
    chunks: list[Document] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        first = buffer[0]
        text = "\n".join(item["text"] for item in buffer if item.get("text"))
        digest = sha256("|".join(item["element_id"] for item in buffer).encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{first['report_id']}_p{int(first.get('page_number') or 0):03d}_u{digest}"
        content_types = sorted({item.get("content_type", "unknown") for item in buffer})
        metadata = {
            "chunk_id": chunk_id,
            "report_id": first["report_id"],
            "report_period": first["report_period"],
            "source_file": first["source_file"],
            "page": first.get("page_number"),
            "page_number": first.get("page_number"),
            "parser_name": first["parser_name"],
            "parser_version": first.get("parser_version"),
            "content_type": content_types[0] if len(content_types) == 1 else "mixed",
            "element_ids": [item["element_id"] for item in buffer],
            "chunk_char_count": len(text),
        }
        table_html = [item.get("table_html") for item in buffer if item.get("table_html")]
        table_text = [item.get("table_text") for item in buffer if item.get("table_text")]
        if table_html:
            metadata["table_html"] = "\n".join(table_html)
        if table_text:
            metadata["table_text"] = "\n".join(table_text)
        chunks.append(Document(page_content=text, metadata=metadata))
        buffer.clear()

    for item in normalised:
        text = item.get("text", "")
        same_scope = (
            buffer
            and buffer[-1]["report_id"] == item["report_id"]
            and buffer[-1].get("page_number") == item.get("page_number")
        )
        is_table = item.get("content_type") in {"table", "table_text"}
        if is_table:
            flush()
            buffer.append(item)
            flush()
            continue
        if not same_scope:
            flush()
        current_len = sum(len(value.get("text", "")) for value in buffer)
        should_merge = (
            buffer
            and same_scope
            and current_len + len(text) + 1 <= max_chars
            and (current_len < merge_below_chars or item.get("content_type") in {"narrative_text", "list_item", "mixed"})
        )
        if should_merge or not buffer:
            buffer.append(item)
        else:
            flush()
            buffer.append(item)
    flush()
    return chunks


def page_element_counts(records: Iterable[UnstructuredElementRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int | None], dict[str, Any]] = {}
    for raw in records:
        item = raw.to_dict() if isinstance(raw, UnstructuredElementRecord) else raw
        key = (item["report_id"], item.get("page_number"))
        row = counts.setdefault(
            key,
            {
                "report_id": item["report_id"],
                "report_period": item["report_period"],
                "page_number": item.get("page_number"),
                "element_count": 0,
                "character_count": 0,
                "table_like_element_count": 0,
            },
        )
        row["element_count"] += 1
        row["character_count"] += int(item.get("text_length", 0))
        if item.get("content_type") in {"table", "table_text"}:
            row["table_like_element_count"] += 1
    return [counts[key] for key in sorted(counts, key=lambda item: (item[0], item[1] or 0))]
