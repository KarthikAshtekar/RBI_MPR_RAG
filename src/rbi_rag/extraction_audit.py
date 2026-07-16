from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def _select_pages(pages):
    texts = [page.page_content or "" for page in pages]
    chosen, used = {}, set()
    rules = {
        "narrative_heavy": lambda t: len(t) > 1800 and len(re.findall(r"\d", t)) < 80 and "Chart " not in t and "Table " not in t,
        "table_heavy": lambda t: "Table " in t or len(re.findall(r"\d", t)) > 160,
        "chart_or_figure_heavy": lambda t: "Chart " in t or "Figure " in t,
        "footnotes": lambda t: "Note:" in t or "Notes:" in t or "footnote" in t.lower(),
        "section_transition": lambda t: "CHAPTER" in t or bool(re.search(r"\n[IVX]+\.\s+[A-Z]", t)),
    }
    targets = {"narrative_heavy": 5, "table_heavy": 3, "chart_or_figure_heavy": 3,
               "footnotes": 2, "section_transition": 2}
    for page_type, rule in rules.items():
        candidates = [i for i, text in enumerate(texts) if rule(text) and i not in used]
        if len(candidates) < targets[page_type]:
            candidates += [i for i in range(len(texts)) if i not in used and i not in candidates]
        selected = candidates[:targets[page_type]]
        chosen[page_type] = selected; used.update(selected)
    return chosen, texts


def audit_reports(registry, output_json: Path, output_markdown: Path):
    records = []
    for report in registry.available():
        pages = PyPDFLoader(str(report.pdf_path)).load()
        selections, texts = _select_pages(pages)
        for page_type, indices in selections.items():
            for index in indices:
                text = texts[index]
                if not text.strip():
                    severity = "unusable_for_precise_QA"
                elif page_type in ("table_heavy", "chart_or_figure_heavy"):
                    severity = "material_issue"
                elif page_type == "footnotes":
                    severity = "minor_issue"
                else:
                    severity = "clean"
                records.append({
                    "report_id": report.report_id, "page": index + 1,
                    "page_index": index, "page_type": page_type,
                    "text_extraction_status": "present" if text.strip() else "missing",
                    "reading_order_status": "requires_visual_check" if page_type in ("table_heavy", "chart_or_figure_heavy") else "appears_linear",
                    "table_preservation_status": "flattened_text" if page_type == "table_heavy" else "not_primary_content",
                    "chart_label_preservation_status": "labels_only_no_chart_semantics" if page_type == "chart_or_figure_heavy" else "not_primary_content",
                    "severity": severity,
                    "notes": f"PyPDFLoader extracted {len(text)} characters; classification based on extracted page markers and structure.",
                    "recommended_action": "Manually verify labels and values before numeric QA; consider a layout-aware parser later." if severity in ("material_issue", "unusable_for_precise_QA") else "Usable with normal source verification.",
                })
    payload = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "numeric_evaluation_exclusions": [
            {"report_id": r["report_id"], "page": r["page"], "reason": r["severity"]}
            for r in records if r["severity"] in ("material_issue", "unusable_for_precise_QA")
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts = Counter((r["report_id"], r["severity"]) for r in records)
    lines = ["# PyPDFLoader Extraction Audit", "",
             "Pages marked material or unusable must not support numeric evaluation without manual verification.", "",
             "| Report | Page | Type | Severity | Reading order | Recommendation |", "|---|---:|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['report_id']} | {r['page']} | {r['page_type']} | {r['severity']} | {r['reading_order_status']} | {r['recommended_action']} |")
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
