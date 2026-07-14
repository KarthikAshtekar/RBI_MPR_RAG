from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_demo_examples(root: Path = ROOT) -> list[dict[str, Any]]:
    return load_json(root / "reports/v2_sufficiency/dev_generation_sufficiency_raw_results.json", [])


def load_comparison_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    return load_json(root / "reports/final_comparison/rag_methods_master_comparison.json", [])


def find_demo_answer(query: str, examples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not examples:
        return None
    query_norm = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    for row in examples:
        row_query = " ".join(re.findall(r"[a-z0-9]+", str(row.get("original_query", "")).lower()))
        if query_norm and query_norm == row_query:
            return row
    for row in examples:
        row_query = " ".join(re.findall(r"[a-z0-9]+", str(row.get("original_query", "")).lower()))
        if query_norm and (query_norm in row_query or row_query in query_norm):
            return row
    return examples[0]


def extract_context_snippets(source_labelled_context: str, max_snippets: int = 8) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    pattern = re.compile(r"\[SOURCE: (?P<label>.+?) \| page (?P<page>\d+) \| chunk (?P<chunk>[^\]]+)\]\s*(?P<text>.*?)(?=\n\[SOURCE:|\n## |\Z)", re.S)
    for match in pattern.finditer(source_labelled_context or ""):
        snippets.append(
            {
                "source": match.group("label"),
                "page": match.group("page"),
                "chunk_id": match.group("chunk"),
                "text": " ".join(match.group("text").split())[:900],
            }
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def metric_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install optional requirements and run: streamlit run streamlit_app.py")
        return

    st.set_page_config(page_title="RBI Temporal RAG Demo", layout="wide")
    examples = load_demo_examples()
    rows = load_comparison_rows()

    st.title("Temporal Multi-Document RAG for RBI Monetary Policy Reports")
    st.caption("Demo mode uses saved evaluated examples; live mode can be enabled separately with API keys.")

    with st.sidebar:
        st.subheader("Selected pipeline")
        st.write("Reports: April 2025, October 2025, April 2026")
        st.write("Retrieval: PyPDFLoader + Dense + BM25 + RRF + Cohere rerank-v3.5")
        st.write("Generation: Groq llama-3.1-8b-instant")
        st.write("Sufficiency gate: enabled in final dev generation")

    demo_tab, comparison_tab, limits_tab = st.tabs(["Ask / Demo", "Method comparison", "Limitations"])

    with demo_tab:
        options = [row.get("original_query") for row in examples if row.get("original_query")]
        default_examples = [
            "What happened to food inflation in June and July 2025 according to the October 2025 MPR?",
            "Compare the inflation outlook between April 2025 and October 2025.",
            "How did the policy stance and narrative evolution change across all three reports?",
            "What does the report say about a non-existent RBI MPR from 2030?",
        ]
        question = st.selectbox("Example questions", options or default_examples)
        query = st.text_area("Query", value=question, height=90)
        row = find_demo_answer(query, examples)
        if row is None:
            st.warning("No saved demo examples were found. Generate V2 sufficiency artifacts to populate this demo.")
        else:
            st.subheader("Generated answer")
            st.markdown(row.get("generated_answer") or "No generated answer saved for this example.")
            st.subheader("Sufficiency status")
            st.write(row.get("sufficiency_status", "not_available"))
            if row.get("sufficiency_reasons"):
                st.write(row["sufficiency_reasons"])
            st.subheader("Source citations")
            st.dataframe(row.get("citations") or [], use_container_width=True)
            st.subheader("Retrieved chunks grouped by report")
            for snippet in extract_context_snippets(row.get("source_labelled_context", "")):
                with st.expander(f"{snippet['source']} | page {snippet['page']} | {snippet['chunk_id']}"):
                    st.write(snippet["text"])
            st.subheader("Saved retrieval metadata")
            st.json(
                {
                    "question_id": row.get("question_id"),
                    "query_type": row.get("query_type"),
                    "required_report_ids": row.get("required_report_ids"),
                    "selected_pages": row.get("selected_pages"),
                    "retrieval_experiment_id": row.get("retrieval_experiment_id"),
                }
            )

    with comparison_tab:
        st.subheader("Best current system")
        retrieval = metric_row(rows, "V2 Cohere retrieval")
        generation = metric_row(rows, "V2 Cohere retrieval + sufficiency-gated generation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CER", retrieval.get("complete_evidence_recall"))
        c2.metric("All-Reports Hit", retrieval.get("all_reports_hit"))
        c3.metric("Evidence Recall", retrieval.get("evidence_recall"))
        c4.metric("Macro MRR", retrieval.get("macro_mrr"))
        st.write("Generation metrics")
        st.dataframe([generation], use_container_width=True)
        st.write("Retrieval method comparison")
        keep = [
            row
            for row in rows
            if row.get("method")
            in {
                "Single-document Hybrid RRF + reranker",
                "V2 baseline final retrieval",
                "V2 Cohere retrieval",
                "MMR / diversity selection",
                "True MMR lambda 0.6",
                "True MMR lambda 0.7",
                "True MMR lambda 0.8",
            }
        ]
        st.dataframe(keep, use_container_width=True)

    with limits_tab:
        st.markdown(
            """
            - Final V2 generation metrics are development-only.
            - The old Phase 7 held-out set is historical and is not a fresh V2 benchmark.
            - Generation metrics are deterministic heuristics, not human evaluation.
            - Cohere reranking improved retrieval quality but materially increased latency.
            - Unstructured extraction was attempted but remains blocked for these PDFs without OCR/Tesseract.
            - The system is demo/interview-ready, not production-ready.
            """
        )


if __name__ == "__main__":
    main()
