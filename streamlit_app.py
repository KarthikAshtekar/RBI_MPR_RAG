from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rbi_rag.streamlit_demo_helpers import (  # noqa: E402
    SELECTED_GENERATION,
    SELECTED_MODEL,
    SELECTED_PROMPT,
    SELECTED_RETRIEVAL,
    caveats,
    compact_method_rows,
    extract_context_snippets,
    find_demo_answer,
    group_citations,
    key_availability_status,
    load_final_metrics,
    load_saved_examples,
    mrr_mmr_explanation,
    pct,
    production_status_text,
    score,
    status_label,
)


def _answer_body(answer: str) -> str:
    if not answer:
        return "No saved answer is available for this example."
    if "Citations:" in answer:
        answer = answer.split("Citations:", 1)[0].strip()
    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer.strip()


def _install_css(st) -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
        .hero {
            border: 1px solid #d9e2ec;
            border-radius: 18px;
            padding: 1.05rem 1.3rem;
            background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
            margin-bottom: 1rem;
        }
        .badge {
            display: inline-block;
            padding: .25rem .65rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #1e3a8a;
            font-weight: 700;
            font-size: .82rem;
            margin-bottom: .65rem;
        }
        .subtle {color: #52616b; font-size: .98rem;}
        .hero h1 {font-size: 2.2rem; line-height: 1.18; margin-bottom: .65rem;}
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: .9rem;
            background: white;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .metric-label {color: #64748b; font-size: .82rem; margin-bottom: .25rem;}
        .metric-value {font-size: 1.35rem; font-weight: 800; color: #0f172a;}
        .answer-card {
            border: 1px solid #d1d5db;
            border-radius: 16px;
            padding: 1rem 1.15rem;
            background: #ffffff;
            margin-top: .25rem;
        }
        .evidence-card {
            border-left: 4px solid #2563eb;
            padding: .6rem .8rem;
            margin: .45rem 0;
            background: #f8fafc;
            border-radius: 8px;
        }
        .small-note {font-size: .9rem; color: #475569;}
        @media (max-width: 700px) {
            .hero {padding: 1rem;}
            .metric-value {font-size: 1.1rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(st, label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install optional requirements and run: streamlit run streamlit_app.py")
        return

    st.set_page_config(page_title="RBI Temporal RAG Demo", page_icon="📘", layout="wide")
    _install_css(st)

    metrics = load_final_metrics(ROOT)
    examples = load_saved_examples(ROOT)
    method_rows = compact_method_rows(ROOT)
    keys = key_availability_status()
    explainer_path = ROOT / "RBI_RAG_Project_Explainer.html"

    with st.sidebar:
        st.markdown("### Selected system")
        st.caption("Development-evaluated demo")
        st.write(f"Retrieval: `{SELECTED_RETRIEVAL}`")
        st.write(f"Generation: `{SELECTED_GENERATION}`")
        st.caption(f"Model: {SELECTED_MODEL}")
        st.caption(f"Prompt: {SELECTED_PROMPT}")
        st.divider()
        st.markdown("### Reports included")
        st.write("April 2025")
        st.write("October 2025")
        st.write("April 2026")
        st.divider()
        st.markdown("### Mode")
        st.success("Saved-example demo mode")
        st.caption("No API key is required to open or use the demo.")
        if explainer_path.exists():
            st.download_button(
                "Download explainer HTML",
                data=explainer_path.read_bytes(),
                file_name=explainer_path.name,
                mime="text/html",
                width="stretch",
            )
        with st.expander("Live mode status"):
            st.write("Live generation is not enabled in this UI polish build.")
            st.write(f"Groq key available to process: `{keys['groq']}`")
            st.write(f"Cohere key available to process: `{keys['cohere']}`")

    st.markdown(
        """
        <div class="hero">
            <div class="badge">Development-evaluated demo</div>
            <h1>Temporal Multi-Document RAG for RBI Monetary Policy Reports</h1>
            <p class="subtle">
            Answers policy questions across April 2025, October 2025, and April 2026 RBI MPRs
            with citations and sufficiency checks.
            </p>
            <p class="small-note">Not production-ready; demo/interview-ready with known limitations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r = metrics["retrieval"]
    g = metrics["generation"]
    metric_cols = st.columns(5)
    with metric_cols[0]:
        _metric_card(st, "Complete Evidence Recall", pct(r["complete_evidence_recall"]))
    with metric_cols[1]:
        _metric_card(st, "All-Reports Hit", pct(r["all_reports_hit"]))
    with metric_cols[2]:
        _metric_card(st, "Evidence Recall", pct(r["evidence_recall"]))
    with metric_cols[3]:
        _metric_card(st, "Factual Correctness", pct(g["factual_correctness"]))
    with metric_cols[4]:
        _metric_card(st, "Citation Correctness", pct(g["citation_correctness"]))

    with st.expander("Detailed development metrics"):
        st.markdown("These are development evaluation results, not held-out or production results.")
        if metrics.get("generation_source"):
            st.caption(f"Generation metrics loaded from `{metrics['generation_source']}`.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Retrieval**")
            st.table(
                [
                    {"Metric": "CER", "Value": score(r["complete_evidence_recall"])},
                    {"Metric": "All-Reports Hit", "Value": score(r["all_reports_hit"])},
                    {"Metric": "Evidence Recall", "Value": score(r["evidence_recall"])},
                    {"Metric": "Macro MRR", "Value": score(r["macro_mrr"])},
                ]
            )
        with c2:
            st.markdown("**Generation**")
            st.table(
                [
                    {"Metric": "Factual correctness", "Value": score(g["factual_correctness"])},
                    {"Metric": "Faithfulness", "Value": score(g["faithfulness_to_context"])},
                    {"Metric": "Contextual relevancy", "Value": score(g["contextual_relevancy"])},
                    {"Metric": "Contextual recall", "Value": score(g["contextual_recall"])},
                    {"Metric": "Abstention correctness", "Value": score(g["abstention_correctness"])},
                    {"Metric": "Citation completeness", "Value": score(g["citation_completeness"])},
                    {"Metric": "Temporal attribution", "Value": score(g["temporal_attribution_correctness"])},
                    {"Metric": "Comparative correctness", "Value": score(g["comparative_correctness"])},
                ]
            )

    st.divider()
    st.subheader("Project verification pack")
    st.caption("Use these artifacts to explain the full project without rerunning APIs.")
    pack_cols = st.columns(3)
    with pack_cols[0]:
        st.markdown("**Explainer HTML**")
        if explainer_path.exists():
            st.write("Root file: `RBI_RAG_Project_Explainer.html`")
            st.download_button(
                "Download HTML",
                data=explainer_path.read_bytes(),
                file_name=explainer_path.name,
                mime="text/html",
                key="download_explainer_main",
                width="content",
            )
        else:
            st.warning("Run `python scripts/generate_project_explainer_html.py` to create the root explainer.")
    with pack_cols[1]:
        st.markdown("**Notebook**")
        st.write("`Multi_Report_RAG_Explainer.ipynb` runs/explains the modular pipeline.")
    with pack_cols[2]:
        st.markdown("**Saved reports**")
        st.write("`reports/final_packaging/final_project_report.md`")
        st.write("`reports/final_comparison/rag_methods_master_comparison.md`")

    st.divider()
    st.subheader("Try a saved demo question")
    st.caption("Demo mode displays saved development-generation artifacts. It does not call Groq or Cohere.")

    options = [row.get("original_query") for row in examples if row.get("original_query")]
    fallback_options = [
        "What happened to food inflation in June and July 2025 according to the October 2025 MPR?",
        "Compare the inflation outlook between April 2025 and October 2025.",
        "How did the policy stance and narrative evolution change across all three reports?",
        "What does the report say about a non-existent RBI MPR from 2030?",
    ]
    demo_left, demo_right = st.columns([0.9, 1.1], gap="large")
    with demo_left:
        selected = st.selectbox("Example question", options or fallback_options, label_visibility="visible")
        custom_query = st.text_input("Optional custom wording", value=selected)
        st.button("Show saved demo answer", type="primary", width="content")
        st.caption("The nearest saved evaluated example is shown; no live model call is made.")

    row = find_demo_answer(custom_query, examples)
    with demo_right:
        if row is None:
            st.warning("No saved demo examples were found. The app still loads, but answer/citation demo content is unavailable.")
        else:
            label, kind = status_label(row.get("sufficiency_status"))
            if kind == "success":
                st.success(label)
            elif kind == "warning":
                st.warning(label)
            elif kind == "error":
                st.error(label)
            else:
                st.info(label)

            with st.container(border=True):
                st.markdown("#### Saved answer")
                st.write(_answer_body(row.get("generated_answer", "")))

            citation_groups = group_citations(row.get("citations"))
            st.markdown("#### Citations")
            if citation_groups:
                for period, citations in citation_groups.items():
                    with st.expander(f"{period} — {len(citations)} citation(s)", expanded=True):
                        for citation in citations:
                            page = citation.get("page") or citation.get("page_number") or "n/a"
                            chunk = citation.get("chunk_id") or "n/a"
                            st.markdown(f"- Page `{page}`, chunk `{chunk}`")
            else:
                st.caption("No parsed citations for this saved answer.")

    if row is not None:
        st.markdown("#### Retrieved evidence")
        snippets = extract_context_snippets(row.get("source_labelled_context", ""))
        if snippets:
            for period, items in snippets.items():
                with st.expander(f"{period} evidence", expanded=False):
                    for item in items:
                        st.markdown(
                            f"""
                            <div class="evidence-card">
                            <strong>Page {item['page']} · {item['chunk_id']}</strong><br/>
                            {item['text']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.caption("No source-labelled context was found for this saved answer.")

    st.divider()
    st.subheader("Method comparison")
    st.caption("Final selected system is shown first; full reports remain in `reports/final_comparison/`.")
    if method_rows:
        st.dataframe(method_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Method comparison artifact is missing. Final metrics above use the documented development fallback values.")
    with st.expander("MRR vs MMR distinction"):
        st.write(mrr_mmr_explanation())

    st.divider()
    st.subheader("Limitations")
    st.markdown("\n".join(f"- {item}" for item in caveats()))
    st.info(production_status_text())


if __name__ == "__main__":
    main()
