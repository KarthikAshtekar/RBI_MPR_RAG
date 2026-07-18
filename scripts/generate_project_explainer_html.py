from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "RBI_RAG_Project_Explainer.html"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_yaml(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return yaml.safe_load(path.read_text(encoding="utf-8")) or default


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.0f} ms"
    except (TypeError, ValueError):
        return "n/a"


def num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def text(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return escape(str(value))


def metric(summary: dict[str, Any], name: str) -> float | None:
    value = ((summary.get("metrics") or {}).get(name) or {}).get("mean_score")
    return float(value) if isinstance(value, (int, float)) else None


def first(rows: list[dict[str, Any]], **criteria: str) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == value for key, value in criteria.items()):
            return row
    return {}


def row(cells: list[str], *, header: bool = False) -> str:
    tag = "th" if header else "td"
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    body = "\n".join(row(values) for values in rows) or row(["No saved rows found."] + [""] * (len(headers) - 1))
    return (
        "<div class=\"table-wrap\"><table>"
        "<thead>"
        f"{row(headers, header=True)}"
        "</thead><tbody>"
        f"{body}"
        "</tbody></table></div>"
    )


def section(title: str, body: str, section_id: str) -> str:
    return f"<section id=\"{section_id}\"><h2>{escape(title)}</h2>{body}</section>"


def card(title: str, value: str, detail: str = "") -> str:
    detail_html = f"<p>{escape(detail)}</p>" if detail else ""
    return f"<div class=\"card\"><span>{escape(title)}</span><strong>{value}</strong>{detail_html}</div>"


def selected_metric_cards(mmr_best: dict[str, Any], suff_eval: dict[str, Any]) -> str:
    cards = [
        card("Best retrieval CER", pct(mmr_best.get("complete_evidence_recall")), "MMR_LAMBDA_06, development split"),
        card("Best retrieval evidence recall", pct(mmr_best.get("evidence_recall")), "MMR_LAMBDA_06"),
        card("Best retrieval Macro MRR", score(mmr_best.get("macro_report_mrr")), "MMR is a diversity selector; MRR is a ranking metric"),
        card("Factual correctness", pct(metric(suff_eval, "factual_correctness")), "V2 Cohere + sufficiency gate"),
        card("Citation correctness", pct(metric(suff_eval, "citation_correctness")), "Citations parsed against supplied chunks"),
        card("Abstention correctness", pct(metric(suff_eval, "abstention_correctness")), "Sufficiency-gated generation"),
    ]
    return "<div class=\"cards\">" + "\n".join(cards) + "</div>"


def report_rows(registry: dict[str, Any]) -> list[list[str]]:
    rows = []
    for report in registry.get("reports", []):
        path = ROOT / str(report.get("pdf_path", ""))
        rows.append([
            text(report.get("report_period")),
            text(report.get("report_id")),
            text(report.get("pdf_path")),
            "yes" if path.exists() else "missing",
        ])
    return rows


def single_doc_table(master_rows: list[dict[str, Any]]) -> str:
    rows = [
        r for r in master_rows
        if r.get("scope") == "April 2025 only" and r.get("retrieval_or_generation") == "retrieval"
    ]
    rows = sorted(rows, key=lambda item: item.get("sort_order") or 999)
    return table(
        ["Strategy", "Hit-Rate@4", "MRR", "Latency", "Notes"],
        [[text(r.get("method")), pct(r.get("hit_rate")), score(r.get("mrr")), ms(r.get("mean_latency_ms")), text(r.get("notes"))] for r in rows],
    )


def v2_table(v2: dict[str, Any]) -> str:
    rows = []
    for item in v2.get("completed", []):
        rows.append([
            text(item.get("experiment_id")),
            pct(item.get("complete_evidence_recall")),
            pct(item.get("all_reports_hit")),
            pct(item.get("evidence_recall")),
            score(item.get("macro_report_mrr")),
            ms(item.get("median_latency_ms")),
            text(item.get("reranker_provider")),
            text(item.get("parser_name")),
        ])
    for item in v2.get("skipped", []):
        rows.append([
            text(item.get("experiment_id")),
            "skipped",
            "skipped",
            "skipped",
            "skipped",
            "skipped",
            "n/a",
            text(item.get("reason")),
        ])
    return table(
        ["Experiment", "CER", "All-Reports Hit", "Evidence Recall", "Macro MRR", "Median latency", "Reranker", "Parser/status"],
        rows,
    )


def mmr_table(mmr: dict[str, Any]) -> str:
    rows = [
        [
            text(item.get("experiment_id")),
            pct(item.get("complete_evidence_recall")),
            pct(item.get("all_reports_hit")),
            pct(item.get("evidence_recall")),
            score(item.get("macro_report_mrr")),
            ms(item.get("median_latency_ms")),
            num(item.get("mean_estimated_tokens")),
            text(item.get("eligibility")),
        ]
        for item in mmr.get("completed", [])
    ]
    return table(
        ["Experiment", "CER", "All-Reports Hit", "Evidence Recall", "Macro MRR", "Median latency", "Mean tokens", "Eligibility"],
        rows,
    )


def generation_table(master_rows: list[dict[str, Any]], suff_eval: dict[str, Any]) -> str:
    before = first(master_rows, method="V2 Cohere retrieval + generation")
    after = first(master_rows, method="V2 Cohere retrieval + sufficiency-gated generation")
    rows = []
    if before:
        rows.append([
            "Before sufficiency gate",
            pct(before.get("factual_correctness")),
            pct(before.get("faithfulness_to_context")),
            pct(before.get("abstention_correctness")),
            pct(before.get("citation_correctness")),
            pct(before.get("temporal_attribution_correctness")),
            pct(before.get("comparative_correctness")),
        ])
    if after:
        rows.append([
            "After sufficiency gate",
            pct(after.get("factual_correctness")),
            pct(after.get("faithfulness_to_context")),
            pct(after.get("abstention_correctness")),
            pct(after.get("citation_correctness")),
            pct(after.get("temporal_attribution_correctness")),
            pct(after.get("comparative_correctness")),
        ])
    if not rows and suff_eval:
        rows.append([
            "After sufficiency gate",
            pct(metric(suff_eval, "factual_correctness")),
            pct(metric(suff_eval, "faithfulness_to_context")),
            pct(metric(suff_eval, "abstention_correctness")),
            pct(metric(suff_eval, "citation_correctness")),
            pct(metric(suff_eval, "temporal_attribution_correctness")),
            pct(metric(suff_eval, "comparative_correctness")),
        ])
    return table(
        ["Generation setting", "Factual", "Faithfulness", "Abstention", "Citation", "Temporal attribution", "Comparative"],
        rows,
    )


def category_table(category_rows: list[dict[str, Any]]) -> str:
    wanted = [
        ("V2_BASELINE_FINAL", "table_or_numeric_questions", "True"),
        ("V2_COHERE_ONLY", "table_or_numeric_questions", "True"),
        ("V2_BASELINE_FINAL", "source_structure", "table"),
        ("V2_COHERE_ONLY", "source_structure", "table"),
    ]
    rows = []
    for experiment, category_type, category_name in wanted:
        item = next(
            (
                r for r in category_rows
                if r.get("experiment_id") == experiment
                and r.get("category_type") == category_type
                and str(r.get("category")) == category_name
            ),
            {},
        )
        if item:
            label = "table/numeric questions" if category_type == "table_or_numeric_questions" else "table source pages"
            rows.append([
                text(experiment),
                label,
                text(item.get("case_count")),
                pct(item.get("complete_evidence_recall")),
                pct(item.get("evidence_recall")),
                score(item.get("macro_report_mrr")),
            ])
    return table(["Experiment", "Slice", "Cases", "CER", "Evidence Recall", "Macro MRR"], rows)


def artifact_list() -> str:
    artifacts = [
        "README.md",
        "Multi_Report_RAG_Explainer.ipynb",
        "streamlit_app.py",
        "reports/final_packaging/final_project_report.md",
        "reports/final_comparison/rag_methods_master_comparison.md",
        "reports/v2_unstructured_cohere/v2_experiment_leaderboard.md",
        "reports/mmr_experiments/mmr_leaderboard.md",
        "reports/v2_sufficiency/sufficiency_results_for_presentation.md",
    ]
    items = []
    for rel in artifacts:
        exists = (ROOT / rel).exists()
        items.append(f"<li><code>{escape(rel)}</code> <span class=\"pill {'ok' if exists else 'warn'}\">{'available' if exists else 'missing'}</span></li>")
    return "<ul class=\"artifact-list\">" + "\n".join(items) + "</ul>"


def build_html() -> str:
    registry = read_yaml(ROOT / "configs/reports.yaml", {})
    master_rows = read_json(ROOT / "reports/final_comparison/rag_methods_master_comparison.json", [])
    v2 = read_json(ROOT / "reports/v2_unstructured_cohere/v2_experiment_leaderboard.json", {"completed": [], "skipped": []})
    mmr = read_json(ROOT / "reports/mmr_experiments/mmr_leaderboard.json", {"completed": []})
    category_rows = read_json(ROOT / "reports/v2_unstructured_cohere/v2_category_results.json", [])
    suff_eval = read_json(ROOT / "reports/v2_sufficiency/dev_sufficiency_eval_summary.json", {"metrics": {}})
    mmr_best = next((row for row in mmr.get("completed", []) if row.get("experiment_id") == "MMR_LAMBDA_06"), {})
    v2_cohere = next((row for row in v2.get("completed", []) if row.get("experiment_id") == "V2_COHERE_ONLY"), {})
    v2_baseline = next((row for row in v2.get("completed", []) if row.get("experiment_id") == "V2_BASELINE_FINAL"), {})
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cohere_delta = (
        float(v2_cohere.get("complete_evidence_recall", 0)) - float(v2_baseline.get("complete_evidence_recall", 0))
        if v2_cohere and v2_baseline else None
    )
    mmr_delta = (
        float(mmr_best.get("complete_evidence_recall", 0)) - float(v2_cohere.get("complete_evidence_recall", 0))
        if mmr_best and v2_cohere else None
    )

    nav = """
    <nav>
      <a href="#overview">Overview</a>
      <a href="#data">Data</a>
      <a href="#pipeline">Pipeline</a>
      <a href="#strategies">Strategies</a>
      <a href="#results">Results</a>
      <a href="#insights">Insights</a>
      <a href="#limitations">Limitations</a>
      <a href="#future">Future scope</a>
    </nav>
    """

    overview = section(
        "Project overview",
        f"""
        <p>This project is a <strong>Temporal multi-document RAG for RBI Monetary Policy Reports</strong>.
        It answers questions that may require one, two, or all three RBI MPRs while preserving report-period attribution.
        The framing is <strong>policy stance and narrative evolution</strong>, not generic sentiment analysis.</p>
        {selected_metric_cards(mmr_best, suff_eval)}
        <div class="callout">
          <strong>Current honest status:</strong> best retrieval-only setting is <code>MMR_LAMBDA_06</code>.
          Best evaluated answer-generation setting in the current checkout is <code>V2_COHERE_ONLY + Groq + sufficiency gate</code>.
          The project is demo/interview-ready, <strong>not production-ready</strong>.
        </div>
        """,
        "overview",
    )

    data = section(
        "Data and evaluation design",
        f"""
        <p>The corpus contains three RBI Monetary Policy Reports. The evaluation evolved from an April 2025
        single-report baseline into a three-report temporal benchmark with single-report, pairwise, all-report trend,
        and unsupported-period questions.</p>
        {table(["Report period", "Report ID", "PDF path", "File status"], report_rows(registry))}
        <p class="note">Single-document Hit-Rate@4/MRR is preserved for historical comparison only. Multi-report retrieval
        uses stricter metrics: Complete Evidence Recall, All-Reports Hit, Evidence Recall, Macro Report MRR,
        report coverage, contamination, latency, and context size.</p>
        """,
        "data",
    )

    pipeline = section(
        "End-to-end pipeline",
        """
        <div class="pipeline">
          <div><b>1. PDF ingestion</b><span>PyPDFLoader remained the valid parser. Unstructured was tested but blocked by OCR/Tesseract requirements.</span></div>
          <div><b>2. Chunking</b><span>Report-aware chunks preserve source report, page, chunk ID, and metadata.</span></div>
          <div><b>3. Retrieval</b><span>Dense MiniLM and BM25 candidates are retrieved per required report.</span></div>
          <div><b>4. Fusion</b><span>Reciprocal Rank Fusion merges dense and lexical candidates.</span></div>
          <div><b>5. Reranking</b><span>Cohere rerank-v3.5 improves development retrieval quality over the local cross-encoder baseline.</span></div>
          <div><b>6. Context selection</b><span>Report-aware quotas and MMR reduce repeated evidence and preserve temporal coverage.</span></div>
          <div><b>7. Sufficiency gate</b><span>Classifies evidence as sufficient, partial, or insufficient before generation.</span></div>
          <div><b>8. Generation</b><span>Groq Llama 3.1 8B answers from source-labelled contexts with citations.</span></div>
        </div>
        """,
        "pipeline",
    )

    strategies = section(
        "Strategies evaluated",
        f"""
        <h3>Original one-document strategies</h3>
        {single_doc_table(master_rows)}
        <h3>Controlled V2 parser/reranker comparison</h3>
        {v2_table(v2)}
        <h3>MMR context-selection comparison</h3>
        {mmr_table(mmr)}
        """,
        "strategies",
    )

    results = section(
        "Key results",
        f"""
        <h3>Generation quality</h3>
        {generation_table(master_rows, suff_eval)}
        <h3>Table and numeric evidence recovery</h3>
        {category_table(category_rows)}
        <div class="callout">
          Cohere improved Complete Evidence Recall by <strong>{pct(cohere_delta)}</strong> over the V2 local-reranker baseline.
          MMR_LAMBDA_06 then improved Complete Evidence Recall by <strong>{pct(mmr_delta)}</strong> over V2 Cohere retrieval,
          while slightly lowering Macro MRR versus V2 Cohere. That is expected because MMR optimizes context diversity,
          not pure rank position.
        </div>
        """,
        "results",
    )

    insights = section(
        "Practical insights",
        """
        <ul>
          <li><strong>Multi-report retrieval is stricter than single-report retrieval.</strong> A question can hit one report and still fail Complete Evidence Recall if another required report is missing.</li>
          <li><strong>Cohere helped retrieval quality.</strong> It improved Complete Evidence Recall, Evidence Recall, Macro MRR, and table/numeric retrieval versus the V2 local-reranker baseline, but added latency.</li>
          <li><strong>MMR helped context completeness.</strong> It improved Complete Evidence Recall and All-Reports Hit by reducing redundancy, even though Macro MRR was slightly lower.</li>
          <li><strong>The sufficiency gate helped generation behavior.</strong> It improved abstention correctness and faithfulness by forcing the answer path to caveat or abstain when retrieval was incomplete.</li>
          <li><strong>Tables/charts remain hard.</strong> PyPDFLoader flattens layout. Unstructured needs a deliberate OCR/Tesseract setup before it can be fairly retested here.</li>
        </ul>
        """,
        "insights",
    )

    artifacts = section(
        "Verification artifacts",
        f"""
        <p>The explainer is backed by saved local artifacts, not live API calls.</p>
        {artifact_list()}
        """,
        "artifacts",
    )

    limitations = section(
        "Limitations and scientific caveats",
        """
        <ul>
          <li>Most final answer-quality metrics are development-only and deterministic heuristic evaluations, not human evaluation.</li>
          <li>The old Phase 7 held-out set must not be presented as a fresh unbiased V2 benchmark.</li>
          <li>Final-generation bake-off top-level artifacts are incomplete in the current checkout, so the safest evaluated generation claim remains V2 Cohere retrieval plus sufficiency-gated generation.</li>
          <li>Unstructured extraction was attempted but blocked for these PDFs because OCR fallback requires Tesseract.</li>
          <li>No API keys are embedded in this HTML; live Groq/Cohere execution still depends on local environment variables.</li>
          <li>The system is not production-ready.</li>
        </ul>
        """,
        "limitations",
    )

    future = section(
        "Future scope",
        """
        <ol>
          <li>Create a fresh V2 held-out evaluation set and keep it sealed until final selection.</li>
          <li>Add human or LLM-judge answer-quality review with robust failure/null-score handling.</li>
          <li>Install Tesseract/OCR support and rerun Unstructured extraction as a real layout-aware comparison.</li>
          <li>Cache Cohere reranker calls and separate retrieval latency from API latency in the dashboard.</li>
          <li>Add history-aware query rewriting for conversational follow-ups.</li>
          <li>Promote the Streamlit app from saved-example demo mode to guarded live-query mode.</li>
          <li>Add deployment hardening, observability, rate-limit handling, and monitoring before any production claim.</li>
        </ol>
        """,
        "future",
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RBI Temporal RAG Project Explainer</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #52616b;
      --line: #d9e2ec;
      --blue: #1d4ed8;
      --blue-soft: #dbeafe;
      --green: #047857;
      --amber: #b45309;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 56px min(7vw, 88px) 34px;
      background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
      color: white;
    }}
    header p {{ max-width: 980px; color: #dbeafe; font-size: 1.08rem; }}
    h1 {{ margin: 0 0 14px; font-size: clamp(2rem, 4vw, 4rem); line-height: 1.05; letter-spacing: -0.04em; }}
    h2 {{ margin: 0 0 18px; font-size: 1.7rem; letter-spacing: -0.02em; }}
    h3 {{ margin: 26px 0 12px; }}
    code {{ background: #eff6ff; color: #1e3a8a; padding: 2px 6px; border-radius: 6px; }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      padding: 12px min(7vw, 88px);
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(8px);
    }}
    nav a {{ color: var(--blue); text-decoration: none; font-weight: 700; font-size: 0.92rem; }}
    main {{ padding: 28px min(7vw, 88px) 70px; }}
    section {{
      margin: 0 0 28px;
      padding: 26px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      background: var(--blue-soft);
      color: #1e3a8a;
      font-weight: 700;
      font-size: 0.86rem;
    }}
    .pill.ok {{ background: #dcfce7; color: #166534; }}
    .pill.warn {{ background: #fef3c7; color: #92400e; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin: 20px 0;
    }}
    .card {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fbfdff;
    }}
    .card span {{ display: block; color: var(--muted); font-size: 0.84rem; }}
    .card strong {{ display: block; margin: 3px 0; font-size: 1.5rem; }}
    .card p {{ margin: 0; color: var(--muted); font-size: 0.86rem; }}
    .table-wrap {{ overflow-x: auto; margin: 14px 0 20px; border: 1px solid var(--line); border-radius: 14px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    th {{ background: #f1f5f9; font-size: 0.86rem; color: #334155; }}
    td {{ font-size: 0.92rem; }}
    tr:last-child td {{ border-bottom: none; }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }}
    .pipeline div {{
      padding: 15px;
      border-left: 4px solid var(--blue);
      background: #f8fafc;
      border-radius: 12px;
    }}
    .pipeline b {{ display: block; margin-bottom: 6px; }}
    .pipeline span, .note, .callout {{ color: var(--muted); }}
    .callout {{
      margin: 18px 0 4px;
      padding: 14px 16px;
      border: 1px solid #bfdbfe;
      border-left: 5px solid var(--blue);
      border-radius: 12px;
      background: #eff6ff;
    }}
    .artifact-list li {{ margin: 8px 0; }}
    footer {{ color: var(--muted); padding: 0 min(7vw, 88px) 40px; }}
  </style>
</head>
<body>
  <header>
    <h1>RBI Temporal Multi-Document RAG Explainer</h1>
    <p>Verification companion for the RBI Monetary Policy Report RAG project: data, pipeline, retrieval/generation strategies,
    results, insights, limitations, and future scope.</p>
    <div class="meta">
      <span class="pill">Generated {escape(created)}</span>
      <span class="pill">Offline static HTML</span>
      <span class="pill warn">Not production-ready</span>
    </div>
  </header>
  {nav}
  <main>
    {overview}
    {data}
    {pipeline}
    {strategies}
    {results}
    {insights}
    {artifacts}
    {limitations}
    {future}
  </main>
  <footer>
    Generated from saved local artifacts by <code>scripts/generate_project_explainer_html.py</code>.
  </footer>
</body>
</html>
"""


def main() -> int:
    html = build_html()
    OUT_PATH.write_text(html, encoding="utf-8")
    print(json.dumps({"status": "written", "path": str(OUT_PATH), "bytes": OUT_PATH.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
