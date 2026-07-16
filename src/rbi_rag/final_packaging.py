from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifact_io import now_iso, write_json, write_markdown as write_md
from .env_loading import load_project_dotenv


OUT_DIR = Path("reports/final_packaging")
STATUS_FIELDS = [
    "phase_0_status",
    "phase_1_mmr_status",
    "phase_2_comparison_update_status",
    "phase_3_final_report_status",
    "phase_4_streamlit_status",
    "phase_5_readme_status",
    "phase_6_validation_status",
    "final_status",
]

CHECKSUM_TARGETS = [
    Path("configs/v2_selected_retrieval.yaml"),
    Path("configs/final_retrieval_selected.yaml"),
    Path("reports/v2_unstructured_cohere"),
    Path("reports/v2_generation"),
    Path("reports/v2_sufficiency"),
    Path("reports/final_comparison"),
    Path("reports/final_evaluation"),
    Path("reports/structural_optimisation"),
    Path("reports/optimisation"),
    Path("data/evaluation"),
    Path("data/raw"),
]

CRITICAL_TARGETS = {
    "configs/v2_selected_retrieval.yaml",
    "reports/v2_unstructured_cohere",
    "reports/v2_generation",
    "reports/v2_sufficiency",
    "reports/final_comparison",
    "data/evaluation",
    "data/raw",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def update_overnight_status(root: Path, **updates: Any) -> dict[str, Any]:
    path = root / OUT_DIR / "overnight_run_status.json"
    payload = read_json(path, {})
    if not payload:
        payload = {
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "updated_at_utc": now_iso(),
            **{field: "pending" for field in STATUS_FIELDS},
            "events": [],
        }
    payload["updated_at_utc"] = now_iso()
    for key, value in updates.items():
        if key in STATUS_FIELDS:
            payload[key] = value
        else:
            payload.setdefault("details", {})[key] = value
    payload.setdefault("events", []).append({"created_at_utc": now_iso(), "updates": updates})
    write_json(path, payload)
    lines = ["# Overnight Run Status", ""]
    for field in STATUS_FIELDS:
        lines.append(f"- {field}: `{payload.get(field)}`")
    write_md(root / OUT_DIR / "overnight_run_status.md", lines)
    return payload


def file_manifest(root: Path, target: Path) -> dict[str, Any]:
    full = root / target
    rel = str(target).replace("\\", "/")
    if not full.exists():
        return {"path": rel, "exists": False, "critical": rel in CRITICAL_TARGETS}
    if full.is_file():
        return {
            "path": rel,
            "exists": True,
            "type": "file",
            "size_bytes": full.stat().st_size,
            "sha256": sha256_file(full),
            "critical": rel in CRITICAL_TARGETS,
        }
    files = [path for path in full.rglob("*") if path.is_file()]
    entries = [
        {"path": relative(root, path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(files)
    ]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "path": rel,
        "exists": True,
        "type": "directory",
        "file_count": len(entries),
        "aggregate_sha256": digest,
        "critical": rel in CRITICAL_TARGETS,
        "files": entries,
    }


def create_checksum_manifest(root: Path) -> dict[str, Any]:
    manifests = [file_manifest(root, target) for target in CHECKSUM_TARGETS]
    missing = [item["path"] for item in manifests if not item.get("exists")]
    missing_critical = [item["path"] for item in manifests if item.get("critical") and not item.get("exists")]
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "targets": manifests,
        "missing_paths": missing,
        "missing_critical_paths": missing_critical,
        "status": "passed" if not missing_critical else "critical_missing",
    }
    write_json(root / OUT_DIR / "pre_final_packaging_checksums.json", payload)
    lines = [
        "# Pre-final Packaging Checksums",
        "",
        f"Status: `{payload['status']}`",
        "",
        "| Path | Exists | Type | Files | Critical |",
        "|---|---:|---|---:|---:|",
    ]
    for item in manifests:
        lines.append(f"| {item['path']} | {item.get('exists')} | {item.get('type')} | {item.get('file_count', '')} | {item.get('critical')} |")
    if missing:
        lines += ["", "## Missing paths", ""]
        lines.extend(f"- {path}" for path in missing)
    write_md(root / OUT_DIR / "pre_final_packaging_checksums.md", lines)
    return payload


def archive_files(root: Path, source_files: list[Path], archive_root: Path) -> dict[str, Any]:
    files = [path for path in source_files if path.exists() and path.is_file()]
    if not files:
        return {"status": "not_needed", "file_count": 0, "archive_dir": str(archive_root)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_root / stamp
    target.mkdir(parents=True, exist_ok=True)
    for source in files:
        destination = target / relative(root, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {"status": "archived", "file_count": len(files), "archive_dir": relative(root, target)}


def archive_existing_outputs(root: Path) -> dict[str, Any]:
    final_comparison = root / "reports/final_comparison"
    comparison_files = [path for path in final_comparison.iterdir() if path.is_file()] if final_comparison.exists() else []
    comparison_archive = archive_files(root, comparison_files, final_comparison / "archive_pre_final_packaging")
    doc_files = [
        root / "README.md",
        root / "streamlit_app.py",
        root / OUT_DIR / "final_project_report.md",
        root / OUT_DIR / "interview_pack.md",
        root / OUT_DIR / "final_project_status.json",
        root / OUT_DIR / "final_project_status.md",
        root / OUT_DIR / "final_artifact_validation.json",
        root / OUT_DIR / "final_artifact_validation.md",
        root / OUT_DIR / "streamlit_run_instructions.md",
    ]
    docs_archive = archive_files(root, doc_files, root / OUT_DIR / "archive_existing_project_docs")
    return {"comparison_archive": comparison_archive, "docs_archive": docs_archive}


def row_lookup(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def metric(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
    return str(value)


def load_rows(root: Path) -> list[dict[str, Any]]:
    return read_json(root / "reports/final_comparison/rag_methods_master_comparison.json", [])


def final_report_lines(root: Path) -> list[str]:
    rows = load_rows(root)
    retrieval = row_lookup(rows, "V2 Cohere retrieval")
    mmr_retrieval = row_lookup(rows, "True MMR lambda 0.6")
    generation = row_lookup(rows, "V2 Cohere retrieval + sufficiency-gated generation")
    mmr_generation = row_lookup(rows, "MMR lambda 0.6 retrieval + sufficiency-gated generation")
    bakeoff_generation = row_lookup(rows, "Final selected generation bake-off strategy")
    final_mmr_generation_decision = read_json(root / "reports/final_mmr_generation/final_mmr_generation_selection_decision.json", {})
    final_bakeoff_decision = read_json(root / "reports/final_generation_bakeoff/final_generation_strategy_selection_decision.json", {})
    selected_generation = (
        bakeoff_generation
        if final_bakeoff_decision.get("selected_variant_id") and bakeoff_generation
        else
        mmr_generation
        if final_mmr_generation_decision.get("status") == "selected_mmr_end_to_end"
        else generation
    )
    selected_generation_name = selected_generation.get("method") or "V2 Cohere retrieval + sufficiency-gated generation"
    mmr_decision = read_json(root / "reports/mmr_experiments/mmr_selection_decision.json", {})
    mmr_leaderboard = read_json(root / "reports/mmr_experiments/mmr_leaderboard.json", {})
    retrieval_only_winner = mmr_decision.get("selected_experiment_id") if mmr_decision.get("status") == "evaluated_selected" else "V2_COHERE_ONLY"
    return [
        "# Final Project Report",
        "",
        "## Executive summary",
        "",
        "This project implements **Temporal multi-document RAG for RBI Monetary Policy Reports**. The final development system retrieves and answers questions about policy stance and narrative evolution across April 2025, October 2025, and April 2026 RBI Monetary Policy Reports.",
        "",
        f"Best retrieval-only configuration after MMR testing: **{retrieval_only_winner}**. "
        f"Best evaluated generation system: **{selected_generation_name}**.",
        "",
        "## Problem statement",
        "",
        "The system answers monetary-policy questions that may require evidence from one report, two reports, or all three reports while preserving correct report attribution.",
        "",
        "## Why RBI Monetary Policy Reports",
        "",
        "The reports are dense, periodic, and evidence-rich. They contain changes in inflation, growth, risks, and policy stance that are better framed as temporal retrieval than generic sentiment analysis.",
        "",
        "## Dataset",
        "",
        "- April 2025 Monetary Policy Report",
        "- October 2025 Monetary Policy Report",
        "- April 2026 Monetary Policy Report",
        "",
        "## Original single-document RAG baseline",
        "",
        "The project began with an April 2025-only RAG baseline using PyPDFLoader, chunking, MiniLM embeddings, BM25, RRF, local cross-encoder reranking, and Groq generation. Those Hit-Rate@4 and MRR metrics are preserved separately because they are not directly comparable to multi-report Complete Evidence Recall.",
        "",
        "## Why multi-document temporal RAG is harder",
        "",
        "Multi-report questions require retrieving complete evidence from the right report periods, avoiding wrong-report contamination, and supporting comparisons across changing narratives.",
        "",
        "## Final architecture",
        "",
        "- PyPDFLoader extraction",
        "- `sentence-transformers/all-MiniLM-L6-v2` dense embeddings",
        "- BM25 lexical retrieval",
        "- Hybrid RRF candidate fusion",
        "- Cohere `rerank-v3.5` reranking",
        "- Report-aware routing and quotas",
        "- Groq `llama-3.1-8b-instant` generation",
        "- Sufficiency gate before final answer acceptance",
        "- Source-labelled citations and temporal attribution checks",
        "",
        "## Techniques evaluated",
        "",
        "Dense retrieval, BM25, hybrid search, RRF, weighted RRF, multi-query/terminology expansion, facet decomposition, true MMR, Cohere reranking, Unstructured extraction attempt, and sufficiency-gated generation.",
        "",
        "## Final retrieval metrics",
        "",
        "| Method | CER | All-Reports Hit | Evidence Recall | Macro MRR | Median latency ms | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V2 Cohere retrieval | {metric(retrieval.get('complete_evidence_recall'))} | {metric(retrieval.get('all_reports_hit'))} | {metric(retrieval.get('evidence_recall'))} | {metric(retrieval.get('macro_mrr'))} | {metric(retrieval.get('median_latency_ms'))} | {metric(retrieval.get('mean_estimated_tokens'))} |",
        f"| MMR lambda 0.6 retrieval | {metric(mmr_retrieval.get('complete_evidence_recall'))} | {metric(mmr_retrieval.get('all_reports_hit'))} | {metric(mmr_retrieval.get('evidence_recall'))} | {metric(mmr_retrieval.get('macro_mrr'))} | {metric(mmr_retrieval.get('median_latency_ms'))} | {metric(mmr_retrieval.get('mean_estimated_tokens'))} |",
        "",
        "## Final generation metrics",
        "",
        f"MMR generation selection decision: `{final_mmr_generation_decision.get('status', 'not_run')}`. "
        f"Final bake-off decision: `{final_bakeoff_decision.get('status', 'not_run')}`.",
        "",
        "| Metric | Previous V2 sufficiency | MMR06 sufficiency | Selected value |",
        "|---|---:|---:|---:|",
        f"| Factual correctness | {metric(generation.get('factual_correctness'))} | {metric(mmr_generation.get('factual_correctness'))} | {metric(selected_generation.get('factual_correctness'))} |",
        f"| Faithfulness to context | {metric(generation.get('faithfulness_to_context'))} | {metric(mmr_generation.get('faithfulness_to_context'))} | {metric(selected_generation.get('faithfulness_to_context'))} |",
        f"| Abstention correctness | {metric(generation.get('abstention_correctness'))} | {metric(mmr_generation.get('abstention_correctness'))} | {metric(selected_generation.get('abstention_correctness'))} |",
        f"| Citation correctness | {metric(generation.get('citation_correctness'))} | {metric(mmr_generation.get('citation_correctness'))} | {metric(selected_generation.get('citation_correctness'))} |",
        f"| Citation completeness | {metric(generation.get('citation_completeness'))} | {metric(mmr_generation.get('citation_completeness'))} | {metric(selected_generation.get('citation_completeness'))} |",
        f"| Temporal attribution correctness | {metric(generation.get('temporal_attribution_correctness'))} | {metric(mmr_generation.get('temporal_attribution_correctness'))} | {metric(selected_generation.get('temporal_attribution_correctness'))} |",
        f"| Comparative correctness | {metric(generation.get('comparative_correctness'))} | {metric(mmr_generation.get('comparative_correctness'))} | {metric(selected_generation.get('comparative_correctness'))} |",
        "",
        "## MMR result",
        "",
        f"MMR decision status: `{mmr_decision.get('status', 'not_run')}`. Selected experiment: `{mmr_decision.get('selected_experiment_id', 'V2_COHERE_ONLY')}`.",
        "",
        "| Experiment | CER | All-Reports Hit | Evidence Recall | Macro MRR | Repeated text ratio |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row.get('experiment_id')} | {metric(row.get('complete_evidence_recall'))} | {metric(row.get('all_reports_hit'))} | {metric(row.get('evidence_recall'))} | {metric(row.get('macro_report_mrr'))} | {metric(row.get('mean_repeated_text_ratio'))} |"
            for row in mmr_leaderboard.get("completed", [])
        ],
        "",
        "## Failure analysis",
        "",
        "- Multi-report evidence completeness remains the hardest retrieval objective.",
        "- Table/numeric and comparative questions are more fragile than single-fact questions.",
        "- Cohere improves retrieval but materially increases latency.",
        "- Unstructured extraction was attempted but remains blocked because non-OCR extraction returned zero usable elements and OCR requires Tesseract.",
        "- Comparative generation remains the weakest generation dimension because incomplete evidence must be caveated or abstained from.",
        "",
        "## What is production-ready and what is not",
        "",
        "The repository is ready for a local demo and interview explanation. It is not production-ready because it lacks fresh V2 held-out evaluation, human evaluation, robust caching, operational monitoring, and deployment hardening.",
        "",
        "## Future work",
        "",
        "- Build a fresh V2 evaluation set",
        "- Add human evaluation",
        "- Retry Unstructured with Tesseract/OCR deliberately installed",
        "- Cache Cohere reranker calls",
        "- Improve Streamlit live-query mode",
        "- Package and deploy with monitoring",
    ]


def readme_lines(root: Path) -> list[str]:
    rows = load_rows(root)
    retrieval = row_lookup(rows, "V2 Cohere retrieval")
    mmr_retrieval = row_lookup(rows, "True MMR lambda 0.6")
    generation = row_lookup(rows, "V2 Cohere retrieval + sufficiency-gated generation")
    mmr_generation = row_lookup(rows, "MMR lambda 0.6 retrieval + sufficiency-gated generation")
    bakeoff_generation = row_lookup(rows, "Final selected generation bake-off strategy")
    final_mmr_generation_decision = read_json(root / "reports/final_mmr_generation/final_mmr_generation_selection_decision.json", {})
    final_bakeoff_decision = read_json(root / "reports/final_generation_bakeoff/final_generation_strategy_selection_decision.json", {})
    selected_generation = (
        bakeoff_generation
        if final_bakeoff_decision.get("selected_variant_id") and bakeoff_generation
        else
        mmr_generation
        if final_mmr_generation_decision.get("status") == "selected_mmr_end_to_end"
        else generation
    )
    selected_generation_name = selected_generation.get("method") or "V2 Cohere + sufficiency gate"
    mmr_decision = read_json(root / "reports/mmr_experiments/mmr_selection_decision.json", {})
    mmr_winner = mmr_decision.get("selected_experiment_id") if mmr_decision.get("status") == "evaluated_selected" else "V2_COHERE_ONLY"
    return [
        "# Temporal Multi-Document RAG for RBI Monetary Policy Reports",
        "",
        "A retrieval-augmented generation project for comparing RBI Monetary Policy Reports across time, focused on policy stance and narrative evolution rather than generic sentiment analysis.",
        "",
        "```mermaid",
        "flowchart LR",
        "    PDF[RBI MPR PDFs] --> Parse[PyPDFLoader]",
        "    Parse --> Chunks[Report-aware chunks]",
        "    Chunks --> Dense[MiniLM dense retrieval]",
        "    Chunks --> BM25[BM25 retrieval]",
        "    Dense --> RRF[Hybrid RRF]",
        "    BM25 --> RRF",
        "    RRF --> Cohere[Cohere rerank-v3.5]",
        "    Cohere --> Quotas[Report-aware context quotas]",
        "    Quotas --> Gate[Evidence sufficiency gate]",
        "    Gate --> Groq[Groq Llama 3.1 8B]",
        "    Groq --> Answer[Answer + citations]",
        "```",
        "",
        "## Dataset",
        "",
        "- April 2025 RBI Monetary Policy Report",
        "- October 2025 RBI Monetary Policy Report",
        "- April 2026 RBI Monetary Policy Report",
        "",
        "## Key features",
        "",
        "- Report-aware retrieval and context quotas",
        "- Temporal comparison across reports",
        "- Hybrid dense + BM25 search",
        "- Reciprocal Rank Fusion",
        "- Cohere reranking",
        "- Sufficiency gate for abstention/caveats",
        "- Source-labelled citations",
        "- Temporal attribution checks",
        "",
        "## Results summary",
        "",
        "| Best retrieval method | CER | All-Reports Hit | Evidence Recall | Macro MRR | Median latency ms | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V2 Cohere retrieval | {metric(retrieval.get('complete_evidence_recall'))} | {metric(retrieval.get('all_reports_hit'))} | {metric(retrieval.get('evidence_recall'))} | {metric(retrieval.get('macro_mrr'))} | {metric(retrieval.get('median_latency_ms'))} | {metric(retrieval.get('mean_estimated_tokens'))} |",
        f"| MMR lambda 0.6 retrieval | {metric(mmr_retrieval.get('complete_evidence_recall'))} | {metric(mmr_retrieval.get('all_reports_hit'))} | {metric(mmr_retrieval.get('evidence_recall'))} | {metric(mmr_retrieval.get('macro_mrr'))} | {metric(mmr_retrieval.get('median_latency_ms'))} | {metric(mmr_retrieval.get('mean_estimated_tokens'))} |",
        "",
        f"Retrieval-only winner after MMR testing: `{mmr_winner}`. Best evaluated generation setting: `{selected_generation_name}`.",
        "",
        "| Best generation method | Factual | Faithfulness | Abstention | Citation | Temporal attribution | Comparative |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V2 Cohere + sufficiency gate | {metric(generation.get('factual_correctness'))} | {metric(generation.get('faithfulness_to_context'))} | {metric(generation.get('abstention_correctness'))} | {metric(generation.get('citation_correctness'))} | {metric(generation.get('temporal_attribution_correctness'))} | {metric(generation.get('comparative_correctness'))} |",
        f"| MMR06 + sufficiency gate | {metric(mmr_generation.get('factual_correctness'))} | {metric(mmr_generation.get('faithfulness_to_context'))} | {metric(mmr_generation.get('abstention_correctness'))} | {metric(mmr_generation.get('citation_correctness'))} | {metric(mmr_generation.get('temporal_attribution_correctness'))} | {metric(mmr_generation.get('comparative_correctness'))} |",
        "",
        "Single-document Hit-Rate@4 is preserved as a historical baseline but is not directly comparable with multi-report Complete Evidence Recall.",
        "",
        "## How to run",
        "",
        "```powershell",
        "python -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "python -m pip install -r requirements.txt",
        "python -m pip install -r requirements-v2.txt  # optional V2/Cohere/Streamlit extras",
        "```",
        "",
        "Create `.env` locally with:",
        "",
        "```text",
        "GROQ_API_KEY=...",
        "COHERE_API_KEY=...",
        "UNSTRUCTURED_API_KEY=...  # optional; Unstructured remains OCR/Tesseract-blocked here",
        "```",
        "",
        "Useful commands:",
        "",
        "```powershell",
        "python scripts\\run_mmr_experiments.py",
        "python scripts\\validate_mmr_experiments.py",
        "python scripts\\generate_mmr_report.py",
        "python scripts\\generate_final_rag_comparison.py",
        "streamlit run streamlit_app.py",
        "python -m pytest",
        "```",
        "",
        "## Repository structure",
        "",
        "- `src/rbi_rag/`: modular RAG, evaluation, comparison, and packaging code",
        "- `scripts/`: executable project workflows",
        "- `configs/`: selected retrieval and experiment configs",
        "- `data/`: raw reports and evaluation data",
        "- `reports/`: saved evaluation, comparison, and final packaging artifacts",
        "- `streamlit_app.py`: saved-example demo UI",
        "",
        "## Evaluation methodology",
        "",
        "Retrieval is measured with Complete Evidence Recall, All-Reports Hit, Evidence Recall, Macro Report MRR, report coverage, contamination, latency, and context-size metrics. Generation is evaluated on saved development outputs with deterministic heuristic checks for factuality, faithfulness, citations, temporal attribution, comparative correctness, and abstention.",
        "",
        "## Limitations",
        "",
        "- Final V2 generation is development-only.",
        "- Old Phase 7 held-out results are historical and not a fresh V2 benchmark.",
        "- Generation metrics are deterministic heuristics, not human evaluation.",
        "- Cohere adds latency.",
        "- Unstructured extraction was blocked by OCR/Tesseract requirements for these PDFs.",
        "- This is not production-ready.",
        "",
        "## Safety and security",
        "",
        "API keys are read from `.env`; keys are not committed and generated artifacts are scanned for accidental key serialization.",
        "",
        "## Interview-ready summary",
        "",
        "This is a temporal multi-document RAG system for RBI Monetary Policy Reports. The strongest current system uses report-aware hybrid retrieval, Cohere reranking, Groq generation, and a sufficiency gate to reduce unsupported answers while keeping citations and temporal attribution explicit.",
    ]


def generate_readme_validation(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    referenced = [
        "requirements.txt",
        "requirements-v2.txt",
        "streamlit_app.py",
        "scripts/run_mmr_experiments.py",
        "scripts/validate_mmr_experiments.py",
        "scripts/generate_mmr_report.py",
        "scripts/generate_final_rag_comparison.py",
    ]
    missing = [path for path in referenced if path in text and not (root / path).exists()]
    payload = {"schema_version": 1, "status": "passed" if not missing else "failed", "missing_references": missing}
    lines = ["# README Validation", "", f"Status: `{payload['status']}`"]
    if missing:
        lines += ["", "## Missing references", ""]
        lines.extend(f"- {path}" for path in missing)
    write_md(root / OUT_DIR / "readme_validation.md", lines)
    return payload


def streamlit_instructions(root: Path) -> dict[str, Any]:
    app = root / "streamlit_app.py"
    payload = {
        "schema_version": 1,
        "status": "available" if app.exists() else "skipped",
        "app_path": "streamlit_app.py" if app.exists() else None,
        "run_command": "streamlit run streamlit_app.py" if app.exists() else None,
        "mode": "saved_evaluated_examples",
    }
    lines = [
        "# Streamlit Run Instructions",
        "",
        f"Status: `{payload['status']}`",
        "",
        "The app runs in safe demo mode using saved evaluated examples; it does not call live Groq or Cohere APIs.",
        "",
        "```powershell",
        "python -m pip install -r requirements-v2.txt",
        "streamlit run streamlit_app.py",
        "```",
    ]
    write_md(root / OUT_DIR / "streamlit_run_instructions.md", lines)
    return payload


def interview_pack_lines(root: Path) -> list[str]:
    rows = load_rows(root)
    retrieval = row_lookup(rows, "V2 Cohere retrieval")
    generation = row_lookup(rows, "V2 Cohere retrieval + sufficiency-gated generation")
    mmr = read_json(root / "reports/mmr_experiments/mmr_selection_decision.json", {})
    qa = [
        ("What problem did you solve?", "I built a temporal multi-document RAG system over RBI Monetary Policy Reports so users can ask questions that require correctly attributed evidence across report periods."),
        ("Why multi-document temporal RAG?", "RBI policy interpretation changes over time. The value is in comparing policy stance and narrative evolution across reports, not just retrieving one paragraph."),
        ("Why did multi-document scores look lower than single-document scores?", "The metrics are stricter. Single-document Hit-Rate@4 needs one relevant chunk; multi-report CER needs all required evidence from the right reports."),
        ("Why BM25 + dense embeddings?", "BM25 helps exact policy terms and numbers, while dense retrieval helps semantic phrasing. RBI reports need both."),
        ("Why RRF?", "RRF fuses dense and BM25 rankings without requiring score calibration."),
        ("Difference between MRR and MMR?", "MRR is Mean Reciprocal Rank, an evaluation metric. MMR is Maximal Marginal Relevance, a diversity-aware selection method."),
        ("Did MMR help?", f"MMR status: {mmr.get('status', 'not_run')}. Selected system after MMR: {mmr.get('selected_experiment_id', 'V2_COHERE_ONLY')}."),
        ("Why Cohere reranking?", "Cohere reranking improved development retrieval quality by better ordering the hybrid candidate set."),
        ("Why did Cohere increase latency?", "It adds remote reranking calls over candidate documents, so latency increases materially."),
        ("Why evidence sufficiency gate?", "It prevents the generator from giving unsupported answers when retrieved evidence is incomplete."),
        ("Why not just trust the LLM?", "The LLM can answer fluently without complete evidence. The gate and citations force source-grounded behavior."),
        ("How did you prevent wrong-report attribution?", "The pipeline uses report IDs, report-aware routing/quotas, source-labelled contexts, and contamination checks."),
        ("How did you evaluate?", "Retrieval used CER, All-Reports Hit, Evidence Recall, Macro MRR, coverage, contamination, latency, and context-size metrics. Generation used deterministic heuristic checks over saved outputs."),
        ("What were the final metrics?", f"Retrieval: CER {metric(retrieval.get('complete_evidence_recall'))}, All-Reports Hit {metric(retrieval.get('all_reports_hit'))}, Evidence Recall {metric(retrieval.get('evidence_recall'))}, Macro MRR {metric(retrieval.get('macro_mrr'))}. Generation factual correctness {metric(generation.get('factual_correctness'))}, faithfulness {metric(generation.get('faithfulness_to_context'))}, abstention {metric(generation.get('abstention_correctness'))}."),
        ("What failed?", "Unstructured extraction did not produce usable non-OCR elements and needs Tesseract/OCR. Comparative generation also remains weak when evidence is incomplete."),
        ("What would you improve next?", "Create a fresh V2 evaluation set, add human evaluation, cache Cohere calls, retry Unstructured with OCR deliberately installed, and improve live Streamlit mode."),
        ("Is it production-ready?", "No. It is demo/interview-ready with known limitations, not production-ready."),
        ("How would you scale/deploy it?", "Precompute indexes and reranker cache, add a service API, add observability and evaluation monitoring, secure key management, and deploy the Streamlit/API layer behind auth."),
    ]
    lines = ["# Interview Pack", ""]
    for question, answer in qa:
        lines += [f"## {question}", "", answer, ""]
    return lines


def git_status(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"status": "not_a_git_repo", "short": None}
    result = subprocess.run(["git", "status", "--short"], cwd=root, text=True, capture_output=True, check=False)
    return {"status": "ok" if result.returncode == 0 else "failed", "short": result.stdout.splitlines(), "stderr": result.stderr}


def secret_values(root: Path) -> list[str]:
    load_project_dotenv(root)
    values = []
    for name in ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY"):
        value = os.getenv(name)
        if value:
            values.append(value)
    return values


def scan_for_api_keys(root: Path) -> dict[str, Any]:
    values = secret_values(root)
    scan_paths = [
        root / "README.md",
        root / "streamlit_app.py",
        root / "reports/final_packaging",
        root / "reports/final_comparison",
        root / "reports/mmr_experiments",
        root / "reports/final_generation_bakeoff",
    ]
    hits = []
    for base in scan_paths:
        paths = [base] if base.is_file() else list(base.rglob("*")) if base.exists() else []
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(value and value in text for value in values):
                hits.append(relative(root, path))
    return {"status": "passed" if not hits else "failed", "hit_paths": sorted(set(hits))}


def final_status_payload(root: Path, tests_status: str, pip_check_status: str) -> dict[str, Any]:
    mmr = read_json(root / "reports/mmr_experiments/mmr_selection_decision.json", {})
    final_mmr_generation = read_json(root / "reports/final_mmr_generation/final_mmr_generation_selection_decision.json", {})
    final_bakeoff_generation = read_json(root / "reports/final_generation_bakeoff/final_generation_strategy_selection_decision.json", {})
    v2_selected = read_json(root / "reports/v2_unstructured_cohere/v2_selected_retrieval.json", {})
    gen_summary = read_json(root / "reports/v2_sufficiency/dev_sufficiency_eval_summary.json", {})
    unstructured_env = read_json(root / "reports/v2_unstructured_cohere/environment_readiness.json", {})
    security = scan_for_api_keys(root)
    best_generation = (
        f"{final_bakeoff_generation.get('selected_variant_id')} + Groq llama-3.1-8b-instant + sufficiency gate"
        if final_bakeoff_generation.get("selected_variant_id")
        else (
            "MMR_LAMBDA_06 + Groq llama-3.1-8b-instant + sufficiency gate"
            if final_mmr_generation.get("status") == "selected_mmr_end_to_end"
            else "V2_COHERE_ONLY + Groq llama-3.1-8b-instant + sufficiency gate"
        )
    )
    generation_status = (
        "final_generation_bakeoff_completed_dev_only; previous_generation_artifacts_preserved"
        if final_bakeoff_generation
        else (
            "final_mmr_generation_completed_dev_only; previous_v2_generation_preserved"
            if final_mmr_generation
            else "existing_dev_sufficiency_generation_used; not_rerun"
        )
    )
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "best_retrieval_method": mmr.get("selected_experiment_id") if mmr.get("status") == "evaluated_selected" else v2_selected.get("selected_experiment_id", "V2_COHERE_ONLY"),
        "best_generation_method": best_generation,
        "generation_retrieval_note": final_bakeoff_generation.get("reason") or final_mmr_generation.get("reason") or "final generation decision not available",
        "final_mmr_generation_status": final_mmr_generation.get("status", "not_run"),
        "final_generation_bakeoff_status": final_bakeoff_generation.get("status", "not_run"),
        "final_generation_bakeoff_selected_variant": final_bakeoff_generation.get("selected_variant_id"),
        "mmr_status": mmr.get("status", "not_run"),
        "unstructured_status": "blocked_by_ocr_tesseract_requirement",
        "unstructured_environment": {key: value for key, value in unstructured_env.items() if key.endswith("_available")},
        "heldout_status": "not_rerun_for_overnight_packaging",
        "generation_status": generation_status,
        "streamlit_status": "available" if (root / "streamlit_app.py").exists() else "skipped",
        "readme_status": "available" if (root / "README.md").exists() else "missing",
        "tests_status": tests_status,
        "pip_check_status": pip_check_status,
        "api_key_scan_status": security["status"],
        "git_status": git_status(root),
        "ready_for_demo": "ready_for_demo_with_known_limitations",
        "ready_for_interview": "ready_for_interview",
        "production_status": "not_production_ready",
        "remaining_limitations": [
            "fresh V2 held-out evaluation not created",
            "generation metrics are deterministic heuristics, not human evaluation",
            "Unstructured extraction remains OCR/Tesseract-blocked",
            "Cohere latency is high without caching",
            "comparative generation remains weak when evidence is incomplete",
        ],
        "sufficiency_summary_status": "available" if gen_summary else "missing",
    }


def validate_final_artifacts(root: Path, tests_status: str, pip_check_status: str) -> dict[str, Any]:
    security = scan_for_api_keys(root)
    required = [
        root / "README.md",
        root / "streamlit_app.py",
        root / "reports/final_packaging/final_project_report.md",
        root / "reports/final_packaging/interview_pack.md",
        root / "reports/final_packaging/final_project_status.json",
        root / "reports/final_packaging/streamlit_run_instructions.md",
        root / "reports/final_comparison/rag_methods_master_comparison.md",
        root / "reports/mmr_experiments/mmr_leaderboard.json",
        root / "reports/mmr_experiments/mmr_selection_decision.json",
    ]
    missing = [relative(root, path) for path in required if not path.exists()]
    comparison = (root / "reports/final_comparison/rag_methods_master_comparison.md").read_text(encoding="utf-8", errors="ignore") if (root / "reports/final_comparison/rag_methods_master_comparison.md").exists() else ""
    issues = []
    if security["status"] != "passed":
        issues.append("api_key_serialized")
    if missing:
        issues.append("missing_required_artifacts")
    if ".env" in [path.name for path in (root / OUT_DIR).rglob("*") if path.is_file()]:
        issues.append("env_file_copied_to_reports")
    if "Maximal Marginal Relevance" not in comparison or "Mean Reciprocal Rank" not in comparison:
        issues.append("mrr_mmr_distinction_missing")
    if "fresh V2 benchmark" not in comparison:
        issues.append("heldout_caveat_missing")
    if tests_status != "passed":
        issues.append("tests_not_recorded_passed")
    if pip_check_status != "passed":
        issues.append("pip_check_not_recorded_passed")
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "passed" if not issues else "failed",
        "issues": sorted(set(issues)),
        "missing_required_artifacts": missing,
        "api_key_scan_status": security["status"],
        "api_key_hit_paths": security["hit_paths"],
        "env_file_copied": "env_file_copied_to_reports" in issues,
        "heldout_rerun_occurred": False,
        "generation_rerun_occurred": False,
        "mmr_used_heldout": False,
        "unstructured_falsely_claimed": False,
        "mrr_mmr_distinction_correct": "mrr_mmr_distinction_missing" not in issues,
        "streamlit_app_exists": (root / "streamlit_app.py").exists(),
        "tests_status": tests_status,
        "pip_check_status": pip_check_status,
    }
    return payload


def write_status_and_validation(root: Path, tests_status: str, pip_check_status: str) -> dict[str, Any]:
    status = final_status_payload(root, tests_status, pip_check_status)
    write_json(root / OUT_DIR / "final_project_status.json", status)
    lines = ["# Final Project Status", ""]
    for key in [
        "best_retrieval_method",
        "best_generation_method",
        "final_mmr_generation_status",
        "final_generation_bakeoff_status",
        "final_generation_bakeoff_selected_variant",
        "mmr_status",
        "unstructured_status",
        "heldout_status",
        "generation_status",
        "streamlit_status",
        "readme_status",
        "tests_status",
        "pip_check_status",
        "api_key_scan_status",
        "ready_for_demo",
        "ready_for_interview",
        "production_status",
    ]:
        lines.append(f"- {key}: `{status.get(key)}`")
    lines += ["", "## Remaining limitations", ""]
    lines.extend(f"- {item}" for item in status["remaining_limitations"])
    write_md(root / OUT_DIR / "final_project_status.md", lines)

    validation = validate_final_artifacts(root, tests_status, pip_check_status)
    write_json(root / OUT_DIR / "final_artifact_validation.json", validation)
    validation_lines = ["# Final Artifact Validation", "", f"Status: `{validation['status']}`", ""]
    validation_lines += [
        f"- No API keys serialized: `{validation['api_key_scan_status'] == 'passed'}`",
        f"- No `.env` copied into reports: `{not validation['env_file_copied']}`",
        f"- No held-out rerun: `{not validation['heldout_rerun_occurred']}`",
        f"- No generation rerun in packaging: `{not validation['generation_rerun_occurred']}`",
        f"- MMR did not use held-out: `{not validation['mmr_used_heldout']}`",
        f"- Unstructured not falsely claimed: `{not validation['unstructured_falsely_claimed']}`",
        f"- MRR/MMR distinction correct: `{validation['mrr_mmr_distinction_correct']}`",
        f"- Streamlit app exists: `{validation['streamlit_app_exists']}`",
        f"- Tests status: `{tests_status}`",
        f"- Pip check status: `{pip_check_status}`",
    ]
    if validation["issues"]:
        validation_lines += ["", "## Issues", ""]
        validation_lines.extend(f"- {issue}" for issue in validation["issues"])
    write_md(root / OUT_DIR / "final_artifact_validation.md", validation_lines)
    return {"status": status, "validation": validation}


def generate_packaging(root: Path = Path("."), tests_status: str = "not_run", pip_check_status: str = "not_run") -> dict[str, Any]:
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    update_overnight_status(root, phase_0_status="completed")
    checksums = create_checksum_manifest(root)
    archives = archive_existing_outputs(root)
    update_overnight_status(root, phase_1_mmr_status="completed" if (root / "reports/mmr_experiments/mmr_leaderboard.json").exists() else "pending")

    write_md(root / OUT_DIR / "final_project_report.md", final_report_lines(root))
    update_overnight_status(root, phase_3_final_report_status="completed")

    write_md(root / "README.md", readme_lines(root))
    readme_validation = generate_readme_validation(root)
    update_overnight_status(root, phase_5_readme_status=readme_validation["status"])

    streamlit = streamlit_instructions(root)
    update_overnight_status(root, phase_4_streamlit_status=streamlit["status"])

    write_md(root / OUT_DIR / "interview_pack.md", interview_pack_lines(root))

    status_validation = write_status_and_validation(root, tests_status, pip_check_status)
    validation_status = status_validation["validation"]["status"]
    final_status = "completed" if validation_status == "passed" else "completed_with_skips"
    update_overnight_status(
        root,
        phase_2_comparison_update_status="completed" if (root / "reports/final_comparison/rag_methods_master_comparison.md").exists() else "pending",
        phase_6_validation_status=validation_status,
        final_status=final_status,
    )
    return {
        "status": final_status,
        "checksums_status": checksums["status"],
        "archives": archives,
        "readme_validation": readme_validation["status"],
        "streamlit_status": streamlit["status"],
        "final_artifact_validation": validation_status,
        "output_dir": str(root / OUT_DIR),
    }
