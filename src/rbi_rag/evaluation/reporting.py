from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

from ..utils import git_commit_hash


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_retrieval_outputs(payload: dict, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_json = output_directory / "retrieval_raw_results.json"
    atomic_write_json(raw_json, payload)  # raw evidence first
    rows = [detail for strategy in payload["strategies"] for detail in strategy["details"]]
    with (output_directory / "retrieval_question_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    summaries = [{key: value for key, value in strategy.items() if key != "details"} for strategy in payload["strategies"]]
    with (output_directory / "retrieval_pipeline_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    summary_payload = {key: value for key, value in payload.items() if key != "strategies"}
    summary_payload["strategies"] = summaries
    atomic_write_json(output_directory / "retrieval_summary.json", summary_payload)


def generate_markdown_report(retrieval: dict, generation: dict | None, config_path: Path, output: Path):
    config = retrieval["config"]
    lines = [
        "# RBI April 2025 RAG Baseline Report", "",
        f"- Run timestamp: {retrieval['created_at_utc']}",
        f"- Repository commit: {git_commit_hash() or 'not available (not a Git repository)'}",
        f"- Configuration: `{config_path.as_posix()}`", f"- Dataset version: `{config['dataset_version']}`",
        f"- PDF checksum: `{retrieval['pdf_sha256']}`", f"- Embedding model: `{config['embedding_model']}`",
        f"- Reranker: `{config['reranker_model']}`", f"- Generator: `{config['generator_model']}`",
        f"- Judge: `{config['judge_model']}`", "",
        "## Retrieval comparison", "", "| Pipeline | Hit-Rate@4 | MRR | Mean latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for row in retrieval["strategies"]:
        lines.append(f"| {row['strategy']} | {row['hit_rate_at_k']:.2%} | {row['mrr']:.3f} | {row['mean_latency_ms']:.1f} |")
    lines += ["", "Reranking cannot recover a relevant chunk absent from its initial candidate pool. It can promote a relevant chunk from ranks 5–15 into the final top four, so it may improve both MRR and Hit-Rate@4.", "", "## Generation evaluation", ""]
    if generation is None:
        lines.append("Not executed or not yet validated. No generation metric is reported without coverage.")
    else:
        lines += ["| Metric | Mean | Median | Std. dev. | Successful | Failed | Coverage |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, value in generation["summary"].items():
            fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
            lines.append(f"| {name} | {fmt(value['mean'])} | {fmt(value['median'])} | {fmt(value['standard_deviation'])} | {value['successful_cases']} | {value['failed_cases']} | {value['coverage_percentage']:.2f}% |")
    lines += ["", "## Result status", "", "The retrieval table is validated from the current saved raw run. Generation remains unavailable until a credentialed evaluation completes with explicitly reported coverage. Metrics in archived artifacts are historical and are not carried into this report.", "", "## Settings", "", f"- Chunking: {config['chunk_size']} characters, {config['chunk_overlap']} overlap", f"- Candidate pools: dense={config['dense_k']}, BM25={config['bm25_k']}, reranker={config['reranker_candidate_k']}", f"- RRF constant: {config['fusion_rrf_k']}; final context: top {config['final_k']}", "", "## Known limitations", "", "- The 30 questions are a development set, not a held-out test set.", "- Relevance is judged at page level, not chunk level.", "- Hosted LLM generation is not guaranteed byte-identical even at temperature zero.", "- Archived PDF and preliminary metrics are not validated current results.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
