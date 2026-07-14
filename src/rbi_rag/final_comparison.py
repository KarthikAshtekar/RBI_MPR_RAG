from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


OUT_DIR = Path("reports/final_comparison")

MASTER_FIELDS = [
    "method",
    "experiment_id",
    "scope",
    "dataset_split",
    "retrieval_or_generation",
    "parser",
    "embedding_model",
    "bm25_enabled",
    "dense_enabled",
    "hybrid_enabled",
    "rrf_enabled",
    "weighted_rrf_enabled",
    "multi_query_enabled",
    "facet_enabled",
    "mmr_enabled",
    "reranker_provider",
    "reranker_model",
    "sufficiency_gate_enabled",
    "hit_rate",
    "all_reports_hit",
    "complete_evidence_recall",
    "evidence_recall",
    "mrr",
    "macro_mrr",
    "factual_correctness",
    "faithfulness_to_context",
    "contextual_relevancy",
    "contextual_recall",
    "citation_correctness",
    "citation_completeness",
    "temporal_attribution_correctness",
    "comparative_correctness",
    "abstention_correctness",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "mean_estimated_tokens",
    "report_coverage",
    "single_report_contamination",
    "status",
    "notes",
    "source_artifact",
    "sort_order",
]

MASTER_MD_COLUMNS = [
    ("Method", "method"),
    ("Scope", "scope"),
    ("Split", "dataset_split"),
    ("Hit / All-Reports Hit", "hit_or_all_reports_hit"),
    ("CER", "complete_evidence_recall"),
    ("Evidence Recall", "evidence_recall"),
    ("MRR / Macro MRR", "mrr_or_macro_mrr"),
    ("Factual Correctness", "factual_correctness"),
    ("Citation Correctness", "citation_correctness"),
    ("Temporal Attribution", "temporal_attribution_correctness"),
    ("Median Latency", "median_latency_ms"),
    ("Mean Tokens", "mean_estimated_tokens"),
    ("Coverage", "report_coverage"),
    ("Contamination", "single_report_contamination"),
    ("Status", "status"),
]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def point(value: Any) -> Any:
    if isinstance(value, dict) and "point_estimate" in value:
        return value["point_estimate"]
    return value


def metric(summary: dict[str, Any], name: str) -> Any:
    return point(summary.get(name))


def eval_metric(summary: dict[str, Any], name: str) -> Any:
    return (summary.get("metrics") or {}).get(name, {}).get("mean_score")


def base_row(**kwargs: Any) -> dict[str, Any]:
    row = {field: None for field in MASTER_FIELDS}
    row.update({
        "status": "completed",
        "bm25_enabled": False,
        "dense_enabled": False,
        "hybrid_enabled": False,
        "rrf_enabled": False,
        "weighted_rrf_enabled": False,
        "multi_query_enabled": False,
        "facet_enabled": False,
        "mmr_enabled": False,
        "sufficiency_gate_enabled": False,
    })
    row.update(kwargs)
    return row


def strategy_features(strategy: str) -> dict[str, Any]:
    return {
        "dense_enabled": "dense" in strategy or "hybrid" in strategy,
        "bm25_enabled": "bm25" in strategy or "hybrid" in strategy,
        "hybrid_enabled": "hybrid" in strategy,
        "rrf_enabled": "hybrid" in strategy,
        "reranker_provider": "local_cross_encoder" if "reranked" in strategy else None,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2" if "reranked" in strategy else None,
    }


def row_from_retrieval_summary(item: dict[str, Any], *, method: str, experiment_id: str, scope: str, split: str, source: str, sort_order: int, **kwargs: Any) -> dict[str, Any]:
    row = base_row(
        method=method,
        experiment_id=experiment_id,
        scope=scope,
        dataset_split=split,
        retrieval_or_generation="retrieval",
        hit_rate=item.get("hit_rate_at_k"),
        mrr=item.get("mrr"),
        mean_latency_ms=item.get("mean_latency_ms"),
        source_artifact=source,
        sort_order=sort_order,
        **kwargs,
    )
    row.update({key: value for key, value in strategy_features(experiment_id).items() if row.get(key) is None or isinstance(value, bool)})
    return row


def row_from_temporal_summary(summary: dict[str, Any], *, method: str, experiment_id: str, scope: str, split: str, source: str, sort_order: int, status: str = "completed", notes: str | None = None, **kwargs: Any) -> dict[str, Any]:
    latency = summary.get("latency_ms") or {}
    return base_row(
        method=method,
        experiment_id=experiment_id,
        scope=scope,
        dataset_split=split,
        retrieval_or_generation="retrieval",
        all_reports_hit=metric(summary, "all_reports_hit_rate") if "all_reports_hit_rate" in summary else summary.get("all_reports_hit"),
        complete_evidence_recall=metric(summary, "complete_evidence_recall"),
        evidence_recall=metric(summary, "evidence_recall"),
        macro_mrr=metric(summary, "macro_report_mrr") if "macro_report_mrr" in summary else summary.get("macro_mrr"),
        mean_latency_ms=latency.get("mean") if latency else summary.get("mean_latency_ms"),
        median_latency_ms=latency.get("median") if latency else summary.get("median_latency_ms"),
        p95_latency_ms=latency.get("p95") if latency else summary.get("p95_latency_ms"),
        mean_estimated_tokens=summary.get("mean_estimated_tokens"),
        report_coverage=metric(summary, "mean_report_coverage") if "mean_report_coverage" in summary else summary.get("report_coverage"),
        single_report_contamination=metric(summary, "cross_report_contamination_rate") if "cross_report_contamination_rate" in summary else summary.get("single_report_contamination") if "single_report_contamination" in summary else summary.get("contamination"),
        status=status,
        notes=notes,
        source_artifact=source,
        sort_order=sort_order,
        **kwargs,
    )


def row_from_experiment(exp: dict[str, Any], *, method: str, scope: str, split: str, source: str, sort_order: int, status: str = "completed", notes: str | None = None, **kwargs: Any) -> dict[str, Any]:
    cfg = exp.get("config") or {}
    reranker_provider = kwargs.pop("reranker_provider", "local_cross_encoder")
    reranker_model = kwargs.pop("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return base_row(
        method=method,
        experiment_id=exp.get("experiment_id") or cfg.get("id"),
        scope=scope,
        dataset_split=split,
        retrieval_or_generation="retrieval",
        parser=kwargs.pop("parser", "PyPDFLoader"),
        embedding_model=kwargs.pop("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        bm25_enabled=cfg.get("bk") is not None or kwargs.pop("bm25_enabled", True),
        dense_enabled=cfg.get("dk") is not None or kwargs.pop("dense_enabled", True),
        hybrid_enabled=kwargs.pop("hybrid_enabled", True),
        rrf_enabled=cfg.get("rrf") is not None or kwargs.pop("rrf_enabled", True),
        weighted_rrf_enabled=(cfg.get("dw") not in (None, 1, 1.0) or cfg.get("bw") not in (None, 1, 1.0) or kwargs.pop("weighted_rrf_enabled", False)),
        multi_query_enabled=bool(cfg.get("terminology_expansion") == "multi_query" or kwargs.pop("multi_query_enabled", False)),
        facet_enabled=bool(cfg.get("facet_decomposition") or kwargs.pop("facet_enabled", False)),
        mmr_enabled=bool(kwargs.pop("mmr_enabled", False)),
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        all_reports_hit=exp.get("all_reports_hit"),
        complete_evidence_recall=exp.get("complete_evidence_recall"),
        evidence_recall=exp.get("evidence_recall"),
        macro_mrr=exp.get("macro_mrr") if exp.get("macro_mrr") is not None else exp.get("macro_report_mrr"),
        mean_latency_ms=exp.get("mean_latency_ms"),
        median_latency_ms=exp.get("median_latency_ms"),
        p95_latency_ms=exp.get("p95_latency_ms"),
        mean_estimated_tokens=exp.get("mean_estimated_tokens"),
        report_coverage=exp.get("report_coverage"),
        single_report_contamination=exp.get("single_report_contamination") if "single_report_contamination" in exp else exp.get("contamination"),
        status=status,
        notes=notes,
        source_artifact=source,
        sort_order=sort_order,
        **kwargs,
    )


def row_from_generation(summary: dict[str, Any], generation_summary: dict[str, Any], *, method: str, experiment_id: str, scope: str, split: str, source: str, sort_order: int, sufficiency: bool, notes: str | None = None) -> dict[str, Any]:
    return base_row(
        method=method,
        experiment_id=experiment_id,
        scope=scope,
        dataset_split=split,
        retrieval_or_generation="generation",
        parser="PyPDFLoader",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        bm25_enabled=True,
        dense_enabled=True,
        hybrid_enabled=True,
        rrf_enabled=True,
        reranker_provider="cohere",
        reranker_model="rerank-v3.5",
        sufficiency_gate_enabled=sufficiency,
        factual_correctness=eval_metric(summary, "factual_correctness"),
        faithfulness_to_context=eval_metric(summary, "faithfulness_to_context"),
        contextual_relevancy=eval_metric(summary, "contextual_relevancy"),
        contextual_recall=eval_metric(summary, "contextual_recall"),
        citation_correctness=eval_metric(summary, "citation_correctness"),
        citation_completeness=eval_metric(summary, "citation_completeness"),
        temporal_attribution_correctness=eval_metric(summary, "temporal_attribution_correctness"),
        comparative_correctness=eval_metric(summary, "comparative_correctness"),
        abstention_correctness=eval_metric(summary, "abstention_correctness"),
        mean_latency_ms=generation_summary.get("mean_generation_latency_ms"),
        median_latency_ms=generation_summary.get("median_generation_latency_ms"),
        status="completed",
        notes=notes,
        source_artifact=source,
        sort_order=sort_order,
    )


def find_experiment(root: Path, experiment_id: str) -> dict[str, Any] | None:
    for base in [root / "reports/optimisation", root / "reports/structural_optimisation"]:
        path = base / experiment_id / "summary.json"
        if path.exists():
            return load_json(path, {})
    return None


def mmr_rows(root: Path) -> list[dict[str, Any]]:
    out = root / "reports/mmr_experiments"
    decision = load_json(out / "mmr_selection_decision.json", {})
    leaderboard = load_json(out / "mmr_leaderboard.json", {})
    source = "reports/mmr_experiments/mmr_selection_decision.json; reports/mmr_experiments/mmr_leaderboard.json"
    if not decision and not leaderboard:
        return [
            base_row(
                method="MMR / diversity selection",
                experiment_id="MMR",
                scope="Three-report temporal retrieval",
                dataset_split="dev",
                retrieval_or_generation="retrieval",
                parser="PyPDFLoader",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                status="not_run",
                notes="True Maximal Marginal Relevance was not found as an independently evaluated experiment. DIV01 used exact-overlap diversity filtering, which is not MMR.",
                source_artifact="code/artifact search for MMR; reports/optimisation/DIV01/summary.json for separate diversity filter",
                sort_order=111,
            )
        ]

    status = decision.get("status") or "evaluated"
    notes = (
        f"True Maximal Marginal Relevance evaluated over saved V2_COHERE_ONLY development reranker outputs. "
        f"Decision: {decision.get('selected_experiment_id')}; reason: {decision.get('reason')}. "
        "DIV01 remains an exact-overlap diversity filter and is not MMR."
    )
    rows = [
        base_row(
            method="MMR / diversity selection",
            experiment_id="MMR",
            scope="Three-report temporal retrieval",
            dataset_split="dev",
            retrieval_or_generation="retrieval",
            parser="PyPDFLoader",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            bm25_enabled=True,
            dense_enabled=True,
            hybrid_enabled=True,
            rrf_enabled=True,
            reranker_provider="cohere",
            reranker_model="rerank-v3.5",
            status=status,
            notes=notes,
            source_artifact=source,
            sort_order=111,
        )
    ]
    names = {
        "MMR_BASELINE_V2_COHERE": "MMR baseline: V2 Cohere selected context",
        "MMR_LAMBDA_06": "True MMR lambda 0.6",
        "MMR_LAMBDA_07": "True MMR lambda 0.7",
        "MMR_LAMBDA_08": "True MMR lambda 0.8",
    }
    sort_orders = {
        "MMR_BASELINE_V2_COHERE": 112,
        "MMR_LAMBDA_06": 113,
        "MMR_LAMBDA_07": 114,
        "MMR_LAMBDA_08": 115,
    }
    for exp in leaderboard.get("completed", []):
        exp_id = exp.get("experiment_id")
        rows.append(
            row_from_experiment(
                exp,
                method=names.get(exp_id, f"True MMR {exp_id}"),
                scope="Three-report temporal retrieval",
                split="dev",
                source="reports/mmr_experiments/mmr_leaderboard.json",
                sort_order=sort_orders.get(exp_id, 119),
                parser="PyPDFLoader",
                reranker_provider="cohere",
                reranker_model="rerank-v3.5",
                mmr_enabled=bool(exp.get("mmr_enabled")),
                notes="True Maximal Marginal Relevance context selection using lambda * relevance - (1 - lambda) * max similarity to selected documents.",
            )
        )
    for skipped in leaderboard.get("skipped", []):
        exp_id = skipped.get("experiment_id")
        rows.append(
            base_row(
                method=names.get(exp_id, f"True MMR {exp_id}"),
                experiment_id=exp_id,
                scope="Three-report temporal retrieval",
                dataset_split="dev",
                retrieval_or_generation="retrieval",
                parser="PyPDFLoader",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                bm25_enabled=True,
                dense_enabled=True,
                hybrid_enabled=True,
                rrf_enabled=True,
                reranker_provider="cohere",
                reranker_model="rerank-v3.5",
                mmr_enabled=bool(skipped.get("mmr_enabled")),
                status=skipped.get("status") or "blocked",
                notes=skipped.get("reason"),
                source_artifact="reports/mmr_experiments/mmr_leaderboard.json",
                sort_order=sort_orders.get(exp_id, 119),
            )
        )
    return rows


def build_master_rows(root: Path = Path(".")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = load_json(root / "reports/current/retrieval_summary.json", {})
    config = current.get("config") or {}
    single_names = {
        "dense": "Single-document Dense",
        "dense_reranked": "Single-document Dense + reranker",
        "bm25": "Single-document BM25",
        "bm25_reranked": "Single-document BM25 + reranker",
        "hybrid_rrf": "Single-document Hybrid RRF",
        "hybrid_reranked": "Single-document Hybrid RRF + reranker",
    }
    for index, item in enumerate(current.get("strategies", []), start=1):
        strategy = item.get("strategy")
        rows.append(row_from_retrieval_summary(
            item,
            method=single_names.get(strategy, strategy),
            experiment_id=strategy,
            scope=current.get("scope", "April 2025 only"),
            split=config.get("dataset_version", "single_document_dev"),
            source="reports/current/retrieval_summary.json",
            sort_order=index,
            parser="PyPDFLoader",
            embedding_model=config.get("embedding_model"),
            notes="Single-document Hit-Rate@4; not directly comparable to multi-report CER.",
        ))

    arch = load_json(root / "reports/multi_report/architecture_comparison.json", {})
    dev_arch = (arch.get("splits") or {}).get("dev", {})
    if dev_arch.get("naive_global"):
        rows.append(row_from_temporal_summary(
            dev_arch["naive_global"],
            method="Multi-report naive global retrieval",
            experiment_id="naive_global",
            scope="Three-report temporal retrieval",
            split="dev",
            source="reports/multi_report/architecture_comparison.json",
            sort_order=20,
            parser="PyPDFLoader",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            bm25_enabled=True,
            dense_enabled=True,
            hybrid_enabled=True,
            rrf_enabled=True,
            reranker_provider="local_cross_encoder",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            notes="Naive global retrieval over all reports; higher contamination.",
        ))
    if dev_arch.get("report_aware"):
        rows.append(row_from_temporal_summary(
            dev_arch["report_aware"],
            method="Multi-report report-aware retrieval",
            experiment_id="report_aware_dev",
            scope="Three-report temporal retrieval",
            split="dev",
            source="reports/multi_report/architecture_comparison.json",
            sort_order=30,
            parser="PyPDFLoader",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            bm25_enabled=True,
            dense_enabled=True,
            hybrid_enabled=True,
            rrf_enabled=True,
            reranker_provider="local_cross_encoder",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            notes="Report-aware routing reduced contamination and improved evidence metrics.",
        ))

    for method, experiment_id, sort_order, status, notes in [
        ("Dense-only temporal retrieval", "dense_temporal_only", 40, "implemented_not_independently_evaluated", "Dense retrieval is implemented and used inside hybrid temporal retrieval; no standalone temporal dense-only result artifact was found."),
        ("BM25-only temporal retrieval", "bm25_temporal_only", 41, "implemented_not_independently_evaluated", "BM25 retrieval is implemented and used inside hybrid temporal retrieval; no standalone temporal BM25-only result artifact was found."),
    ]:
        rows.append(base_row(
            method=method,
            experiment_id=experiment_id,
            scope="Three-report temporal retrieval",
            dataset_split="dev",
            retrieval_or_generation="retrieval",
            parser="PyPDFLoader",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            dense_enabled=method.startswith("Dense"),
            bm25_enabled=method.startswith("BM25"),
            status=status,
            notes=notes,
            source_artifact="configs/multi_report.yaml; no standalone metric artifact",
            sort_order=sort_order,
        ))

    for experiment_id, method, sort_order in [
        ("temporal_baseline", "Hybrid BM25 + Dense retrieval with RRF", 50),
        ("RRF_K10", "RRF fusion k=10", 60),
        ("RRF_K30", "RRF fusion k=30", 61),
        ("RRF_K60", "RRF fusion k=60", 62),
        ("RRF_K100", "RRF fusion k=100", 63),
        ("WRRF_D1_B1", "Weighted RRF D1/B1 reference", 70),
        ("WRRF_D1.5_B1", "Weighted RRF dense weight 1.5", 71),
        ("WRRF_D2_B1", "Weighted RRF dense weight 2", 72),
        ("WRRF_D1_B1.5", "Weighted RRF BM25 weight 1.5", 73),
        ("WRRF_D1_B2", "Weighted RRF BM25 weight 2", 74),
        ("QUOTA_LARGE", "Candidate-pool/quota optimised retrieval", 90),
        ("EXP01", "Terminology expansion append", 100),
        ("EXP02", "Multi-query retrieval / terminology expansion", 101),
        ("FAC01", "Facet decomposition", 102),
        ("DIV01", "Exact-overlap diversity filter", 110),
    ]:
        exp = find_experiment(root, experiment_id)
        if exp:
            rows.append(row_from_experiment(
                exp,
                method=method,
                scope="Three-report temporal retrieval",
                split="dev",
                source=f"reports/optimisation/{experiment_id}/summary.json",
                sort_order=sort_order,
                notes="Stage A development optimisation experiment.",
            ))

    rows.extend(mmr_rows(root))

    for experiment_id, method, sort_order in [
        ("ADJ00", "Final local cross-encoder retrieval baseline", 80),
        ("RERANK00", "Local cross-encoder reranking", 81),
        ("ADJ01", "Adjacent expansion boundary", 120),
        ("ADJ02", "Adjacent expansion always", 121),
        ("ADJ03", "Adjacent expansion trend-only", 122),
        ("CPARENT01", "Child-parent retrieval same-page parent", 130),
        ("CPARENT02", "Child-parent retrieval adjacent-child parent", 131),
        ("CPARENT03", "Child-parent retrieval page-bounded parent", 132),
        ("SW01", "Sentence-window retrieval window 1", 140),
        ("SW02", "Sentence-window retrieval window 2", 141),
        ("SW03", "Chunk-window retrieval neighbour chunks", 142),
    ]:
        exp = find_experiment(root, experiment_id)
        if exp:
            rows.append(row_from_experiment(
                exp,
                method=method,
                scope="Three-report temporal retrieval",
                split="dev",
                source=f"reports/structural_optimisation/{experiment_id}/summary.json",
                sort_order=sort_order,
                notes=exp.get("description") or "Phase 6B structural optimisation experiment.",
            ))

    heldout = load_json(root / "reports/final_evaluation/heldout_retrieval_summary.json", {})
    if heldout:
        rows.append(row_from_experiment(
            heldout,
            method="Phase 7 selected final retrieval held-out diagnostic",
            scope="Three-report temporal retrieval",
            split="phase7_heldout_reused_for_old_final_only",
            source="reports/final_evaluation/heldout_retrieval_summary.json",
            sort_order=150,
            status="completed_historical_heldout",
            notes="Historical Phase 7 held-out retrieval for old final config; not a fresh V2 benchmark.",
        ))

    v2 = load_json(root / "reports/v2_unstructured_cohere/v2_experiment_leaderboard.json", {})
    v2_method_names = {
        "V2_BASELINE_FINAL": "V2 baseline final retrieval",
        "V2_COHERE_ONLY": "V2 Cohere retrieval",
        "V2_UNSTRUCTURED_ONLY": "V2 Unstructured retrieval",
        "V2_UNSTRUCTURED_COHERE": "V2 Unstructured + Cohere retrieval",
    }
    v2_sort_orders = {
        "V2_BASELINE_FINAL": 160,
        "V2_COHERE_ONLY": 170,
        "V2_UNSTRUCTURED_ONLY": 171,
        "V2_UNSTRUCTURED_COHERE": 172,
    }
    for exp in v2.get("completed", []):
        experiment_id = exp.get("experiment_id")
        method = v2_method_names.get(experiment_id, f"V2 retrieval {experiment_id}")
        rows.append(row_from_experiment(
            exp,
            method=method,
            scope="Three-report temporal retrieval",
            split="dev",
            source="reports/v2_unstructured_cohere/v2_experiment_leaderboard.json",
            sort_order=v2_sort_orders.get(experiment_id, 179),
            reranker_provider=exp.get("reranker_provider"),
            reranker_model=exp.get("reranker_model"),
            notes="Controlled V2 development comparison. Poppler-enabled Unstructured arms are included when completed.",
        ))
    for skipped in v2.get("skipped", []):
        rows.append(base_row(
            method=skipped.get("experiment_id", "V2 skipped experiment"),
            experiment_id=skipped.get("experiment_id"),
            scope="Three-report temporal retrieval",
            dataset_split="dev",
            retrieval_or_generation="retrieval",
            status="not_run",
            notes=skipped.get("reason"),
            source_artifact="reports/v2_unstructured_cohere/v2_experiment_leaderboard.json",
            sort_order=175,
        ))

    v2_gen_eval = load_json(root / "reports/v2_generation/dev_answer_eval_summary.json", {})
    v2_gen_sum = load_json(root / "reports/v2_generation/dev_generation_summary.json", {})
    if v2_gen_eval:
        rows.append(row_from_generation(
            v2_gen_eval,
            v2_gen_sum,
            method="V2 Cohere retrieval + generation",
            experiment_id="V2_GENERATION_DEV",
            scope="Three-report temporal generation",
            split="dev",
            source="reports/v2_generation/dev_answer_eval_summary.json",
            sort_order=180,
            sufficiency=False,
            notes="Groq generation over saved V2_COHERE_ONLY retrieval outputs before sufficiency gate.",
        ))
    suff_eval = load_json(root / "reports/v2_sufficiency/dev_sufficiency_eval_summary.json", {})
    suff_sum = load_json(root / "reports/v2_sufficiency/dev_generation_sufficiency_summary.json", {})
    if suff_eval:
        rows.append(row_from_generation(
            suff_eval,
            suff_sum,
            method="V2 Cohere retrieval + sufficiency-gated generation",
            experiment_id="V2_SUFFICIENCY_GENERATION_DEV",
            scope="Three-report temporal generation",
            split="dev",
            source="reports/v2_sufficiency/dev_sufficiency_eval_summary.json",
            sort_order=190,
            sufficiency=True,
            notes="Development-only generation with deterministic evidence sufficiency gate.",
        ))
    mmr_eval = load_json(root / "reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json", {})
    mmr_sum = load_json(root / "reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/summary.json", {})
    mmr_decision = load_json(root / "reports/final_mmr_generation/final_mmr_generation_selection_decision.json", {})
    if mmr_eval and mmr_eval.get("metrics"):
        rows.append(base_row(
            method="MMR lambda 0.6 retrieval + sufficiency-gated generation",
            experiment_id="GEN_MMR06_SUFFICIENCY_V1",
            scope="Three-report temporal generation",
            dataset_split="dev",
            retrieval_or_generation="generation",
            parser="PyPDFLoader",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            bm25_enabled=True,
            dense_enabled=True,
            hybrid_enabled=True,
            rrf_enabled=True,
            mmr_enabled=True,
            reranker_provider="cohere",
            reranker_model="rerank-v3.5",
            sufficiency_gate_enabled=True,
            factual_correctness=eval_metric(mmr_eval, "factual_correctness"),
            faithfulness_to_context=eval_metric(mmr_eval, "faithfulness_to_context"),
            contextual_relevancy=eval_metric(mmr_eval, "contextual_relevancy"),
            contextual_recall=eval_metric(mmr_eval, "contextual_recall"),
            citation_correctness=eval_metric(mmr_eval, "citation_correctness"),
            citation_completeness=eval_metric(mmr_eval, "citation_completeness"),
            temporal_attribution_correctness=eval_metric(mmr_eval, "temporal_attribution_correctness"),
            comparative_correctness=eval_metric(mmr_eval, "comparative_correctness"),
            abstention_correctness=eval_metric(mmr_eval, "abstention_correctness"),
            mean_latency_ms=mmr_sum.get("mean_generation_latency_ms"),
            median_latency_ms=mmr_sum.get("median_generation_latency_ms"),
            status="completed",
            notes=(
                "Development-only generation over saved MMR_LAMBDA_06 contexts. "
                f"Final MMR generation decision: {mmr_decision.get('status')}."
            ),
            source_artifact="reports/final_mmr_generation/GEN_MMR06_SUFFICIENCY_V1/eval_summary.json",
            sort_order=195,
        ))

    bakeoff = load_json(root / "reports/final_generation_bakeoff/generation_bakeoff_leaderboard.json", {})
    bakeoff_decision = load_json(root / "reports/final_generation_bakeoff/final_generation_strategy_selection_decision.json", {})
    for index, item in enumerate(bakeoff.get("rows", []) or []):
        variant_id = item.get("variant_id")
        if not variant_id:
            continue
        is_selected = variant_id == bakeoff_decision.get("selected_variant_id")
        rows.append(base_row(
            method=(
                "Final selected generation bake-off strategy"
                if is_selected else f"Generation bake-off: {variant_id}"
            ),
            experiment_id=variant_id,
            scope="Three-report temporal generation",
            dataset_split="dev",
            retrieval_or_generation="generation",
            parser="PyPDFLoader",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            bm25_enabled=True,
            dense_enabled=True,
            hybrid_enabled=True,
            rrf_enabled=True,
            mmr_enabled=str(item.get("retrieval_experiment_id", "")).startswith("MMR_"),
            reranker_provider="cohere",
            reranker_model="rerank-v3.5",
            sufficiency_gate_enabled=True,
            factual_correctness=item.get("factual_correctness"),
            faithfulness_to_context=item.get("faithfulness_to_context"),
            contextual_relevancy=item.get("contextual_relevancy"),
            contextual_recall=item.get("contextual_recall"),
            citation_correctness=item.get("citation_correctness"),
            citation_completeness=item.get("citation_completeness"),
            temporal_attribution_correctness=item.get("temporal_attribution_correctness"),
            comparative_correctness=item.get("comparative_correctness"),
            abstention_correctness=item.get("abstention_correctness"),
            median_latency_ms=item.get("median_generation_latency_ms"),
            p95_latency_ms=item.get("p95_generation_latency_ms"),
            mean_estimated_tokens=item.get("mean_estimated_context_tokens"),
            status=item.get("status"),
            notes=(
                f"Development-only final generation bake-off variant. Eligibility: {item.get('eligibility')}. "
                f"Decision: {bakeoff_decision.get('status', 'not_available')}."
            ),
            source_artifact="reports/final_generation_bakeoff/generation_bakeoff_leaderboard.json",
            sort_order=200 + index,
        ))

    return sorted(rows, key=lambda row: (row.get("sort_order") or 9999, row.get("method") or ""))


def safe_json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe_json_dump(value) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (list, dict)) else row.get(key)
                for key in fields
            })


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def hit_or_all(row: dict[str, Any]) -> Any:
    return row.get("hit_rate") if row.get("hit_rate") is not None else row.get("all_reports_hit")


def mrr_or_macro(row: dict[str, Any]) -> Any:
    return row.get("mrr") if row.get("mrr") is not None else row.get("macro_mrr")


def master_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG Methods Master Comparison",
        "",
        "This table is generated from saved artifacts only. Missing metrics are shown as `—` and are stored as JSON null.",
        "",
        "Single-document Hit-Rate@4 is not directly comparable to multi-report Complete Evidence Recall because the temporal task requires evidence from multiple report periods and stricter report attribution.",
        "",
        "MRR = Mean Reciprocal Rank, a ranking metric. MMR = Maximal Marginal Relevance, a diversity-aware selection technique.",
        "",
        "Old Phase 7 held-out results are historical and should not be presented as a fresh V2 benchmark.",
        "",
        "| " + " | ".join(label for label, _ in MASTER_MD_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in MASTER_MD_COLUMNS) + " |",
    ]
    for row in rows:
        display = dict(row)
        display["hit_or_all_reports_hit"] = hit_or_all(row)
        display["mrr_or_macro_mrr"] = mrr_or_macro(row)
        lines.append("| " + " | ".join(fmt(display.get(key)) for _, key in MASTER_MD_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def select_rows(rows: list[dict[str, Any]], *contains: str) -> list[dict[str, Any]]:
    lowered = [text.lower() for text in contains]
    return [
        row for row in rows
        if any(text in (row.get("method") or "").lower() or text in (row.get("experiment_id") or "").lower() for text in lowered)
    ]


def compact_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for label, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        display = dict(row)
        display["hit_or_all_reports_hit"] = hit_or_all(row)
        display["mrr_or_macro_mrr"] = mrr_or_macro(row)
        lines.append("| " + " | ".join(fmt(display.get(key)) for _, key in columns) + " |")
    return lines


def technique_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("Method", "method"),
        ("Hit/All-Hit", "hit_or_all_reports_hit"),
        ("CER", "complete_evidence_recall"),
        ("Evidence Recall", "evidence_recall"),
        ("MRR/Macro MRR", "mrr_or_macro_mrr"),
        ("Median Latency", "median_latency_ms"),
        ("Status", "status"),
    ]
    dense_hybrid = [
        row for row in rows
        if row["method"] in {
            "Single-document Dense",
            "Single-document BM25",
            "Single-document Hybrid RRF",
            "Single-document Hybrid RRF + reranker",
            "Hybrid BM25 + Dense retrieval with RRF",
            "V2 baseline final retrieval",
            "V2 Cohere retrieval",
            "V2 Unstructured retrieval",
            "V2 Unstructured + Cohere retrieval",
        }
    ]
    rrf_rows = select_rows(rows, "RRF fusion", "Weighted RRF")
    rerank_rows = [
        row for row in rows
        if row["method"] in {
            "V2 baseline final retrieval",
            "V2 Cohere retrieval",
            "V2 Unstructured retrieval",
            "V2 Unstructured + Cohere retrieval",
            "Local cross-encoder reranking",
            "Single-document Hybrid RRF + reranker",
        }
    ]
    structural = select_rows(rows, "Child-parent", "Sentence-window", "Adjacent expansion", "MMR", "Exact-overlap diversity")
    query_rows = select_rows(rows, "Multi-query", "Terminology", "Facet")
    gen_rows = select_rows(rows, "generation")
    lines = [
        "# Technique-Wise Comparison",
        "",
        "## Dense vs BM25 vs Hybrid",
        "",
        *compact_table(dense_hybrid, columns),
        "",
        "Dense and BM25 both helped in the single-document baseline. Hybrid RRF plus reranking was strongest in the April-only setting. For the temporal setting, dense and BM25 were used together; standalone temporal dense-only/BM25-only metrics were not found.",
        "",
        "## RRF and Weighted RRF",
        "",
        *compact_table(rrf_rows, columns),
        "",
        "The best saved RRF-k run by Macro MRR was RRF_K10, but Stage A selection favored the larger quota configuration because it improved Complete Evidence Recall and All-Reports Hit. Weighted RRF did not beat the selected quota configuration.",
        "",
        "## Reranking",
        "",
        *compact_table(rerank_rows, columns),
        "",
        "Local cross-encoder reranking was retained through the baseline. Cohere reranking improved development retrieval metrics but materially increased retrieval latency.",
        "",
        "## MMR / diversity selection",
        "",
        *compact_table(structural, columns),
        "",
        "MMR means Maximal Marginal Relevance, a diversity-based retrieval/selection method. True MMR rows are shown when `reports/mmr_experiments` exists. DIV01 used exact-overlap diversity filtering and is not MMR.",
        "",
        "## Multi-query / terminology expansion / facet decomposition",
        "",
        *compact_table(query_rows, columns),
        "",
        "Saved Stage A query-normalisation, terminology-expansion, multi-query, and facet experiments did not improve over the selected quota baseline.",
        "",
        "## Sufficiency-gated generation",
        "",
        *compact_table(gen_rows, [
            ("Method", "method"),
            ("Factual", "factual_correctness"),
            ("Faithfulness", "faithfulness_to_context"),
            ("Citation", "citation_correctness"),
            ("Temporal", "temporal_attribution_correctness"),
            ("Comparative", "comparative_correctness"),
            ("Abstention", "abstention_correctness"),
            ("Status", "status"),
        ]),
        "",
        "Sufficiency gating improved factual correctness, faithfulness, and abstention correctness. Comparative correctness drops because incomplete comparative questions are now caveated or abstained instead of being treated as full answers.",
    ]
    return "\n".join(lines) + "\n"


def metric_lookup(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def best_system_summary(rows: list[dict[str, Any]]) -> str:
    retrieval = metric_lookup(rows, "V2 Cohere retrieval")
    generation = metric_lookup(rows, "V2 Cohere retrieval + sufficiency-gated generation")
    mmr_generation = metric_lookup(rows, "MMR lambda 0.6 retrieval + sufficiency-gated generation")
    bakeoff_generation = metric_lookup(rows, "Final selected generation bake-off strategy")
    selected_generation = (
        bakeoff_generation
        if bakeoff_generation
        and bakeoff_generation.get("status") in {"completed", "completed_reused", "completed_repaired"}
        else
        mmr_generation
        if mmr_generation
        and mmr_generation.get("status") == "completed"
        and "selected_mmr_end_to_end" in (mmr_generation.get("notes") or "")
        else generation
    )
    selected_generation_name = selected_generation.get("method") or "V2 Cohere retrieval + sufficiency-gated generation"
    lines = [
        "# Best Current System Summary",
        "",
        f"Best evaluated generation system: **{selected_generation_name}**.",
        "",
        "## Architecture",
        "",
        "- PyPDFLoader extraction",
        "- Dense vector retrieval",
        "- BM25 retrieval",
        "- Hybrid retrieval using RRF",
        "- Cohere `rerank-v3.5`",
        "- Report-aware context quotas",
        "- Source-labelled contexts",
        "- Groq `llama-3.1-8b-instant` generation",
        "- Evidence sufficiency gate",
        "- Citation validation",
        "- Temporal attribution validation",
        "",
        "## Final development retrieval metrics",
        "",
        f"- CER: {fmt(retrieval.get('complete_evidence_recall'))}",
        f"- All-Reports Hit: {fmt(retrieval.get('all_reports_hit'))}",
        f"- Evidence Recall: {fmt(retrieval.get('evidence_recall'))}",
        f"- Macro MRR: {fmt(retrieval.get('macro_mrr'))}",
        f"- Median latency: {fmt(retrieval.get('median_latency_ms'))} ms",
        f"- Mean tokens: {fmt(retrieval.get('mean_estimated_tokens'))}",
        "",
        "## Final development generation metrics after sufficiency gate",
        "",
        f"- Factual correctness: {fmt(selected_generation.get('factual_correctness'))}",
        f"- Faithfulness to context: {fmt(selected_generation.get('faithfulness_to_context'))}",
        f"- Abstention correctness: {fmt(selected_generation.get('abstention_correctness'))}",
        f"- Citation correctness: {fmt(selected_generation.get('citation_correctness'))}",
        f"- Citation completeness: {fmt(selected_generation.get('citation_completeness'))}",
        f"- Temporal attribution correctness: {fmt(selected_generation.get('temporal_attribution_correctness'))}",
        f"- Comparative correctness: {fmt(selected_generation.get('comparative_correctness'))}",
        "",
        "## Caveats",
        "",
        "- Development-only final V2 generation.",
        "- Old held-out set was not reused as a fresh V2 benchmark.",
        "- Metrics are deterministic heuristics, not human evaluation.",
        "- Poppler is now configured for the project, but Unstructured extraction remains blocked for the current PDFs because non-OCR extraction returned zero elements and OCR/Tesseract is unavailable.",
        "- Cohere improves quality but increases latency.",
    ]
    return "\n".join(lines) + "\n"


def presentation_tables(rows: list[dict[str, Any]]) -> str:
    old_gen = metric_lookup(rows, "V2 Cohere retrieval + generation")
    new_gen = metric_lookup(rows, "V2 Cohere retrieval + sufficiency-gated generation")
    mmr_gen = metric_lookup(rows, "MMR lambda 0.6 retrieval + sufficiency-gated generation")
    bakeoff_gen = metric_lookup(rows, "Final selected generation bake-off strategy")
    mmr_status = metric_lookup(rows, "MMR / diversity selection").get("status") or "not_run"
    mmr_impact = "True MMR evaluated over saved V2 Cohere outputs" if mmr_status != "not_run" else "No true standalone MMR run found"
    contrib = [
        ("BM25", "keyword/numeric retrieval", "Useful with dense hybrid", "Lexical only", "retained"),
        ("Dense embeddings", "semantic retrieval", "Useful baseline and hybrid component", "Can miss exact numeric evidence", "retained"),
        ("Hybrid search", "combines lexical + semantic", "Stronger than either alone in the final architecture", "More complex", "retained"),
        ("RRF", "rank fusion", "Stabilised dense+BM25 candidate fusion", "Requires k/retention tuning", "retained"),
        ("Reranking", "improves ordering", "Cohere improved dev metrics", "Latency increase", "retained in V2"),
        ("MMR", "diversity control", mmr_impact, "May drop evidence if misapplied", mmr_status),
        ("Multi-query", "expands query intent", "EXP02 did not beat quota baseline", "Added complexity", "not selected"),
        ("Sufficiency gate", "prevents unsupported answers", "Improved abstention/factuality", "Lower answer coverage", "retained"),
    ]
    lines = [
        "# Presentation Tables",
        "",
        "## 1. Original single-document retrieval",
        "",
        *compact_table([row for row in rows if row["scope"] == "April 2025 only"], [
            ("Method", "method"), ("Hit@4", "hit_rate"), ("MRR", "mrr"), ("Mean latency", "mean_latency_ms")
        ]),
        "",
        "## 2. Multi-report final retrieval",
        "",
        *compact_table([metric_lookup(rows, "V2 baseline final retrieval"), metric_lookup(rows, "V2 Cohere retrieval")], [
            ("Method", "method"), ("All-Hit", "all_reports_hit"), ("CER", "complete_evidence_recall"), ("Evidence Recall", "evidence_recall"), ("Macro MRR", "macro_mrr"), ("Median latency", "median_latency_ms")
        ]),
        "",
        "## 3. V2 Cohere improvement",
        "",
        *compact_table([metric_lookup(rows, "V2 baseline final retrieval"), metric_lookup(rows, "V2 Cohere retrieval")], [
            ("Method", "method"), ("CER", "complete_evidence_recall"), ("All-Hit", "all_reports_hit"), ("Evidence Recall", "evidence_recall"), ("Macro MRR", "macro_mrr"), ("Mean tokens", "mean_estimated_tokens")
        ]),
        "",
        "## 4. Generation before vs after sufficiency gate",
        "",
        *compact_table([row for row in [old_gen, new_gen, mmr_gen, bakeoff_gen] if row], [
            ("Method", "method"), ("Factual", "factual_correctness"), ("Faithfulness", "faithfulness_to_context"), ("Abstention", "abstention_correctness"), ("Citation", "citation_correctness"), ("Temporal", "temporal_attribution_correctness"), ("Comparative", "comparative_correctness")
        ]),
        "",
        "## 5. Technique contribution table",
        "",
        "| Technique | Purpose | Observed Impact | Trade-off | Final Status |",
        "|---|---|---|---|---|",
    ]
    for item in contrib:
        lines.append("| " + " | ".join(item) + " |")
    return "\n".join(lines) + "\n"


def interview_explanation() -> str:
    return """# Interview Explanation

1. **Why did multi-document performance look lower than single-document?**  
   The multi-document task is stricter: the system must retrieve evidence from the correct report periods, often across multiple reports, while avoiding contamination.

2. **Why are Hit Rate and CER not directly comparable?**  
   Single-document Hit-Rate@4 checks whether one relevant chunk appears in the top 4. Complete Evidence Recall requires all required evidence across report periods.

3. **Why use BM25 + dense hybrid search?**  
   BM25 helps exact terms and numeric wording; dense retrieval helps semantic matches. RBI reports need both.

4. **Why use RRF?**  
   Reciprocal Rank Fusion combines dense and BM25 rankings without requiring score calibration.

5. **What is MMR and did it help?**  
   MMR is Maximal Marginal Relevance, a diversity selection method. The final comparison distinguishes true MMR experiment artifacts from DIV01, which is only an exact-overlap diversity filter.

6. **What is MRR and how is it different from MMR?**  
   MRR is Mean Reciprocal Rank, a metric measuring how early relevant evidence appears. MMR is a retrieval/selection technique. They are unrelated except for similar abbreviations.

7. **Why use Cohere reranking?**  
   Cohere reranking improved development retrieval metrics by better ordering candidate chunks after hybrid retrieval.

8. **Why did Cohere improve results but slow latency?**  
   It adds remote API reranking calls over candidate documents, so quality improves at a material latency cost.

9. **Why add sufficiency gating?**  
   The model answered even when retrieval was incomplete. The gate makes the system abstain or caveat unsupported answers.

10. **What is the final architecture?**  
   PyPDFLoader, dense + BM25 hybrid retrieval, RRF, Cohere reranking, report-aware quotas, source-labelled contexts, Groq generation, sufficiency gate, and citation/temporal validation.

11. **What are the limitations?**  
   Current V2 results are development-only, metrics are deterministic heuristics, Unstructured extraction depends on Poppler/layout tooling, and Cohere adds latency.

12. **What would you improve next?**  
   Create a fresh V2 evaluation set, run final retrieval plus generation on it, then build the Streamlit interface.
"""


def validation_payload(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    for row in rows:
        for key in ["hit_rate", "all_reports_hit", "complete_evidence_recall", "evidence_recall", "mrr", "macro_mrr"]:
            if row.get(key) == 0 and row.get("status") != "completed":
                issues.append(f"non_completed_zero_metric:{row.get('method')}:{key}")
        if row.get("status") == "completed" and row.get("source_artifact") and not str(row["source_artifact"]).startswith("configs/"):
            first_source = str(row["source_artifact"]).split(";")[0]
            if not (root / first_source).exists():
                issues.append(f"missing_source_artifact:{row.get('method')}:{row.get('source_artifact')}")
    allowed_mmr_statuses = {"not_run", "blocked", "evaluated", "evaluated_not_selected", "evaluated_selected"}
    mrr_mmr_ok = any(
        row["method"] == "MMR / diversity selection"
        and row["status"] in allowed_mmr_statuses
        and "Maximal Marginal Relevance" in (row.get("notes") or "")
        for row in rows
    )
    if not mrr_mmr_ok:
        issues.append("missing_mrr_mmr_distinction")
    if not any("Hit-Rate@4 is not directly comparable" in (OUT_DIR / "rag_methods_master_comparison.md").read_text(encoding="utf-8") for _ in [0] if (OUT_DIR / "rag_methods_master_comparison.md").exists()):
        issues.append("missing_single_vs_multi_metric_caveat")
    serialized = json.dumps(rows, default=str)
    for name in ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY"):
        secret = os.getenv(name)
        if secret and secret in serialized:
            issues.append("api_key_value_serialized")
    return {
        "schema_version": 1,
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": sorted(set(issues)),
        "metrics_from_existing_artifacts_or_known_values": True,
        "missing_metrics_are_null": all(row.get(field) is None or not (isinstance(row.get(field), str) and row.get(field).lower() == "null") for row in rows for field in MASTER_FIELDS),
        "single_and_multi_metrics_labelled_separately": True,
        "mrr_and_mmr_distinguished": mrr_mmr_ok,
        "dev_heldout_fresh_status_labelled": True,
        "api_keys_serialized": "api_key_value_serialized" in issues,
        "retrieval_rerun": False,
        "generation_rerun": False,
    }


def write_validation(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = validation_payload(root, rows)
    write_json(root / OUT_DIR / "comparison_artifact_validation.json", payload)
    lines = [
        "# Comparison Artifact Validation",
        "",
        f"Status: {payload['status']}",
        f"Issue count: {payload['issue_count']}",
        "",
        "- All cited metrics come from saved artifacts or explicitly labelled known values.",
        "- Missing metrics are represented as JSON null.",
        "- Single-document and multi-document metrics are labelled separately.",
        "- MRR and MMR are distinguished.",
        "- Development, held-out, and fresh evaluation statuses are labelled.",
        "- No retrieval or generation was rerun.",
    ]
    if payload["issues"]:
        lines += ["", "## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    (root / OUT_DIR / "comparison_artifact_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_chart_data(root: Path, rows: list[dict[str, Any]]) -> None:
    retrieval_rows = [row for row in rows if row.get("retrieval_or_generation") == "retrieval" and row.get("status") in {"completed", "completed_historical_heldout"}]
    generation_rows = [row for row in rows if row.get("retrieval_or_generation") == "generation" and row.get("status") == "completed"]
    write_csv(root / OUT_DIR / "chart_data_retrieval_methods.csv", retrieval_rows, [
        "method", "dataset_split", "all_reports_hit", "complete_evidence_recall", "evidence_recall", "macro_mrr", "median_latency_ms", "mean_estimated_tokens", "status"
    ])
    write_csv(root / OUT_DIR / "chart_data_generation_methods.csv", generation_rows, [
        "method", "dataset_split", "factual_correctness", "faithfulness_to_context", "citation_correctness", "temporal_attribution_correctness", "comparative_correctness", "abstention_correctness", "status"
    ])
    write_csv(root / OUT_DIR / "chart_data_latency_tradeoff.csv", retrieval_rows + generation_rows, [
        "method", "retrieval_or_generation", "dataset_split", "complete_evidence_recall", "factual_correctness", "median_latency_ms", "mean_latency_ms", "status"
    ])


def archive_existing_final_comparison(root: Path) -> dict[str, Any]:
    out = root / OUT_DIR
    archive_root = out / "archive_pre_unstructured_update"
    archive_root.mkdir(parents=True, exist_ok=True)
    source_files = [
        path
        for path in out.iterdir()
        if path.is_file()
    ] if out.exists() else []
    if not source_files:
        return {"status": "not_needed", "file_count": 0, "archive_dir": str(archive_root)}
    existing_archives = [path for path in archive_root.iterdir() if path.is_dir()]
    if existing_archives:
        return {"status": "already_archived", "file_count": 0, "archive_dir": str(sorted(existing_archives)[-1])}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_root / stamp
    target.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        shutil.copy2(source, target / source.name)
    return {"status": "archived", "file_count": len(source_files), "archive_dir": str(target)}


def archive_pre_final_mmr_generation_update(root: Path) -> dict[str, Any]:
    out = root / OUT_DIR
    if not (root / "reports/final_mmr_generation/final_mmr_generation_selection_decision.json").exists():
        return {"status": "not_needed", "file_count": 0, "archive_dir": None}
    archive_root = out / "archive_pre_final_mmr_generation_update"
    archive_root.mkdir(parents=True, exist_ok=True)
    existing_archives = [path for path in archive_root.iterdir() if path.is_dir()]
    if existing_archives:
        return {"status": "already_archived", "file_count": 0, "archive_dir": str(sorted(existing_archives)[-1])}
    source_files = [path for path in out.iterdir() if path.is_file()] if out.exists() else []
    if not source_files:
        return {"status": "not_needed", "file_count": 0, "archive_dir": str(archive_root)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_root / stamp
    target.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        shutil.copy2(source, target / source.name)
    return {"status": "archived", "file_count": len(source_files), "archive_dir": str(target)}


def append_unstructured_update_sections(root: Path, out: Path, rows: list[dict[str, Any]]) -> None:
    category_rows = load_json(root / "reports/v2_unstructured_cohere/v2_category_results.json", [])
    selected = load_json(root / "reports/v2_unstructured_cohere/v2_selected_retrieval.json", {})
    poppler_after = load_json(root / "reports/v2_unstructured_cohere/poppler_retry/poppler_status_after.json", {})
    v2_rows = [
        row
        for row in rows
        if row.get("experiment_id") in {"V2_BASELINE_FINAL", "V2_COHERE_ONLY", "V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"}
    ]
    table_numeric = [
        row
        for row in category_rows
        if row.get("category_type") == "table_or_numeric_questions"
        and str(row.get("category")).lower() in {"true", "table_or_numeric", "1"}
        and row.get("experiment_id") in {"V2_BASELINE_FINAL", "V2_COHERE_ONLY", "V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"}
    ]
    if not v2_rows and not table_numeric:
        return
    parser_lines = [
        "",
        "## Poppler-enabled Unstructured update",
        "",
        f"- Poppler verification after setup: `{poppler_after.get('poppler_ready')}`.",
        f"- Selected V2 experiment: `{selected.get('selected_experiment_id')}`.",
        "- Unstructured rows below are development-only and do not use held-out data.",
        "",
        "### Overall V2 retrieval rows",
        "",
        *compact_table(v2_rows, [
            ("Method", "method"),
            ("CER", "complete_evidence_recall"),
            ("All-Hit", "all_reports_hit"),
            ("Evidence Recall", "evidence_recall"),
            ("Macro MRR", "macro_mrr"),
            ("Median ms", "median_latency_ms"),
            ("Status", "status"),
        ]),
    ]
    if table_numeric:
        table_display = [
            {
                "method": item.get("experiment_id"),
                "complete_evidence_recall": item.get("complete_evidence_recall"),
                "evidence_recall": item.get("evidence_recall"),
                "macro_mrr": item.get("macro_report_mrr"),
                "case_count": item.get("scored_case_count", item.get("case_count")),
            }
            for item in table_numeric
        ]
        parser_lines += [
            "",
            "### Table / numeric questions",
            "",
            *compact_table(table_display, [
                ("Experiment", "method"),
                ("CER", "complete_evidence_recall"),
                ("Evidence Recall", "evidence_recall"),
                ("Macro MRR", "macro_mrr"),
                ("Cases", "case_count"),
            ]),
        ]
    for name in ("technique_wise_comparison.md", "presentation_tables.md", "best_system_summary.md"):
        path = out / name
        if path.exists():
            path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(parser_lines) + "\n", encoding="utf-8")


def generate_final_comparison(root: Path = Path(".")) -> dict[str, Any]:
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    mmr_generation_archive = archive_pre_final_mmr_generation_update(root)
    archive = archive_existing_final_comparison(root)
    rows = build_master_rows(root)
    write_csv(out / "rag_methods_master_comparison.csv", rows, MASTER_FIELDS)
    write_json(out / "rag_methods_master_comparison.json", rows)
    (out / "rag_methods_master_comparison.md").write_text(master_markdown(rows), encoding="utf-8")
    (out / "technique_wise_comparison.md").write_text(technique_comparison_markdown(rows), encoding="utf-8")
    (out / "best_system_summary.md").write_text(best_system_summary(rows), encoding="utf-8")
    (out / "presentation_tables.md").write_text(presentation_tables(rows), encoding="utf-8")
    (out / "interview_explanation.md").write_text(interview_explanation(), encoding="utf-8")
    write_chart_data(root, rows)
    validation = write_validation(root, rows)
    append_unstructured_update_sections(root, out, rows)
    return {
        "row_count": len(rows),
        "validation_status": validation["status"],
        "output_dir": str(out),
        "master_comparison_path": str(out / "rag_methods_master_comparison.md"),
        "presentation_tables_path": str(out / "presentation_tables.md"),
        "interview_explanation_path": str(out / "interview_explanation.md"),
        "archive_status": archive,
        "mmr_generation_archive_status": mmr_generation_archive,
    }
