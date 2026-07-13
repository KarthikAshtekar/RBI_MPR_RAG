from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder

from rbi_rag.bm25_preprocessing import finance_tokens
from rbi_rag.experiment_tracing import (
    LATENCY_FIELDS,
    StageTimer,
    context_statistics,
    first_evidence_rank,
    recompute_loss_stage,
    validate_latency_schema,
)
from rbi_rag.fusion import reciprocal_rank_fusion
from rbi_rag.multi_config import MultiReportConfig
from rbi_rag.multi_evaluation import load_jsonl
from rbi_rag.multi_index import build_multi_report_index
from rbi_rag.query_optimisation import decompose_facets, expand_query, normalise_retrieval_query
from rbi_rag.report_bm25 import BM25ByReport
from rbi_rag.report_registry import ReportRegistry
from rbi_rag.temporal_router import TemporalQueryRouter


ROOT = Path(".")
OUT = ROOT / "reports" / "optimisation"
ARCHIVE = OUT / "invalid_runs_pre_latency_fix"
ACTIVE_ARCHIVE = OUT / f"active_invalid_pre_repair_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
REQUIRED_FILES = {
    "config_snapshot.yaml",
    "environment.json",
    "index_manifest.json",
    "raw_results.json",
    "question_results.csv",
    "report_level_results.csv",
    "summary.json",
    "summary.md",
    "stage_diagnostics.csv",
}


def stable_json_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def groq_key_available() -> bool:
    if os.getenv("GROQ_API_KEY"):
        return True
    env_path = ROOT / ".env"
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GROQ_API_KEY") and "=" in line:
            return bool(line.split("=", 1)[1].strip().strip('"').strip("'"))
    return False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialise_items(items):
    return [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "page": doc.metadata["page"],
            "score": None if score is None else float(score),
            "report_id": doc.metadata["report_id"],
        }
        for doc, score in items
    ]


def ids(items):
    return [doc.metadata["chunk_id"] for doc, _ in items]


def pages(items):
    return [doc.metadata["page"] for doc, _ in items]


def scores(items):
    return [None if score is None else float(score) for _, score in items]


def deduplicate(items):
    selected = []
    seen = set()
    duplicate_count = 0
    for doc, score in items:
        key = (doc.metadata["report_id"], " ".join(doc.page_content.split()).lower())
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        selected.append((doc, score))
    return selected, duplicate_count


def combine_by_chunk(*ranked_lists):
    combined = {}
    order = {}
    sequence = 0
    for source_name, values in ranked_lists:
        for rank, (doc, score) in enumerate(values, 1):
            cid = doc.metadata["chunk_id"]
            if cid not in combined:
                combined[cid] = [doc, score, set(), sequence]
                order[cid] = sequence
                sequence += 1
            combined[cid][2].add(source_name)
    return [(value[0], value[1]) for _, value in sorted(combined.items(), key=lambda item: order[item[0]])]


def weighted_rrf(ranked_lists, *, rrf_k: int, dense_weight: float, bm25_weight: float, limit: int):
    scores_by_id = {}
    docs = {}
    first_seen = {}
    sequence = 0
    for weight, values in ((dense_weight, ranked_lists[0]), (bm25_weight, ranked_lists[1])):
        for rank, (doc, _) in enumerate(values, 1):
            cid = str(doc.metadata["chunk_id"])
            if cid not in first_seen:
                first_seen[cid] = sequence
                sequence += 1
            scores_by_id[cid] = scores_by_id.get(cid, 0.0) + weight / (rrf_k + rank)
            docs[cid] = doc
    ordered = sorted(scores_by_id, key=lambda cid: (-scores_by_id[cid], first_seen[cid], cid))
    return [(docs[cid], scores_by_id[cid]) for cid in ordered[:limit]]


def select_quota(plan, config):
    quota = config.get("quota", [4, 3, 2])
    if plan.query_type == "trend_all_reports":
        return int(quota[2])
    if plan.query_type in ("single_report", "latest_report"):
        return int(quota[0])
    return int(quota[1])


def transform_query(question: str, config: dict):
    normalised = normalise_retrieval_query(question)
    base = normalised["normalised_query"] if config.get("query_normalisation") else question
    expanded = []
    if config.get("terminology_expansion") in {"append", "multi_query"}:
        expansion = expand_query(base)
        expanded = expansion["expansion_terms"]
    facets = decompose_facets(base) if config.get("facet_decomposition") else []
    retrieval_queries = [base]
    if config.get("terminology_expansion") == "multi_query" and expanded:
        retrieval_queries.append(" ".join([base] + expanded[:2]))
    if facets:
        retrieval_queries.extend(facets[:4])
    return {
        "original_query": question,
        "normalised_query": base,
        "expanded_queries": expanded,
        "facet_queries": facets,
        "retrieval_queries": list(dict.fromkeys(q for q in retrieval_queries if q.strip())),
        "reranker_query": base,
        "removed_phrases": normalised["removed_phrases"],
        "preserved_numbers": normalised["preserved_numbers"],
        "preserved_acronyms": normalised["preserved_acronyms"],
    }


def contains_expected_evidence(chunks, expected):
    combined = " ".join(" ".join(doc.page_content.lower().split()) for doc in chunks)
    return [" ".join(text.lower().split()) in combined for text in expected]


def run_reranker(cross_encoder, query, candidates):
    if not candidates:
        return []
    docs = [doc for doc, _ in candidates]
    values = cross_encoder.predict([[query, doc.page_content] for doc in docs])
    ranked = sorted(zip(docs, values), key=lambda item: (-float(item[1]), str(item[0].metadata["chunk_id"])))
    return [(doc, float(score)) for doc, score in ranked]


def run_question(case, config, resources):
    timer = StageTimer()
    report_details = {}
    selected_before_dedup = []
    candidate_sources = defaultdict(lambda: {"retriever": set(), "query": set(), "facet": set(), "expansion": set()})

    with timer.measure("routing_latency_ms"):
        plan = resources["router"].route(case["question"])

    with timer.measure("query_transformation_latency_ms"):
        query_plan = transform_query(case["question"], config)

    required = list(case["required_report_ids"])
    if plan.query_type == "unsupported_period":
        timings = timer.finish()
        stats = context_statistics([])
        row = {
            "experiment_id": config["id"],
            "question_id": case["question_id"],
            "split": "dev",
            "query_type": case["query_type"],
            "required_report_ids": required,
            "original_query": case["question"],
            "normalised_query": query_plan["normalised_query"],
            "expanded_queries": query_plan["expanded_queries"],
            "facet_queries": query_plan["facet_queries"],
            "removed_phrases": query_plan["removed_phrases"],
            "preserved_numbers": query_plan["preserved_numbers"],
            "preserved_acronyms": query_plan["preserved_acronyms"],
            "dataset_checksum": resources["dataset_checksum"],
            "index_fingerprint": resources["index_fingerprint"],
            "configuration_checksum": resources["configuration_checksum"],
            "per_report": {},
            "report_coverage": 0.0,
            "all_reports_hit": None,
            "macro_mrr": 0.0,
            "evidence_recall": None,
            "complete_evidence_recall": None,
            "contamination": 0,
            **stats,
            **timings,
        }
        return row, []

    quota = select_quota(plan, config)
    dense_by_report = {}
    bm25_by_report = {}
    union_by_report = {}
    fused_by_report = {}
    reranked_by_report = {}
    before_by_report = {}
    after_by_report = {}

    with timer.measure("dense_latency_ms"):
        for rid in required:
            values = []
            for q in query_plan["retrieval_queries"]:
                hits = resources["store"].similarity_search_with_relevance_scores(
                    q, k=int(config["dk"]), filter={"report_id": rid}
                )
                for doc, score in hits:
                    cid = doc.metadata["chunk_id"]
                    candidate_sources[cid]["retriever"].add("dense")
                    candidate_sources[cid]["query"].add(q)
                values.extend(hits)
            dense_by_report[rid] = combine_by_chunk(("dense", values))[: int(config["dk"])]

    with timer.measure("bm25_latency_ms"):
        for rid in required:
            values = []
            for q in query_plan["retrieval_queries"]:
                hits = resources["bm25"].search(rid, q, int(config["bk"]))
                for doc, _ in hits:
                    cid = doc.metadata["chunk_id"]
                    candidate_sources[cid]["retriever"].add("bm25")
                    candidate_sources[cid]["query"].add(q)
                values.extend(hits)
            bm25_by_report[rid] = combine_by_chunk(("bm25", values))[: int(config["bk"])]

    with timer.measure("candidate_union_latency_ms"):
        for rid in required:
            union_by_report[rid] = combine_by_chunk(("dense", dense_by_report[rid]), ("bm25", bm25_by_report[rid]))

    with timer.measure("fusion_latency_ms"):
        for rid in required:
            if float(config.get("dw", 1.0)) == 1.0 and float(config.get("bw", 1.0)) == 1.0:
                fused = reciprocal_rank_fusion(
                    [dense_by_report[rid], bm25_by_report[rid]],
                    rrf_k=int(config["rrf"]),
                    limit=int(config["retain"]),
                )
            else:
                fused = weighted_rrf(
                    [dense_by_report[rid], bm25_by_report[rid]],
                    rrf_k=int(config["rrf"]),
                    dense_weight=float(config.get("dw", 1.0)),
                    bm25_weight=float(config.get("bw", 1.0)),
                    limit=int(config["retain"]),
                )
            fused_by_report[rid] = fused

    with timer.measure("reranking_latency_ms"):
        for rid in required:
            reranker_input = fused_by_report[rid][: int(config["retain"])]
            reranked_by_report[rid] = run_reranker(resources["cross_encoder"], query_plan["reranker_query"], reranker_input)

    with timer.measure("selection_latency_ms"):
        for rid in required:
            chosen = reranked_by_report[rid][:quota]
            before_by_report[rid] = chosen
            selected_before_dedup.extend(chosen)

    with timer.measure("deduplication_latency_ms"):
        selected_after_dedup, _ = deduplicate(selected_before_dedup)
        after_by_report = {
            rid: [(doc, score) for doc, score in selected_after_dedup if doc.metadata["report_id"] == rid]
            for rid in required
        }

    with timer.measure("context_construction_latency_ms"):
        context = "\n\n".join(
            f"[{doc.metadata['report_id']} p.{doc.metadata['page']} {doc.metadata['chunk_id']}]\n{doc.page_content}"
            for doc, _ in selected_after_dedup
        )
        del context

    timings = timer.finish()
    selected_docs = [doc for doc, _ in selected_after_dedup]
    stats = context_statistics(selected_docs)
    hits = {}
    reciprocal = {}
    evidence_values = []
    report_rows = []
    for rid in required:
        gt = case.get("ground_truth", {}).get(rid, {})
        accepted = gt.get("accepted_pages", [])
        expected = gt.get("expected_evidence", [])
        final_pages = pages(after_by_report.get(rid, []))
        rank = first_evidence_rank(final_pages, accepted)
        evidence_hits = contains_expected_evidence([doc for doc, _ in after_by_report.get(rid, [])], expected)
        evidence_values.extend(evidence_hits)
        hits[rid] = bool(rank) if accepted else None
        reciprocal[rid] = 1.0 / rank if rank else 0.0
        trace = {
            "report_id": rid,
            "accepted_pages": accepted,
            "expected_evidence": expected,
            "dense_candidate_ids": ids(dense_by_report[rid]),
            "dense_candidate_pages": pages(dense_by_report[rid]),
            "dense_candidate_scores": scores(dense_by_report[rid]),
            "dense_first_evidence_rank": first_evidence_rank(pages(dense_by_report[rid]), accepted),
            "bm25_candidate_ids": ids(bm25_by_report[rid]),
            "bm25_candidate_pages": pages(bm25_by_report[rid]),
            "bm25_candidate_scores": scores(bm25_by_report[rid]),
            "bm25_first_evidence_rank": first_evidence_rank(pages(bm25_by_report[rid]), accepted),
            "candidate_union_ids": ids(union_by_report[rid]),
            "candidate_union_pages": pages(union_by_report[rid]),
            "rrf_candidate_ids": ids(fused_by_report[rid]),
            "rrf_candidate_pages": pages(fused_by_report[rid]),
            "rrf_scores": scores(fused_by_report[rid]),
            "rrf_first_evidence_rank": first_evidence_rank(pages(fused_by_report[rid]), accepted),
            "reranker_input_ids": ids(fused_by_report[rid]),
            "reranker_input_pages": pages(fused_by_report[rid]),
            "reranker_input_ranks": list(range(1, len(fused_by_report[rid]) + 1)),
            "reranker_output_ids": ids(reranked_by_report[rid]),
            "reranker_output_pages": pages(reranked_by_report[rid]),
            "reranker_scores": scores(reranked_by_report[rid]),
            "evidence_rank_before_reranking": first_evidence_rank(pages(fused_by_report[rid]), accepted),
            "evidence_rank_after_reranking": first_evidence_rank(pages(reranked_by_report[rid]), accepted),
            "selected_chunk_ids_before_dedup": ids(before_by_report.get(rid, [])),
            "selected_chunk_ids_after_dedup": ids(after_by_report.get(rid, [])),
            "selected_pages": final_pages,
            "accepted_evidence_found": bool(rank),
            "candidate_source_retriever": {cid: sorted(v["retriever"]) for cid, v in candidate_sources.items()},
            "candidate_source_query": {cid: sorted(v["query"]) for cid, v in candidate_sources.items()},
            "candidate_source_facet": {cid: sorted(v["facet"]) for cid, v in candidate_sources.items()},
            "candidate_source_expansion": {cid: sorted(v["expansion"]) for cid, v in candidate_sources.items()},
        }
        trace["loss_stage"] = recompute_loss_stage(trace)
        report_details[rid] = trace
        report_rows.append(
            {
                "experiment_id": config["id"],
                "question_id": case["question_id"],
                "query_type": case["query_type"],
                "report_id": rid,
                "dense_found": trace["dense_first_evidence_rank"] is not None,
                "bm25_found": trace["bm25_first_evidence_rank"] is not None,
                "union_found": any(page in set(accepted) for page in trace["candidate_union_pages"]),
                "fusion_found": trace["rrf_first_evidence_rank"] is not None,
                "reranker_input_found": any(page in set(accepted) for page in trace["reranker_input_pages"]),
                "reranker_found": trace["evidence_rank_after_reranking"] is not None,
                "final_found": bool(rank),
                "loss_stage": trace["loss_stage"],
            }
        )

    scored_hits = [value for value in hits.values() if value is not None]
    selected_report_ids = {doc.metadata["report_id"] for doc in selected_docs}
    contamination = sum(1 for doc in selected_docs if doc.metadata["report_id"] not in required)
    row = {
        "experiment_id": config["id"],
        "question_id": case["question_id"],
        "split": "dev",
        "query_type": case["query_type"],
        "required_report_ids": required,
        "original_query": case["question"],
        "normalised_query": query_plan["normalised_query"],
        "expanded_queries": query_plan["expanded_queries"],
        "facet_queries": query_plan["facet_queries"],
        "removed_phrases": query_plan["removed_phrases"],
        "preserved_numbers": query_plan["preserved_numbers"],
        "preserved_acronyms": query_plan["preserved_acronyms"],
        "dataset_checksum": resources["dataset_checksum"],
        "index_fingerprint": resources["index_fingerprint"],
        "configuration_checksum": resources["configuration_checksum"],
        "per_report": report_details,
        "report_chunk_counts": Counter(doc.metadata["report_id"] for doc in selected_docs),
        "report_coverage": sum(rid in selected_report_ids for rid in required) / len(required) if required else 0.0,
        "all_reports_hit": all(scored_hits) if scored_hits else None,
        "per_report_hit": hits,
        "per_report_mrr": reciprocal,
        "macro_mrr": sum(reciprocal.values()) / len(required) if required else 0.0,
        "evidence_recall": sum(evidence_values) / len(evidence_values) if evidence_values else None,
        "complete_evidence_recall": all(evidence_values) if evidence_values else None,
        "contamination": contamination,
        **stats,
        **timings,
    }
    return row, report_rows


def mean(values):
    return sum(values) / len(values) if values else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarise(config, rows, report_rows, started_at, finished_at, dataset_checksum, index_fingerprint):
    valid = [row for row in rows if row["required_report_ids"] and row["query_type"] != "unsupported_period"]
    scored = [row for row in valid if row["all_reports_hit"] is not None]
    evidence = [row for row in valid if row["evidence_recall"] is not None]
    latencies = [row["total_retrieval_latency_ms"] for row in valid]
    summary = {
        "experiment_id": config["id"],
        "family": config.get("family", "stage_a"),
        "config": config,
        "case_count": len(rows),
        "report_level_row_count": len(report_rows),
        "report_coverage": mean([row["report_coverage"] for row in valid]),
        "all_reports_hit": mean([float(row["all_reports_hit"]) for row in scored]),
        "macro_mrr": mean([row["macro_mrr"] for row in valid]),
        "evidence_recall": mean([row["evidence_recall"] for row in evidence]),
        "complete_evidence_recall": mean([float(row["complete_evidence_recall"]) for row in evidence]),
        "contamination": mean([row["contamination"] for row in valid]),
        "mean_selected_characters": mean([row["selected_character_count"] for row in valid]),
        "mean_estimated_tokens": mean([row["estimated_token_count"] for row in valid]),
        "mean_selected_chunks": mean([row["selected_chunk_count"] for row in valid]),
        "mean_unique_pages": mean([row["unique_page_count"] for row in valid]),
        "mean_repeated_text_ratio": mean([row["repeated_text_ratio"] for row in valid]),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "loss_stage_counts": dict(Counter(item["loss_stage"] for item in report_rows)),
        "dataset_sha256": dataset_checksum,
        "index_fingerprint": index_fingerprint,
        "configuration_checksum": stable_json_hash(config),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    return summary


def validate_experiment_dir(directory: Path, dataset_checksum: str, expected_count: int):
    issues = []
    files = {path.name for path in directory.iterdir() if path.is_file()}
    issues.extend(f"missing_file:{name}" for name in sorted(REQUIRED_FILES - files))
    if issues:
        return issues
    config = yaml.safe_load((directory / "config_snapshot.yaml").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((directory / "raw_results.json").read_text(encoding="utf-8"))
    env = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    if config.get("id") != directory.name:
        issues.append("experiment_id_mismatch")
    if summary.get("experiment_id") != directory.name:
        issues.append("summary_experiment_id_mismatch")
    if summary.get("configuration_checksum") != stable_json_hash(config):
        issues.append("configuration_checksum_mismatch")
    if summary.get("dataset_sha256") != dataset_checksum:
        issues.append("dataset_checksum_mismatch")
    if env.get("heldout_dataset_loaded") is not False:
        issues.append("heldout_loaded_flag_not_false")
    if len(raw) != expected_count:
        issues.append("raw_question_count_mismatch")
    if any(str(row.get("split")) == "test" or str(row.get("question_id", "")).startswith("test_") for row in raw):
        issues.append("heldout_case_present")
    for row in raw:
        issues.extend(validate_latency_schema(row))
        for field in ("selected_character_count", "estimated_token_count", "selected_chunk_count",
                      "unique_page_count", "duplicate_chunk_count", "repeated_text_ratio",
                      "dataset_checksum", "index_fingerprint", "configuration_checksum"):
            if field not in row:
                issues.append(f"missing_context_or_trace:{field}")
        if row.get("dataset_checksum") != dataset_checksum:
            issues.append("row_dataset_checksum_mismatch")
        if row.get("configuration_checksum") != stable_json_hash(config):
            issues.append("row_configuration_checksum_mismatch")
        for rid, trace in row.get("per_report", {}).items():
            if trace.get("report_id") != rid:
                issues.append("report_trace_id_mismatch")
            if recompute_loss_stage(trace) != trace.get("loss_stage"):
                issues.append("loss_stage_mismatch")
            if len(trace.get("dense_candidate_ids", [])) > int(config.get("dk", 0)):
                issues.append("dense_candidate_limit_exceeded")
            if len(trace.get("bm25_candidate_ids", [])) > int(config.get("bk", 0)):
                issues.append("bm25_candidate_limit_exceeded")
            if len(trace.get("rrf_candidate_ids", [])) > int(config.get("retain", 0)):
                issues.append("rrf_candidate_limit_exceeded")
            if not set(trace.get("selected_chunk_ids_after_dedup", [])).issubset(set(trace.get("reranker_output_ids", []))):
                issues.append("selected_not_in_reranker_output")
    return sorted(set(issues))


def write_csv(path: Path, rows):
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, Counter)) else value
                             for key, value in row.items()})


def write_experiment(path, config, environment, manifest, rows, report_rows, summary):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (path / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "index_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "raw_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=dict) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "summary.md").write_text("# " + config["id"] + "\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    write_csv(path / "question_results.csv", rows)
    write_csv(path / "report_level_results.csv", report_rows)
    write_csv(path / "stage_diagnostics.csv", report_rows)


def load_archived_configs():
    configs = []
    for snapshot in sorted(ARCHIVE.glob("*/config_snapshot.yaml")):
        config = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        configs.append(config)
    return configs


def build_additional_configs(parent):
    base = dict(parent)
    base.update({"parent_experiment": parent["id"]})
    return [
        {**base, "id": "QN01", "family": "query_normalisation", "query_normalisation": True},
        {**base, "id": "EXP01", "family": "terminology_expansion", "query_normalisation": True, "terminology_expansion": "append"},
        {**base, "id": "EXP02", "family": "terminology_expansion", "query_normalisation": True, "terminology_expansion": "multi_query"},
        {**base, "id": "FAC01", "family": "facet_decomposition", "query_normalisation": True, "facet_decomposition": True},
        {**base, "id": "BM02", "family": "bm25_preprocessing", "bm25_preprocessing": "finance_preserving"},
        {**base, "id": "AQ01", "family": "adaptive_quota", "quota": [6, 4, 3], "quota_strategy": "adaptive_bounded"},
        {**base, "id": "DIV01", "family": "diversity", "diversity_strategy": "exact_overlap_filter"},
        {**base, "id": "COMBINED_01", "family": "combined", "query_normalisation": True, "bm25_preprocessing": "finance_preserving"},
        {**base, "id": "COMBINED_02", "family": "combined", "query_normalisation": True, "terminology_expansion": "multi_query", "facet_decomposition": True, "quota": [6, 4, 3]},
        {**base, "id": "COMBINED_03", "family": "combined", "query_normalisation": True, "quota": [6, 4, 3], "diversity_strategy": "exact_overlap_filter"},
    ]


def warm_up(resources):
    query = "monetary policy inflation growth liquidity warmup"
    query_hash = sha256(query.encode("utf-8")).hexdigest()
    report_id = resources["registry"].enabled()[0].report_id
    values = {}
    started = time.perf_counter()
    resources["store"].similarity_search_with_relevance_scores(query, k=1, filter={"report_id": report_id})
    values["dense_warmup_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    resources["bm25"].search(report_id, query, 1)
    values["bm25_warmup_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    resources["cross_encoder"].predict([[query, "Reserve Bank of India monetary policy report."]])
    values["reranker_warmup_ms"] = (time.perf_counter() - started) * 1000
    values["embedding_warmup_ms"] = values["dense_warmup_ms"]
    values["warmup_performed"] = True
    values["warmup_query_hash"] = query_hash
    return values


def archive_active_dirs(configs):
    ACTIVE_ARCHIVE.mkdir(parents=True, exist_ok=True)
    moved = []
    for config in configs:
        path = OUT / config["id"]
        if path.exists():
            destination = ACTIVE_ARCHIVE / config["id"]
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(path), str(destination))
            moved.append(config["id"])
    if moved:
        (ACTIVE_ARCHIVE / "archive_manifest.json").write_text(json.dumps({"moved": moved, "created_at": now_iso()}, indent=2) + "\n", encoding="utf-8")
    return moved


def write_leaderboards(summaries):
    summaries = sorted(summaries, key=lambda item: (
        -(item.get("complete_evidence_recall") or 0),
        -(item.get("all_reports_hit") or 0),
        -(item.get("evidence_recall") or 0),
        -(item.get("macro_mrr") or 0),
        item.get("median_latency_ms") or 10**9,
        item.get("mean_estimated_tokens") or 10**9,
    ))
    (OUT / "experiment_leaderboard.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT / "experiment_leaderboard.csv", summaries)
    lines = ["# Valid Experiment Leaderboard", "", "| Rank | Experiment | Family | CER | Hit | Evidence | MRR | Median latency ms | Tokens |", "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for rank, summary in enumerate(summaries, 1):
        lines.append(
            f"| {rank} | {summary['experiment_id']} | {summary.get('family')} | {summary.get('complete_evidence_recall')} | "
            f"{summary.get('all_reports_hit')} | {summary.get('evidence_recall')} | {summary.get('macro_mrr')} | "
            f"{summary.get('median_latency_ms')} | {summary.get('mean_estimated_tokens')} |"
        )
    (OUT / "experiment_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def write_quota_comparison(summaries):
    by_id = {item["experiment_id"]: item for item in summaries}
    baseline = by_id.get("temporal_baseline", {})
    rows = []
    for exp_id in ("QUOTA_EXPANDED", "QUOTA_LARGE"):
        item = by_id.get(exp_id)
        if not item:
            continue
        token_gain = (item.get("mean_estimated_tokens") or 0) - (baseline.get("mean_estimated_tokens") or 0)
        latency_gain = ((item.get("mean_latency_ms") or 0) - (baseline.get("mean_latency_ms") or 0)) / 1000
        cer_gain = (item.get("complete_evidence_recall") or 0) - (baseline.get("complete_evidence_recall") or 0)
        er_gain = (item.get("evidence_recall") or 0) - (baseline.get("evidence_recall") or 0)
        rows.append({
            "experiment_id": exp_id,
            "complete_evidence_recall": item.get("complete_evidence_recall"),
            "all_reports_hit": item.get("all_reports_hit"),
            "evidence_recall": item.get("evidence_recall"),
            "macro_mrr": item.get("macro_mrr"),
            "report_coverage": item.get("report_coverage"),
            "contamination": item.get("contamination"),
            "mean_latency_ms": item.get("mean_latency_ms"),
            "median_latency_ms": item.get("median_latency_ms"),
            "p95_latency_ms": item.get("p95_latency_ms"),
            "selected_characters": item.get("mean_selected_characters"),
            "estimated_tokens": item.get("mean_estimated_tokens"),
            "selected_chunks": item.get("mean_selected_chunks"),
            "unique_pages": item.get("mean_unique_pages"),
            "repeated_text_ratio": item.get("mean_repeated_text_ratio"),
            "context_increase_vs_baseline": token_gain,
            "latency_increase_vs_baseline_ms": latency_gain * 1000,
            "complete_evidence_gain_per_1000_tokens": cer_gain / token_gain * 1000 if token_gain else None,
            "evidence_recall_gain_per_1000_tokens": er_gain / token_gain * 1000 if token_gain else None,
            "complete_evidence_gain_per_second": cer_gain / latency_gain if latency_gain else None,
            "evidence_recall_gain_per_second": er_gain / latency_gain if latency_gain else None,
        })
    (OUT / "quota_candidate_comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT / "quota_candidate_comparison.csv", rows)
    (OUT / "quota_candidate_comparison.md").write_text("# Quota Candidate Comparison\n\n" + "\n".join(f"- {r['experiment_id']}: CER={r['complete_evidence_recall']}, median_latency_ms={r['median_latency_ms']}" for r in rows) + "\n", encoding="utf-8")


def write_category_leaderboard(all_rows):
    rows = []
    grouped = defaultdict(list)
    for exp_id, raw_rows in all_rows.items():
        for row in raw_rows:
            grouped[(exp_id, row["query_type"])].append(row)
    for (exp_id, query_type), values in grouped.items():
        evidence = [row["evidence_recall"] for row in values if row["evidence_recall"] is not None]
        rows.append({
            "experiment_id": exp_id,
            "query_type": query_type,
            "case_count": len(values),
            "complete_evidence_recall": mean([float(row["complete_evidence_recall"]) for row in values if row["complete_evidence_recall"] is not None]),
            "all_reports_hit": mean([float(row["all_reports_hit"]) for row in values if row["all_reports_hit"] is not None]),
            "evidence_recall": mean(evidence),
            "macro_mrr": mean([row["macro_mrr"] for row in values]),
        })
    (OUT / "category_leaderboard.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT / "category_leaderboard.csv", rows)
    (OUT / "category_leaderboard.md").write_text("# Category Leaderboard\n\n" + "\n".join(f"- {r['experiment_id']} / {r['query_type']}: CER={r['complete_evidence_recall']}" for r in rows) + "\n", encoding="utf-8")


def write_selected(summaries):
    valid = [
        item for item in summaries
        if item.get("contamination") == 0 and item.get("report_coverage") == 1
    ]
    baseline = next((item for item in summaries if item["experiment_id"] == "temporal_baseline"), None)
    if baseline:
        valid = [item for item in valid if (item.get("median_latency_ms") or 0) <= 2 * (baseline.get("median_latency_ms") or 0) or (item.get("complete_evidence_recall") or 0) > (baseline.get("complete_evidence_recall") or 0)]
    ranked = write_leaderboards(valid)
    if not ranked:
        status = {"status": "incomplete_due_to_failed_experiments", "heldout_retrieval_run": False, "generation_evaluation_run": False, "groq_api_key_available": groq_key_available()}
        (OUT / "stage_a_selection_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return None
    winner = ranked[0]
    selected = {
        "selected_experiment_id": winner["experiment_id"],
        "selection_policy": "reports/optimisation/selection_policy.json",
        "selected": winner,
        "dataset_checksum": winner["dataset_sha256"],
        "index_fingerprint": winner["index_fingerprint"],
        "configuration_checksum": winner["configuration_checksum"],
        "heldout_retrieval_run": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": groq_key_available(),
    }
    checksum = stable_json_hash(selected)
    selected["selected_checksum"] = checksum
    (OUT / "stage_a_selected.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "stage_a_selected.md").write_text("# Stage A Selected\n\nSelected: `" + winner["experiment_id"] + "`\n\n```json\n" + json.dumps(selected, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    (OUT / "stage_a_selected_checksum.json").write_text(json.dumps({"sha256": checksum}, indent=2) + "\n", encoding="utf-8")
    (ROOT / "configs" / "stage_a_selected.yaml").write_text(yaml.safe_dump(winner["config"], sort_keys=False), encoding="utf-8")
    status = {
        "status": "completed_and_frozen",
        "selected_experiment_id": winner["experiment_id"],
        "valid_experiment_count": len(ranked),
        "heldout_retrieval_run": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": groq_key_available(),
    }
    (OUT / "stage_a_selection_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selected


def write_stage_a_report(summaries, rerun_status, selected, moved):
    valid_count = sum(row["integrity_status"] == "valid" for row in rerun_status)
    lines = [
        "# Stage A Optimisation Report",
        "",
        "## Runner defect",
        "",
        "The previous Stage A simulations reused a shared query-specific top-50 dense/BM25/reranker cache and then sliced it per experiment. That made per-experiment latency and traces invalid.",
        "",
        "## Cache boundary repair",
        "",
        "Static resources are reused: persistent Chroma collection, BM25 indexes, router, loaded reranker, registry, and development cases. Query-specific retrieval, fusion, reranking, selection, deduplication, and context construction are recomputed inside each experiment.",
        "",
        "## Warm-up",
        "",
        "A fixed non-evaluation warm-up query is used once per process. Warm-up timings are recorded in each `environment.json` and excluded from per-query latency.",
        "",
        "## Rerun status",
        "",
        f"Archived invalid active directories moved: {len(moved)}",
        f"Valid experiment outputs: {valid_count}/{len(rerun_status)}",
        "",
        "## Leaderboard",
        "",
        "| Experiment | Family | CER | Hit | Evidence | MRR | Median latency ms | Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda s: s["experiment_id"]):
        lines.append(f"| {summary['experiment_id']} | {summary.get('family')} | {summary.get('complete_evidence_recall')} | {summary.get('all_reports_hit')} | {summary.get('evidence_recall')} | {summary.get('macro_mrr')} | {summary.get('median_latency_ms')} | {summary.get('mean_estimated_tokens')} |")
    lines += [
        "",
        "## Selection",
        "",
        f"Selected: `{selected['selected_experiment_id'] if selected else 'none'}`",
        "",
        "Held-out retrieval was not run. Generation evaluation was not run. Groq availability is recorded as a boolean only.",
        "",
        "## Next phase",
        "",
        "Proceed to semantic chunking, parent-child retrieval, sentence-window retrieval, Docling/hybrid parsing, alternative embeddings, alternative rerankers, one-time held-out retrieval evaluation, and later generation evaluation.",
    ]
    (OUT / "stage_a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = MultiReportConfig.from_yaml(ROOT / "configs" / "multi_report.yaml")
    registry = ReportRegistry.from_yaml(cfg.reports_registry)
    cases_all = load_jsonl(cfg.dev_cases)
    cases = [case for case in cases_all if case.get("verification_status") == "verified"]
    dataset_checksum = file_sha(cfg.dev_cases)
    store, chunks, manifest = build_multi_report_index(cfg, registry)
    index_fingerprint = stable_json_hash(manifest)
    resources = {
        "store": store,
        "bm25": BM25ByReport(chunks),
        "cross_encoder": CrossEncoder(cfg.reranker_model),
        "router": TemporalQueryRouter(registry),
        "registry": registry,
        "dataset_checksum": dataset_checksum,
        "index_fingerprint": index_fingerprint,
    }
    warmup = warm_up(resources)
    archived = load_archived_configs()
    moved = archive_active_dirs(archived)
    rerun_status = []
    summaries = []
    all_rows = {}
    for config in archived:
        started_at = now_iso()
        wall_start = time.perf_counter()
        path = OUT / config["id"]
        resources["configuration_checksum"] = stable_json_hash(config)
        try:
            rows = []
            report_rows = []
            for case in cases:
                row, report = run_question(case, config, resources)
                rows.append(row)
                report_rows.extend(report)
            finished_at = now_iso()
            environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "runner_version": "stage_a_repaired_v1",
                "trace_schema_version": 2,
                "timing_schema_version": 1,
                "groq_api_key_available": groq_key_available(),
                "heldout_dataset_loaded": False,
                **warmup,
            }
            summary = summarise(config, rows, report_rows, started_at, finished_at, dataset_checksum, index_fingerprint)
            write_experiment(path, config, environment, manifest, rows, report_rows, summary)
            issues = validate_experiment_dir(path, dataset_checksum, len(cases))
            status = "valid" if not issues else "invalid"
            if status == "valid":
                summaries.append(summary)
                all_rows[config["id"]] = rows
            rerun_status.append({
                "experiment_id": config["id"],
                "archived_configuration_checksum": stable_json_hash(config),
                "rerun_configuration_checksum": summary["configuration_checksum"],
                "started_at": started_at,
                "finished_at": finished_at,
                "wall_runtime_seconds": time.perf_counter() - wall_start,
                "question_count": len(rows),
                "status": "completed",
                "integrity_status": status,
                "integrity_issue_count": len(issues),
                "failure_reason": "; ".join(issues),
            })
        except Exception as exc:
            rerun_status.append({
                "experiment_id": config["id"],
                "archived_configuration_checksum": stable_json_hash(config),
                "rerun_configuration_checksum": stable_json_hash(config),
                "started_at": started_at,
                "finished_at": now_iso(),
                "wall_runtime_seconds": time.perf_counter() - wall_start,
                "question_count": 0,
                "status": "failed",
                "integrity_status": "invalid",
                "integrity_issue_count": 1,
                "failure_reason": repr(exc),
            })
    write_csv(OUT / "rerun_status.csv", rerun_status)
    (OUT / "rerun_status.json").write_text(json.dumps(rerun_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "rerun_status.md").write_text("# Rerun Status\n\n" + "\n".join(f"- {row['experiment_id']}: {row['integrity_status']} {row['failure_reason']}" for row in rerun_status) + "\n", encoding="utf-8")

    write_leaderboards(summaries)
    write_quota_comparison(summaries)
    write_category_leaderboard(all_rows)

    if summaries:
        parent = sorted(summaries, key=lambda item: (-(item.get("complete_evidence_recall") or 0), item.get("median_latency_ms") or 10**9))[0]["config"]
        additional = build_additional_configs(parent)
        archive_active_dirs(additional)
        for config in additional:
            started_at = now_iso()
            path = OUT / config["id"]
            resources["configuration_checksum"] = stable_json_hash(config)
            rows, report_rows = [], []
            for case in cases:
                row, report = run_question(case, config, resources)
                rows.append(row)
                report_rows.extend(report)
            finished_at = now_iso()
            environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "runner_version": "stage_a_repaired_v1",
                "trace_schema_version": 2,
                "timing_schema_version": 1,
                "groq_api_key_available": groq_key_available(),
                "heldout_dataset_loaded": False,
                **warmup,
            }
            summary = summarise(config, rows, report_rows, started_at, finished_at, dataset_checksum, index_fingerprint)
            write_experiment(path, config, environment, manifest, rows, report_rows, summary)
            issues = validate_experiment_dir(path, dataset_checksum, len(cases))
            if not issues:
                summaries.append(summary)
                all_rows[config["id"]] = rows
    write_leaderboards(summaries)
    write_quota_comparison(summaries)
    write_category_leaderboard(all_rows)
    selected = write_selected(summaries)
    write_stage_a_report(summaries, rerun_status, selected, moved)
    print(json.dumps({
        "archived_reruns": len(archived),
        "valid_reruns": sum(row["integrity_status"] == "valid" for row in rerun_status),
        "invalid_reruns": sum(row["integrity_status"] != "valid" for row in rerun_status),
        "selected": selected["selected_experiment_id"] if selected else None,
        "heldout_retrieval_run": False,
        "generation_evaluation_run": False,
        "groq_api_key_available": groq_key_available(),
    }, indent=2))


if __name__ == "__main__":
    main()
