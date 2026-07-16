from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from .experiment_tracing import (
    StageTimer,
    context_statistics,
    first_evidence_rank,
    recompute_loss_stage,
)
from .fusion import reciprocal_rank_fusion
from .query_optimisation import decompose_facets, expand_query, normalise_retrieval_query


ROOT = Path(".")


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
            del rank
            cid = doc.metadata["chunk_id"]
            if cid not in combined:
                combined[cid] = [doc, score, {source_name}, sequence]
                order[cid] = sequence
                sequence += 1
            else:
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

    with timer.measure("dense_latency_ms"):
        for rid in required:
            values = []
            for q in query_plan["retrieval_queries"]:
                hits = resources["store"].similarity_search_with_relevance_scores(
                    q, k=int(config["dk"]), filter={"report_id": rid}
                )
                for doc, score in hits:
                    del score
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
