from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists, *, rrf_k: int, limit: int):
    scores: dict[str, float] = {}
    documents: dict[str, object] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            document = item[0] if isinstance(item, tuple) else item
            key = str(document.metadata["chunk_id"])
            if key not in first_seen:
                first_seen[key] = sequence
                sequence += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            documents[key] = document
    keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key], key))
    return [(documents[key], scores[key]) for key in keys[:limit]]
