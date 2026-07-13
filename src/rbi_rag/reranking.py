def rerank(cross_encoder, query: str, candidates, final_k: int):
    if not candidates:
        return []
    documents = [item[0] if isinstance(item, tuple) else item for item in candidates]
    pairs = [[query, document.page_content] for document in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(
        zip(documents, scores),
        key=lambda item: (-float(item[1]), str(item[0].metadata["chunk_id"])),
    )
    return [(document, float(score)) for document, score in ranked[:final_k]]

