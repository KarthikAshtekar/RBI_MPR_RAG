def dense_search(vector_store, query: str, k: int):
    return vector_store.similarity_search_with_relevance_scores(query, k=k)

