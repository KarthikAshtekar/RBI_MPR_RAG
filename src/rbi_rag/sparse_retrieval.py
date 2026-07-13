def bm25_search(retriever, query: str, k: int):
    retriever.k = k
    return [(document, None) for document in retriever.invoke(query)]

