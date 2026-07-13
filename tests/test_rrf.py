from rbi_rag.fusion import reciprocal_rank_fusion


class Doc:
    def __init__(self, chunk_id):
        self.metadata = {"chunk_id": chunk_id}


def test_rrf_ordering_rewards_consensus():
    a, b, c = Doc("a"), Doc("b"), Doc("c")
    result = reciprocal_rank_fusion([[(a, 1), (b, 0)], [(c, 1), (a, 0)]], rrf_k=60, limit=3)
    assert [document.metadata["chunk_id"] for document, _ in result] == ["a", "c", "b"]


def test_rrf_ties_are_deterministic_by_first_seen_then_chunk_id():
    a, b = Doc("a"), Doc("b")
    first = reciprocal_rank_fusion([[(b, None)], [(a, None)]], rrf_k=60, limit=2)
    second = reciprocal_rank_fusion([[(b, None)], [(a, None)]], rrf_k=60, limit=2)
    assert [d.metadata["chunk_id"] for d, _ in first] == ["b", "a"]
    assert [d.metadata["chunk_id"] for d, _ in first] == [d.metadata["chunk_id"] for d, _ in second]

