from rbi_rag.evaluation.retrieval_metrics import score_retrieval
from rbi_rag.schemas import EvaluationQuestion


class Doc:
    def __init__(self, page, chunk):
        self.metadata = {"page": page, "page_number": page, "chunk_id": chunk}


def test_hit_rate_mrr_and_multiple_accepted_pages():
    item = EvaluationQuestion("q1", "Q", "A", (2, 3))
    result = score_retrieval("fake", lambda _: [(Doc(8, "a"), .9), (Doc(3, "b"), .8)], [item])
    assert result["hit_rate_at_k"] == 1.0
    assert result["mrr"] == 0.5
    assert result["details"][0]["rank_first_accepted_page"] == 2
    assert result["details"][0]["retrieved_chunk_ids"] == ["a", "b"]

