from rbi_rag.evaluation.retry import measure_with_retry


def test_failed_metric_score_is_none_and_stale_score_is_never_reused():
    class Broken:
        score = .99
        def measure(self, _): raise RuntimeError("429 rate limit")
    result = measure_with_retry(Broken, object(), max_attempts=2, base_delay_seconds=0, sleep=lambda _: None)
    assert not result.success
    assert result.score is None
    assert result.attempts == 2
    assert result.error_type == "RuntimeError"

