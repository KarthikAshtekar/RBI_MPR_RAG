from rbi_rag.evaluation.retry import measure_with_retry


def test_retry_attempt_count_and_fresh_instances():
    calls = 0
    instances = []
    class Metric:
        def __init__(self): instances.append(self)
        def measure(self, _):
            nonlocal calls
            calls += 1
            if calls < 3: raise ValueError("invalid JSON")
            self.score = .75; self.reason = "ok"
    result = measure_with_retry(Metric, object(), max_attempts=3, base_delay_seconds=0, sleep=lambda _: None)
    assert result.success and result.attempts == 3 and result.score == .75
    assert len(instances) == 3

