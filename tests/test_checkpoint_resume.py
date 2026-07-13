import json
from pathlib import Path
from rbi_rag.config import RAGConfig, file_sha256
from rbi_rag.evaluation.generation_metrics import run_generation_evaluation
from rbi_rag.schemas import EvaluationQuestion


class Doc:
    page_content = "context"
    metadata = {"chunk_id": "c1", "page_number": 1}


def test_resume_reruns_only_failed_metrics(tmp_path):
    pdf = tmp_path / "report.pdf"; pdf.write_bytes(b"pdf")
    checkpoint = tmp_path / "checkpoint.json"
    config = RAGConfig(pdf_path=pdf, retry_base_delay_seconds=0)
    calls = {"success": 0, "failed": 0}
    class Metric:
        def __init__(self, name): self.name = name
        def measure(self, _):
            calls[self.name] += 1
            self.score = .8
    factories = {name: (lambda name=name: Metric(name)) for name in calls}
    item = EvaluationQuestion("q1", "Q", "A", (1,))
    run_generation_evaluation(
        items=[item], retrieve=lambda _: [(Doc(), 1.0)],
        generate=lambda q, d: {"answer": "A", "latency_ms": 1},
        metric_factories=factories, checkpoint_path=checkpoint, config=config,
    )
    saved = json.loads(checkpoint.read_text())
    saved["rows"][0]["metrics"]["failed"].update({"success": False, "score": None, "status": "failed"})
    checkpoint.write_text(json.dumps(saved))
    run_generation_evaluation(
        items=[item], retrieve=lambda _: (_ for _ in ()).throw(AssertionError("must reuse generation")),
        generate=lambda q, d: {}, metric_factories=factories,
        checkpoint_path=checkpoint, config=config,
    )
    assert calls == {"success": 1, "failed": 2}

