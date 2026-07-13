from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from langchain_groq import ChatGroq

from .comparative_generation import ComparativeGenerator
from .evaluation.reporting import atomic_write_json
from .evaluation.retry import measure_with_retry
from .multi_evaluation import load_jsonl


def build_temporal_metric_factories(model_name: str):
    from deepeval.metrics import GEval
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCaseParams

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self): self.model = ChatGroq(model=model_name, temperature=0)
        def load_model(self): return self.model
        def get_model_name(self): return f"Groq-{model_name}"
        def generate(self, prompt: str, schema=None):
            text = self.model.invoke(prompt).content
            return schema.model_validate_json(text) if schema else text
        async def a_generate(self, prompt: str, schema=None):
            text = (await self.model.ainvoke(prompt)).content
            return schema.model_validate_json(text) if schema else text

    params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
    return {
        "temporal_attribution_correctness": lambda: GEval(
            name="Temporal attribution correctness",
            criteria="Facts and figures must be attributed to the correct report period.",
            evaluation_params=params, model=GroqJudge(), threshold=.5),
        "comparative_correctness": lambda: GEval(
            name="Comparative correctness",
            criteria="The comparison must be factually consistent with the expected answer and supplied periods.",
            evaluation_params=params, model=GroqJudge(), threshold=.5),
    }


def run_multi_generation_evaluation(config, registry, router, retriever, checkpoint: Path):
    from deepeval.test_case import LLMTestCase
    generator = ComparativeGenerator(config.generator_model, config.temperature, registry)
    factories = build_temporal_metric_factories(config.generator_model)
    saved = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {"rows": []}
    rows = {row["question_id"]: row for row in saved.get("rows", [])}
    for case in load_jsonl(config.dev_cases):
        row = rows.get(case["question_id"])
        if row is None:
            plan = router.route(case["question"])
            retrieval = retriever.retrieve_from_query_plan(plan)
            output = generator.generate(case["question"], plan, retrieval)
            cited_ids = {value["chunk_id"] for value in output["citations"]}
            context_ids = {chunk.metadata["chunk_id"] for chunk in retrieval["final_selected_chunks"]}
            required = set(case["required_report_ids"])
            used = set(output["reports_used"])
            row = {
                "question_id": case["question_id"], "question": case["question"],
                "expected_answer": case["expected_answer"], "generated_answer": output["answer"],
                "query_plan": output["query_plan"], "citations": output["citations"],
                "reports_used": output["reports_used"], "warnings": output["warnings"],
                "citation_correctness": cited_ids <= context_ids,
                "citation_completeness": required <= used,
                "report_coverage": len(required & used) / len(required),
                "abstention_status": "abstained" if "could not find" in output["answer"].lower() else "answered",
                "metrics": {},
            }
            rows[case["question_id"]] = row
        test_case = LLMTestCase(input=case["question"], actual_output=row["generated_answer"],
                                expected_output=case["expected_answer"])
        for name, factory in factories.items():
            if row["metrics"].get(name, {}).get("success"):
                continue
            row["metrics"][name] = asdict(measure_with_retry(factory, test_case))
            atomic_write_json(checkpoint, {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                           "rows": list(rows.values())})
    return {"rows": list(rows.values())}

