from __future__ import annotations

from langchain_groq import ChatGroq

from .config import RAGConfig


def build_metric_factories(config: RAGConfig):
    """Construct fresh DeepEval metrics for every question and retry."""
    from deepeval.metrics import GEval, ContextualRecallMetric, ContextualRelevancyMetric
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCaseParams

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self):
            self.model = ChatGroq(model=config.judge_model, temperature=0)

        def load_model(self):
            return self.model

        def generate(self, prompt: str, schema=None):
            response = self.model.invoke(prompt).content
            if schema is not None:
                return schema.model_validate_json(response)
            return response

        async def a_generate(self, prompt: str, schema=None):
            response = (await self.model.ainvoke(prompt)).content
            if schema is not None:
                return schema.model_validate_json(response)
            return response

        def get_model_name(self):
            return f"Groq-{config.judge_model}"

    def judge():
        return GroqJudge()

    return {
        "correctness": lambda: GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually consistent with the expected output.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge(),
            threshold=0.5,
        ),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(
            model=judge(), threshold=0.5
        ),
        "contextual_recall": lambda: ContextualRecallMetric(model=judge(), threshold=0.5),
    }
