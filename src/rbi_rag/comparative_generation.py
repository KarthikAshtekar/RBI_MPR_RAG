from __future__ import annotations

from dataclasses import asdict
from langchain_groq import ChatGroq

from .report_registry import ReportRegistry
from .schemas import Citation, QueryPlan

COMPARATIVE_PROMPT_VERSION = "comparative_v1"
COMPARATIVE_PROMPT = """Use only the source-labelled RBI report context below.
Distinguish every report period and never transfer a figure from one report to another.
Cite the report period and page for factual claims. State explicitly when evidence is unavailable.
Separate report facts from your comparison. Do not infer generic sentiment.
Do not perform hidden arithmetic; label any arithmetic as calculated from cited values.

Structure the answer as:
Report-wise findings
[one section per supplied report]
Evolution over time
Interpretation

Context:
{context}

Question: {question}
Answer:"""


def format_source_context(chunks, registry: ReportRegistry) -> str:
    by_id = registry.by_id()
    ordered = sorted(chunks, key=lambda chunk: (
        by_id[chunk.metadata["report_id"]].report_date,
        chunk.metadata["page"], chunk.metadata["chunk_id"],
    ))
    blocks = []
    for chunk in ordered:
        meta = chunk.metadata
        blocks.append(
            f'<SOURCE\n report_id="{meta["report_id"]}"\n '
            f'report_period="{meta["report_period"]}"\n page="{meta["page"]}"\n '
            f'chunk_id="{meta["chunk_id"]}"\n>\n{chunk.page_content}\n</SOURCE>'
        )
    return "\n\n".join(blocks)


def citations_from_context(chunks) -> list[Citation]:
    return [Citation(
        chunk.metadata["report_id"], chunk.metadata["report_period"],
        int(chunk.metadata["page"]), chunk.metadata["chunk_id"],
        chunk.page_content[:300],
    ) for chunk in chunks]


def validate_citations(citations: list[Citation], chunks) -> bool:
    allowed = {chunk.metadata["chunk_id"] for chunk in chunks}
    return all(citation.chunk_id in allowed for citation in citations)


class ComparativeGenerator:
    def __init__(self, model: str, temperature: float, registry: ReportRegistry):
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.registry = registry

    def generate(self, question: str, plan: QueryPlan, retrieval_result: dict):
        chunks = retrieval_result["final_selected_chunks"]
        context = format_source_context(chunks, self.registry)
        response = self.llm.invoke(COMPARATIVE_PROMPT.format(context=context, question=question))
        citations = citations_from_context(chunks)
        if not validate_citations(citations, chunks):
            raise ValueError("citation validation failed")
        warnings = list(retrieval_result["missing_report_warnings"])
        if plan.requires_calculation:
            warnings.append("Any arithmetic must be explicitly calculated from cited values.")
        return {
            "answer": response.content, "citations": [asdict(c) for c in citations],
            "query_plan": asdict(plan),
            "reports_used": sorted({c.report_id for c in citations}), "warnings": warnings,
            "prompt_version": COMPARATIVE_PROMPT_VERSION,
        }

