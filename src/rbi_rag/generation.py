from __future__ import annotations

import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .config import RAGConfig


PROMPT = ChatPromptTemplate.from_template(
    "Use ONLY the supplied RBI report context to answer. "
    "If the answer is not supported by the context, say: "
    "'I could not find this in the supplied RBI report.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


class AnswerGenerator:
    def __init__(self, config: RAGConfig):
        self.llm = ChatGroq(model=config.generator_model, temperature=config.temperature)

    def answer(self, question: str, documents) -> dict[str, object]:
        started = time.perf_counter()
        context = "\n\n".join(document.page_content for document in documents)
        response = self.llm.invoke(PROMPT.format_messages(context=context, question=question))
        citations = [
            {
                "source": document.metadata.get("source_file", document.metadata.get("source")),
                "page": document.metadata.get(
                    "page_number", int(document.metadata.get("page", -1)) + 1
                ),
                "chunk_index": document.metadata.get("chunk_index"),
            }
            for document in documents
        ]
        return {
            "answer": response.content, "citations": citations, "context": context,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
