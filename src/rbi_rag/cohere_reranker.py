from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .env_loading import load_project_dotenv


@dataclass(frozen=True)
class CohereRerankConfig:
    model: str = "rerank-v3.5"
    top_n: int = 30
    max_retries: int = 3
    timeout_seconds: int = 60
    min_interval_seconds: float = 6.5
    fallback: str = "none"


@dataclass
class CohereRerankMetadata:
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-v3.5"
    reranker_latency_ms: float | None = None
    reranker_api_success: bool = False
    reranker_api_attempts: int = 0
    reranker_error_type: str | None = None
    reranker_error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CohereRerankError(RuntimeError):
    def __init__(self, metadata: CohereRerankMetadata):
        super().__init__(metadata.reranker_error_message or "Cohere rerank failed")
        self.metadata = metadata


def cohere_available() -> bool:
    return importlib.util.find_spec("cohere") is not None


def cohere_key_available() -> bool:
    load_project_dotenv(Path.cwd())
    return bool(os.getenv("COHERE_API_KEY"))


def _document_text(document: Any) -> str:
    return getattr(document, "page_content", str(document))


def _parse_results(response: Any) -> list[tuple[int, float]]:
    if isinstance(response, dict):
        results = response.get("results", [])
    else:
        results = getattr(response, "results", response)
    parsed: list[tuple[int, float]] = []
    for item in results:
        if isinstance(item, dict):
            index = item.get("index")
            score = item.get("relevance_score")
        else:
            index = getattr(item, "index")
            score = getattr(item, "relevance_score")
        parsed.append((int(index), float(score)))
    return parsed


class CohereReranker:
    def __init__(
        self,
        config: CohereRerankConfig | None = None,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        sleep=time.sleep,
    ):
        self.config = config or CohereRerankConfig()
        self._sleep = sleep
        self._client = client
        self._api_key = api_key
        self._last_call_started: float | None = None

    def _client_or_create(self):
        if self._client is not None:
            return self._client
        load_project_dotenv(Path.cwd())
        if not cohere_available():
            raise RuntimeError("cohere package is not installed")
        import cohere

        key = self._api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not available")
        try:
            self._client = cohere.ClientV2(
                api_key=key,
                timeout=float(self.config.timeout_seconds),
                max_retries=0,
            )
        except AttributeError:
            self._client = cohere.Client(api_key)
        return self._client

    def _safe_error_message(self, exc: Exception) -> str:
        message = str(exc)
        secrets = [value for value in (self._api_key, os.getenv("COHERE_API_KEY")) if value]
        for secret in secrets:
            message = message.replace(secret, "[redacted]")
        return message

    def _respect_rate_limit(self) -> None:
        if self.config.min_interval_seconds <= 0 or self._last_call_started is None:
            return
        elapsed = time.perf_counter() - self._last_call_started
        remaining = float(self.config.min_interval_seconds) - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def rerank(self, query: str, candidates: Iterable[Any]) -> tuple[list[tuple[Any, float]], CohereRerankMetadata]:
        candidates = list(candidates)
        metadata = CohereRerankMetadata(reranker_model=self.config.model)
        if not candidates:
            metadata.reranker_api_success = True
            metadata.reranker_latency_ms = 0.0
            return [], metadata
        documents = [_document_text(item[0] if isinstance(item, tuple) else item) for item in candidates]
        started = time.perf_counter()
        errors: list[str] = []
        for attempt in range(1, self.config.max_retries + 1):
            metadata.reranker_api_attempts = attempt
            try:
                client = self._client_or_create()
                self._respect_rate_limit()
                self._last_call_started = time.perf_counter()
                if hasattr(client, "rerank"):
                    response = client.rerank(
                        model=self.config.model,
                        query=query,
                        documents=documents,
                        top_n=min(self.config.top_n, len(documents)),
                    )
                else:
                    response = client.rerank(
                        model=self.config.model,
                        query=query,
                        documents=documents,
                        top_n=min(self.config.top_n, len(documents)),
                    )
                ranked = []
                for index, score in _parse_results(response):
                    source = candidates[index]
                    document = source[0] if isinstance(source, tuple) else source
                    ranked.append((document, score))
                metadata.reranker_api_success = True
                metadata.reranker_latency_ms = (time.perf_counter() - started) * 1000
                return ranked, metadata
            except Exception as exc:
                metadata.reranker_error_type = type(exc).__name__
                safe_message = self._safe_error_message(exc)
                errors.append(f"{metadata.reranker_error_type}: {safe_message}")
                if attempt >= self.config.max_retries:
                    metadata.reranker_latency_ms = (time.perf_counter() - started) * 1000
                    metadata.reranker_error_message = " | ".join(errors)
                    raise CohereRerankError(metadata)
                rate_limited = (
                    metadata.reranker_error_type == "TooManyRequestsError"
                    or "429" in safe_message
                    or "rate limit" in safe_message.lower()
                )
                self._sleep(65 if rate_limited else min(2 ** attempt, 10))
        raise AssertionError("unreachable")
