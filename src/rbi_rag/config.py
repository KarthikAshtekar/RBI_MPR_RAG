from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RAGConfig:
    report_id: str = "rbi_mpr_2025_04"
    report_period: str = "April 2025"
    report_date: str = "2025-04-01"
    pdf_path: Path = Path("mpr_april_2025.pdf")
    chroma_path: Path = Path("indexes/chroma")
    evaluation_path: Path = Path("data/evaluation/questions_v1.json")
    output_directory: Path = Path("reports/current")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 1000
    chunk_overlap: int = 300
    dense_k: int = 15
    bm25_k: int = 15
    fusion_rrf_k: int = 60
    reranker_candidate_k: int = 15
    final_k: int = 4
    max_retries: int = 5
    retry_base_delay_seconds: float = 20.0
    retry_timeout_seconds: float = 120.0
    prompt_version: str = "grounded_v1"
    dataset_version: str = "rbi_mpr_april_2025_dev_v1"
    ingestion_version: str = "v1"
    temperature: float = 0.0

    def validate(self) -> None:
        if not self.report_id.strip():
            raise ValueError("report_id is required")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        retrieval_values = (
            self.dense_k, self.bm25_k, self.fusion_rrf_k,
            self.reranker_candidate_k, self.final_k,
        )
        if min(retrieval_values) <= 0:
            raise ValueError("retrieval sizes and RRF constant must be positive")
        if self.reranker_candidate_k < self.final_k:
            raise ValueError("reranker_candidate_k must be at least final_k")
        if min(self.dense_k, self.bm25_k) < self.reranker_candidate_k:
            raise ValueError("dense_k and bm25_k must cover the reranker candidate pool")
        if self.max_retries <= 0 or self.retry_base_delay_seconds < 0:
            raise ValueError("retry limits must be positive and delays non-negative")
        if not self.pdf_path.is_absolute() and ".." in self.pdf_path.parts:
            raise ValueError("pdf_path may not traverse parent directories")

    @classmethod
    def from_yaml(cls, path: Path) -> "RAGConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        for name in ("pdf_path", "chroma_path", "evaluation_path", "output_directory"):
            if name in raw:
                raw[name] = Path(os.path.expandvars(str(raw[name])))
        config = cls(**raw)
        config.validate()
        return config

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        return {key: value.as_posix() if isinstance(value, Path) else value for key, value in values.items()}

    def fingerprint(self, pdf_sha256: str | None = None) -> str:
        payload = self.public_dict() | {"pdf_sha256": pdf_sha256}
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

