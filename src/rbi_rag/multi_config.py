from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class MultiReportConfig:
    config_path: Path
    reports_registry: Path
    collection_name: str
    persistence_path: Path
    manifest_path: Path
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    ingestion_version: str
    dense_k: int
    bm25_k: int
    rrf_k: int
    reranker_k: int
    final_single: int
    final_comparative: int
    final_trend: int
    generator_model: str
    temperature: float
    prompt_version: str
    router_cases: Path
    dev_cases: Path
    test_cases: Path
    output_directory: Path
    router_test_cases: Path
    bootstrap_resamples: int
    confidence_level: float
    random_seed: int
    enable_naive_global_baseline: bool

    @classmethod
    def from_yaml(cls, path: Path) -> "MultiReportConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        index, chunk, retrieval = raw["index"], raw["chunking"], raw["retrieval"]
        generation, evaluation = raw["generation"], raw["evaluation"]
        config = cls(
            path, Path(raw["reports_registry"]), index["collection_name"],
            Path(index["persistence_path"]), Path(index["manifest_path"]),
            raw["embedding_model"], raw["reranker_model"], int(chunk["chunk_size"]),
            int(chunk["chunk_overlap"]), chunk["ingestion_version"],
            int(retrieval["dense_k_per_report"]), int(retrieval["bm25_k_per_report"]),
            int(retrieval["fusion_rrf_k"]), int(retrieval["reranker_candidate_k_per_report"]),
            int(retrieval["final_k_single_report"]),
            int(retrieval["final_k_per_report_comparative"]),
            int(retrieval["final_k_per_report_trend"]), generation["model"],
            float(generation["temperature"]), generation["prompt_version"],
            Path(evaluation["router_cases"]), Path(evaluation["dev_cases"]),
            Path(evaluation["test_cases"]), Path(evaluation["output_directory"]),
            Path(evaluation.get("router_test_cases", "data/evaluation/router_test.jsonl")),
            int(evaluation.get("bootstrap_resamples", 2000)),
            float(evaluation.get("confidence_level", .95)),
            int(evaluation.get("random_seed", 42)),
            bool(raw.get("comparison", {}).get("enable_naive_global_baseline", True)),
        )
        config.validate(); return config

    def validate(self):
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("invalid chunking settings")
        values = (self.dense_k, self.bm25_k, self.rrf_k, self.reranker_k,
                  self.final_single, self.final_comparative, self.final_trend)
        if min(values) <= 0 or min(self.dense_k, self.bm25_k) < self.reranker_k:
            raise ValueError("invalid retrieval settings")
        if self.bootstrap_resamples <= 0 or not 0 < self.confidence_level < 1:
            raise ValueError("invalid uncertainty settings")
