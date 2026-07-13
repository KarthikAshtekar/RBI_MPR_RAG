from pathlib import Path
import pytest
from rbi_rag.config import RAGConfig


def test_config_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        RAGConfig(chunk_size=100, chunk_overlap=100).validate()


def test_config_rejects_candidate_pool_smaller_than_final_k():
    with pytest.raises(ValueError):
        RAGConfig(reranker_candidate_k=3, final_k=4).validate()


def test_yaml_configuration_loads_and_has_no_api_key():
    config = RAGConfig.from_yaml(Path("configs/baseline.yaml"))
    assert config.dense_k == config.bm25_k == config.reranker_candidate_k == 15
    assert config.final_k == 4
    assert "api_key" not in config.public_dict()

