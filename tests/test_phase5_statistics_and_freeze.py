from hashlib import sha256
import json
from pathlib import Path

from rbi_rag.uncertainty import bootstrap_mean_interval, wilson_interval


def test_wilson_interval_bounds_binary_proportion():
    low, high = wilson_interval(8, 10)
    assert 0 <= low < .8 < high <= 1


def test_bootstrap_interval_is_deterministic_for_seed():
    first = bootstrap_mean_interval([0, .5, 1], resamples=200, seed=42)
    second = bootstrap_mean_interval([0, .5, 1], resamples=200, seed=42)
    assert first == second


def test_temporal_dataset_manifest_checksums_and_test_freeze():
    manifest = json.loads(Path("data/evaluation/temporal_dataset_manifest.json").read_text())
    for info in manifest["files"].values():
        assert sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
    assert manifest["verified_scored_counts"]["test"] == 15


def test_complete_corpus_manifest_has_three_validated_reports():
    manifest = json.loads(Path("reports/multi_report/full_corpus_manifest.json").read_text())
    assert len(manifest["reports"]) == 3
    assert all(row["validation_status"] == "validated" for row in manifest["reports"].values())
    assert manifest["duplicate_chunk_ids"] == 0
