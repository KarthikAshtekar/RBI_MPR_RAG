from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbi_rag.multi_config import MultiReportConfig
from rbi_rag.multi_evaluation import load_jsonl
from rbi_rag.multi_index import build_multi_report_index
from rbi_rag.report_bm25 import BM25ByReport
from rbi_rag.report_registry import ReportRegistry
from rbi_rag.stage_a_runner import groq_key_available, run_question, warm_up
from rbi_rag.temporal_router import TemporalQueryRouter

from scripts.run_phase6b_structural_experiments import (
    OUT,
    apply_structural_transform,
    file_sha,
    flatten_for_stage_a,
    stable_json_hash,
    summarise,
    validate_experiment,
    write_experiment,
)


def parser():
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, default=Path("configs/final_retrieval_selected.yaml"))
    root.add_argument("--confirm-one-time-heldout", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.confirm_one_time_heldout:
        raise SystemExit("Refusing to run held-out retrieval without --confirm-one-time-heldout")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["id"] = "heldout_" + config["id"]
    cfg = MultiReportConfig.from_yaml(Path("configs/multi_report.yaml"))
    registry = ReportRegistry.from_yaml(cfg.reports_registry)
    cases = [case for case in load_jsonl(cfg.test_cases) if case.get("verification_status") == "verified"]
    store, chunks_by_report, manifest = build_multi_report_index(cfg, registry)
    resources = {
        "store": store,
        "bm25": BM25ByReport(chunks_by_report),
        "cross_encoder": CrossEncoder(cfg.reranker_model),
        "router": TemporalQueryRouter(registry),
        "registry": registry,
        "dataset_checksum": file_sha(cfg.test_cases),
        "index_fingerprint": stable_json_hash(manifest),
    }
    warmup = warm_up(resources)
    resources["configuration_checksum"] = stable_json_hash(config)
    flat = flatten_for_stage_a(config)
    rows = []
    report_rows = []
    started = time.time()
    for case in cases:
        row, _ = run_question(case, flat, resources)
        row["split"] = "test"
        row["configuration_checksum"] = stable_json_hash(config)
        transformed, per_report = apply_structural_transform(row, config, chunks_by_report)
        transformed["split"] = "test"
        rows.append(transformed)
        report_rows.extend(per_report)
    output_dir = OUT / "heldout_one_time" / config["id"]
    summary = summarise(config, rows, report_rows, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), file_sha(cfg.test_cases), stable_json_hash(manifest))
    environment = {
        "phase": "6B_heldout_one_time",
        "heldout_dataset_loaded": True,
        "generation_evaluation_run": False,
        "groq_api_key_available": groq_key_available(),
        **warmup,
    }
    write_experiment(output_dir, config, environment, manifest, rows, report_rows, summary)
    print(json.dumps({"output": str(output_dir), "cases": len(rows), "generation_evaluation_run": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
