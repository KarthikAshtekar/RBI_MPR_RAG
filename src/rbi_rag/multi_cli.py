from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from .multi_config import MultiReportConfig
from .multi_evaluation import (evaluate_router, generate_multi_report,
                               run_multi_retrieval_evaluation, write_router_results)
from .multi_index import build_multi_report_index, inspect_shared_index
from .multi_retrieval import MultiReportRetriever
from .report_registry import ReportRegistry
from .temporal_router import TemporalQueryRouter


def _load(config_path: Path):
    config = MultiReportConfig.from_yaml(config_path)
    registry = ReportRegistry.from_yaml(config.reports_registry)
    return config, registry


def run_multi_command(args):
    config, registry = _load(args.config)
    if args.command == "validate-reports":
        result = {"registry_version": registry.version,
                  "available": [r.report_id for r in registry.available()],
                  "missing_paths": registry.missing_paths()}
        print(json.dumps(result, indent=2)); return 0
    router = TemporalQueryRouter(registry)
    if args.command == "route-query":
        print(json.dumps(asdict(router.route(args.query)), indent=2)); return 0
    if args.command == "audit-extraction":
        from .extraction_audit import audit_reports
        payload = audit_reports(
            registry, config.output_directory / "extraction_audit.json",
            config.output_directory / "extraction_audit.md",
        )
        print(json.dumps({"audited_pages": len(payload["records"]),
                          "numeric_exclusions": len(payload["numeric_evaluation_exclusions"])}, indent=2))
        return 0
    if args.command == "router-eval":
        split = args.split
        router_path = config.router_cases if split == "dev" else config.router_test_cases
        rows, accuracy = evaluate_router(router, router_path)
        write_router_results(rows, accuracy, config.output_directory, split)
        print(json.dumps({"split": split, "case_count": len(rows), "accuracy": accuracy}, indent=2)); return 0
    if args.command == "report":
        if not config.manifest_path.exists():
            raise FileNotFoundError("Build the multi-report index before generating its report")
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        router_rows, accuracy = evaluate_router(router, config.router_cases)
        write_router_results(router_rows, accuracy, config.output_directory, "dev")
        retrieval_path = config.output_directory / "retrieval_raw_results.json"
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8")) if retrieval_path.exists() else {"summary": {}}
        from .multi_evaluation import generate_full_temporal_report
        generate_full_temporal_report(config.output_directory, registry, manifest)
        print(config.output_directory / "full_temporal_retrieval_report.md"); return 0
    store, chunks_by_report, manifest = build_multi_report_index(config, registry)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    (config.output_directory / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (config.output_directory / "full_corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.command == "build-index":
        print(json.dumps(manifest, indent=2)); return 0
    if args.command == "inspect-index":
        print(json.dumps(inspect_shared_index(store, registry), indent=2)); return 0
    router_path = config.router_cases if getattr(args, "split", "dev") == "dev" else config.router_test_cases
    router_rows, accuracy = evaluate_router(router, router_path)
    write_router_results(router_rows, accuracy, config.output_directory, getattr(args, "split", "dev"))
    retriever = MultiReportRetriever(store, chunks_by_report, registry, config)
    if args.command == "retrieve":
        plan = router.route(args.query); result = retriever.retrieve_from_query_plan(plan)
        printable = {key: value for key, value in result.items() if key != "final_selected_chunks"}
        printable["final_selected_chunks"] = [
            {"chunk_id": d.metadata["chunk_id"], "report_id": d.metadata["report_id"],
             "page": d.metadata["page"], "excerpt": d.page_content[:300]}
            for d in result["final_selected_chunks"]
        ]
        print(json.dumps(printable, indent=2)); return 0
    if args.command == "retrieval-eval":
        cases_path = config.dev_cases if args.split == "dev" else config.test_cases
        payload = run_multi_retrieval_evaluation(
            router, retriever, cases_path, config.output_directory, split=args.split,
            resamples=config.bootstrap_resamples, confidence=config.confidence_level,
            seed=config.random_seed,
        )
        if args.split == "dev":
            from .failure_analysis import analyze_dev_failures
            analyze_dev_failures(payload, config.output_directory)
        print(json.dumps(payload["summary"], indent=2)); return 0
    if args.command == "compare-architectures":
        from .architecture_comparison import compare_architectures, evaluate_naive
        from .naive_global import NaiveGlobalRetriever
        naive = NaiveGlobalRetriever(store, chunks_by_report, registry, config, retriever.cross_encoder)
        payloads = {
            "dev": evaluate_naive(router, naive, config.dev_cases, config, "dev"),
            "test": evaluate_naive(router, naive, config.test_cases, config, "test"),
        }
        comparison = compare_architectures(config, payloads)
        print(json.dumps(comparison, indent=2)); return 0
    if args.command == "generation-eval":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for generation evaluation")
        from .multi_generation_evaluation import run_multi_generation_evaluation
        payload = run_multi_generation_evaluation(
            config, registry, router, retriever,
            config.output_directory / "generation_evaluation.json",
        )
        print(json.dumps({"saved_cases": len(payload["rows"])}, indent=2)); return 0
    raise AssertionError(f"unhandled multi-report command: {args.command}")
