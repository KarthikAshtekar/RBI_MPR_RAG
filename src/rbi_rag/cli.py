from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

from .config import RAGConfig


def parser():
    root = argparse.ArgumentParser(prog="rbi-rag")
    root.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-reports")
    commands.add_parser("build-index")
    commands.add_parser("inspect-index")
    commands.add_parser("audit-extraction")
    retrieval_eval = commands.add_parser("retrieval-eval")
    retrieval_eval.add_argument("--split", choices=("dev", "test"), default="dev")
    router_eval = commands.add_parser("router-eval")
    router_eval.add_argument("--split", choices=("dev", "test"), default="dev")
    generation = commands.add_parser("generation-eval")
    generation.add_argument("--rerun-successful", action="store_true")
    generation.add_argument("--split", choices=("dev", "test"), default="dev")
    commands.add_parser("compare-architectures")
    ask = commands.add_parser("ask"); ask.add_argument("question")
    route = commands.add_parser("route-query"); route.add_argument("--query", required=True)
    retrieve = commands.add_parser("retrieve"); retrieve.add_argument("--query", required=True)
    report = commands.add_parser("report"); report.add_argument("--output", type=Path)
    return root


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    args = parser().parse_args(argv)
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if "reports_registry" in raw_config:
        from .multi_cli import run_multi_command
        return run_multi_command(args)
    config = RAGConfig.from_yaml(args.config)
    if args.command == "report":
        from .evaluation import generate_markdown_report
        retrieval = json.loads((config.output_directory / "retrieval_raw_results.json").read_text(encoding="utf-8"))
        generation_path = config.output_directory / "generation_evaluation.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.exists() else None
        output = args.output or config.output_directory / "baseline_report.md"
        generate_markdown_report(retrieval, generation, args.config, output)
        print(output); return 0
    if args.command == "build-index":
        from .indexing import build_or_open_index
        from .ingestion import load_and_chunk_pdf
        _, chunks = load_and_chunk_pdf(config)
        build_or_open_index(chunks, config)
        print(json.dumps({"index": str(config.chroma_path), "report_id": config.report_id}))
        return 0
    from .pipeline import build_retrieval_suite
    suite = build_retrieval_suite(config)
    from .evaluation import load_evaluation_items
    if args.command == "retrieval-eval":
        from .evaluation import run_retrieval_baseline, write_retrieval_outputs
        payload = run_retrieval_baseline(suite, load_evaluation_items(config.evaluation_path), config)
        write_retrieval_outputs(payload, config.output_directory)
        print(json.dumps({"saved": str(config.output_directory), "strategies": [{k: row[k] for k in ("strategy", "hit_rate_at_k", "mrr")} for row in payload["strategies"]]}, indent=2))
        return 0
    if args.command == "generation-eval":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for generation evaluation")
        from .deepeval_metrics import build_metric_factories
        from .evaluation import run_generation_evaluation
        from .generation import AnswerGenerator
        output = config.output_directory / "generation_evaluation.json"
        payload = run_generation_evaluation(
            items=load_evaluation_items(config.evaluation_path), retrieve=suite.hybrid_reranked,
            generate=AnswerGenerator(config).answer, metric_factories=build_metric_factories(config),
            checkpoint_path=output, config=config, rerun_successful=args.rerun_successful,
        )
        print(json.dumps({"saved": str(output), "summary": payload["summary"]}, indent=2)); return 0
    if args.command == "ask":
        from .generation import AnswerGenerator
        retrieved = suite.hybrid_reranked(args.question)
        docs = [value[0] for value in retrieved]
        print(json.dumps(AnswerGenerator(config).answer(args.question, docs), indent=2, ensure_ascii=False)); return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
