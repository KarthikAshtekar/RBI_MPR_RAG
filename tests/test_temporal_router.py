import json
from pathlib import Path
from rbi_rag.report_registry import ReportRegistry
from rbi_rag.temporal_router import TemporalQueryRouter


def router():
    return TemporalQueryRouter(ReportRegistry.from_yaml(Path("configs/reports.yaml")))


def test_explicit_single_pairwise_trend_latest_and_unsupported_routing():
    r = router()
    assert r.route("Inflation in April 2025?").query_type == "single_report"
    assert r.route("Compare April and October 2025.").query_type == "pairwise_comparison"
    assert r.route("How did inflation evolve across reports?").query_type == "trend_all_reports"
    assert r.route("What does the latest report say?").report_ids == ("rbi_mpr_2026_04",)
    assert r.route("Compare April 2024 and April 2025.").query_type == "unsupported_period"


def test_all_30_router_cases_match():
    r = router(); cases = [json.loads(line) for line in Path("data/evaluation/router_cases.jsonl").read_text().splitlines()]
    failures = []
    for case in cases:
        plan = r.route(case["query"])
        if plan.query_type != case["expected_query_type"] or list(plan.report_ids) != case["expected_report_ids"]:
            failures.append((case["case_id"], plan.query_type, list(plan.report_ids)))
    assert not failures

