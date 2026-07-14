from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbi_rag.env_loading import load_project_dotenv
from rbi_rag.poppler_setup import POPPLER_RETRY_OUT
from rbi_rag.v2_experiments import CONTROLLED_EXPERIMENTS, V2_OUT, validate_v2_raw_rows, write_json

load_project_dotenv(ROOT)


def main() -> int:
    out = ROOT / V2_OUT
    issues: list[str] = []
    unstructured_issues: list[str] = []
    unstructured_blockers: list[str] = []
    checks = []
    for experiment_id in CONTROLLED_EXPERIMENTS:
        path = out / "experiments" / experiment_id
        if not path.exists():
            issues.append(f"{experiment_id}:missing_experiment_directory")
            continue
        integrity_path = path / "integrity.json"
        raw_path = path / "raw_results.json"
        if not integrity_path.exists():
            issues.append(f"{experiment_id}:missing_integrity_json")
            continue
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
        if integrity.get("status") == "valid":
            row_issues = validate_v2_raw_rows(raw)
            issues.extend(f"{experiment_id}:{issue}" for issue in row_issues)
            if experiment_id in {"V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"}:
                parser_issues = [
                    f"{experiment_id}:{row.get('question_id', 'unknown')}:parser_name_not_unstructured"
                    for row in raw
                    if str(row.get("parser_name")).lower() != "unstructured"
                ]
                missing_pages = [
                    f"{experiment_id}:{row.get('question_id', 'unknown')}:missing_selected_pages"
                    for row in raw
                    if not row.get("selected_pages")
                ]
                unstructured_issues.extend(parser_issues)
                unstructured_issues.extend(missing_pages)
        elif experiment_id in {"V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"}:
            unstructured_blockers.append(f"{experiment_id}:{integrity.get('status')}:{integrity.get('reason')}")
        checks.append({
            "experiment_id": experiment_id,
            "status": integrity.get("status"),
            "issue_count": integrity.get("issue_count", 0),
            "row_count": len(raw),
        })
    for required in [
        "v2_experiment_leaderboard.json",
        "v2_category_results.json",
        "v2_paired_comparisons.json",
        "v2_selected_retrieval.json",
        "v2_generation_readiness.json",
        "v2_results_for_presentation.md",
        "v2_unstructured_cohere_report.md",
    ]:
        if not (out / required).exists():
            issues.append(f"missing_required_artifact:{required}")
    payload = {
        "schema_version": 1,
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": sorted(set(issues)),
        "checks": checks,
    }
    write_json(out / "v2_artifact_validation.json", payload)
    lines = ["# V2 Artifact Validation", "", f"Status: {payload['status']}", f"Issues: {payload['issue_count']}", ""]
    if payload["issues"]:
        lines += ["## Issues", ""]
        lines.extend(f"- {issue}" for issue in payload["issues"])
    (out / "v2_artifact_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    retry_out = ROOT / POPPLER_RETRY_OUT
    retry_out.mkdir(parents=True, exist_ok=True)
    unstructured_payload = {
        "schema_version": 1,
        "status": "passed" if not unstructured_issues and not unstructured_blockers else ("blocked" if unstructured_blockers and not unstructured_issues else "failed"),
        "issue_count": len(set(unstructured_issues)),
        "issues": sorted(set(unstructured_issues)),
        "blockers": sorted(set(unstructured_blockers)),
        "checks": [check for check in checks if check["experiment_id"] in {"V2_UNSTRUCTURED_ONLY", "V2_UNSTRUCTURED_COHERE"}],
    }
    write_json(retry_out / "unstructured_experiment_validation.json", unstructured_payload)
    u_lines = [
        "# Unstructured Experiment Validation",
        "",
        f"Status: {unstructured_payload['status']}",
        f"Issues: {unstructured_payload['issue_count']}",
        "",
    ]
    if unstructured_payload["issues"]:
        u_lines += ["## Issues", ""]
        u_lines.extend(f"- {issue}" for issue in unstructured_payload["issues"])
    if unstructured_payload["blockers"]:
        u_lines += ["", "## Blockers", ""]
        u_lines.extend(f"- {issue}" for issue in unstructured_payload["blockers"])
    (retry_out / "unstructured_experiment_validation.md").write_text("\n".join(u_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues and not unstructured_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
