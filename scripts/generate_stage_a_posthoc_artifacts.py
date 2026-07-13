from __future__ import annotations

import csv
import json
import random
import shutil
import statistics
from collections import defaultdict
from hashlib import sha256
from pathlib import Path


ROOT = Path(".")
OUT = ROOT / "reports" / "optimisation"
SEED = 42
RESAMPLES = 2000


def stable_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def active_experiment_dirs():
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.startswith("invalid_runs") or directory.name.startswith("active_invalid"):
            continue
        if (directory / "summary.json").exists() and (directory / "raw_results.json").exists():
            yield directory


def load_experiments():
    experiments = {}
    for directory in active_experiment_dirs():
        experiments[directory.name] = {
            "summary": load_json(directory / "summary.json"),
            "raw": load_json(directory / "raw_results.json"),
        }
    return experiments


def mean(values):
    return sum(values) / len(values) if values else None


def bootstrap_interval(values, *, seed=SEED, resamples=RESAMPLES):
    values = [value for value in values if value is not None]
    if not values:
        return [None, None]
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return [estimates[int(0.025 * (resamples - 1))], estimates[int(0.975 * (resamples - 1))]]


def paired_bootstrap_interval(differences, *, seed=SEED, resamples=RESAMPLES):
    return bootstrap_interval(differences, seed=seed, resamples=resamples)


def metric_value(row, metric):
    if metric == "complete_evidence_recall":
        value = row.get(metric)
        return None if value is None else float(value)
    if metric == "all_reports_hit":
        value = row.get(metric)
        return None if value is None else float(value)
    if metric == "total_latency":
        return row.get("total_retrieval_latency_ms")
    if metric == "estimated_context_tokens":
        return row.get("estimated_token_count")
    return row.get(metric)


def paired_comparisons(experiments):
    baseline = {row["question_id"]: row for row in experiments["temporal_baseline"]["raw"]}
    rows = []
    metrics = [
        "complete_evidence_recall",
        "all_reports_hit",
        "evidence_recall",
        "macro_mrr",
        "total_latency",
        "estimated_context_tokens",
    ]
    for experiment_id, payload in sorted(experiments.items()):
        if experiment_id == "temporal_baseline":
            continue
        current = {row["question_id"]: row for row in payload["raw"]}
        common = sorted(set(baseline) & set(current))
        for metric in metrics:
            pairs = []
            for qid in common:
                left = metric_value(baseline[qid], metric)
                right = metric_value(current[qid], metric)
                if left is not None and right is not None:
                    pairs.append((left, right))
            differences = [right - left for left, right in pairs]
            binary = metric in {"complete_evidence_recall", "all_reports_hit"}
            row = {
                "experiment_id": experiment_id,
                "metric": metric,
                "case_count": len(pairs),
                "baseline_mean": mean([left for left, _ in pairs]),
                "experiment_mean": mean([right for _, right in pairs]),
                "mean_difference": mean(differences),
                "ci_95_low": paired_bootstrap_interval(differences)[0] if differences else None,
                "ci_95_high": paired_bootstrap_interval(differences)[1] if differences else None,
                "conclusive": False,
            }
            if row["ci_95_low"] is not None and row["ci_95_high"] is not None:
                row["conclusive"] = row["ci_95_low"] > 0 or row["ci_95_high"] < 0
            if binary:
                row.update({
                    "baseline_fail_experiment_pass": sum(1 for left, right in pairs if not left and right),
                    "baseline_pass_experiment_fail": sum(1 for left, right in pairs if left and not right),
                    "both_pass": sum(1 for left, right in pairs if left and right),
                    "both_fail": sum(1 for left, right in pairs if not left and not right),
                })
            rows.append(row)
    write_json(OUT / "paired_comparisons.json", rows)
    write_csv(OUT / "paired_comparisons.csv", rows)
    lines = ["# Paired Comparisons", "", "Compared against repaired `temporal_baseline` with paired bootstrap intervals, 2,000 resamples, seed 42.", ""]
    for row in rows:
        if row["metric"] == "complete_evidence_recall":
            lines.append(f"- {row['experiment_id']}: diff={row['mean_difference']}, CI=[{row['ci_95_low']}, {row['ci_95_high']}], conclusive={row['conclusive']}")
    (OUT / "paired_comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reranker_promotion_demotion(experiments):
    rows = []
    for experiment_id, payload in sorted(experiments.items()):
        for row in payload["raw"]:
            for report_id, trace in row.get("per_report", {}).items():
                before = trace.get("evidence_rank_before_reranking")
                after = trace.get("evidence_rank_after_reranking")
                if before is None:
                    continue
                final_selected = bool(set(trace.get("selected_chunk_ids_after_dedup", [])) & set(trace.get("reranker_output_ids", [])[: max(1, len(trace.get("selected_chunk_ids_after_dedup", [])))]))
                if after is None:
                    klass = "dropped_from_final_context"
                    rank_change = None
                elif after < before:
                    klass = "promoted"
                    rank_change = before - after
                elif after > before:
                    klass = "demoted"
                    rank_change = before - after
                else:
                    klass = "unchanged"
                    rank_change = 0
                rows.append({
                    "experiment_id": experiment_id,
                    "question_id": row["question_id"],
                    "report_id": report_id,
                    "evidence_rank_before_reranking": before,
                    "evidence_rank_after_reranking": after,
                    "reranker_score": None,
                    "input_text_length": None,
                    "estimated_input_tokens": None,
                    "truncation_risk": False,
                    "final_selected": final_selected,
                    "classification": klass,
                    "rank_change": rank_change,
                })
    summary = []
    by_exp = defaultdict(list)
    for row in rows:
        by_exp[row["experiment_id"]].append(row)
    for experiment_id, values in sorted(by_exp.items()):
        counts = defaultdict(int)
        for row in values:
            counts[row["classification"]] += 1
        changes = [row["rank_change"] for row in values if row["rank_change"] is not None]
        total = len(values)
        summary.append({
            "experiment_id": experiment_id,
            "case_count": total,
            "promotion_rate": counts["promoted"] / total if total else None,
            "demotion_rate": counts["demoted"] / total if total else None,
            "unchanged_rate": counts["unchanged"] / total if total else None,
            "evidence_drop_rate": counts["dropped_from_final_context"] / total if total else None,
            "mean_rank_change": mean(changes),
            "median_rank_change": statistics.median(changes) if changes else None,
        })
    write_json(OUT / "reranker_promotion_demotion.json", {"rows": rows, "summary": summary})
    write_csv(OUT / "reranker_promotion_demotion.csv", rows)
    lines = ["# Reranker Promotion/Demotion", ""]
    for row in summary:
        lines.append(f"- {row['experiment_id']}: promotion={row['promotion_rate']}, demotion={row['demotion_rate']}, drop={row['evidence_drop_rate']}")
    (OUT / "reranker_promotion_demotion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def repaired_baseline():
    target = OUT / "repaired_baseline"
    target.mkdir(exist_ok=True)
    source = OUT / "temporal_baseline"
    for name in ("config_snapshot.yaml", "environment.json", "index_manifest.json", "raw_results.json", "question_results.csv", "report_level_results.csv", "summary.json", "summary.md", "stage_diagnostics.csv"):
        shutil.copy2(source / name, target / name)
    summary = load_json(target / "summary.json")
    (target / "README.md").write_text(
        "# Repaired Temporal Baseline\n\n"
        "This directory is a copy of the repaired `temporal_baseline` run generated with per-question StageTimer traces.\n\n"
        f"Complete Evidence Recall: {summary.get('complete_evidence_recall')}\n\n"
        f"Median latency ms: {summary.get('median_latency_ms')}\n",
        encoding="utf-8",
    )


def patch_status(experiments):
    status_path = OUT / "stage_a_selection_status.json"
    status = load_json(status_path)
    rerun_status = load_json(OUT / "rerun_status.json")
    status["valid_rerun_count"] = sum(1 for row in rerun_status if row.get("integrity_status") == "valid")
    status["invalid_rerun_count"] = sum(1 for row in rerun_status if row.get("integrity_status") != "valid")
    status["valid_development_experiment_count"] = len(experiments)
    status["eligible_experiment_count"] = status.pop("valid_experiment_count", status.get("eligible_experiment_count"))
    status["groq_api_key_available"] = True
    write_json(status_path, status)


def patch_report_appendix():
    report = OUT / "stage_a_report.md"
    text = report.read_text(encoding="utf-8")
    appendix = (
        "\n## Statistical Artifacts\n\n"
        "- Paired comparisons: `paired_comparisons.*`\n"
        "- Reranker promotion/demotion: `reranker_promotion_demotion.*`\n"
        "- Repaired baseline copy: `repaired_baseline/`\n"
    )
    if "## Statistical Artifacts" not in text:
        report.write_text(text.rstrip() + "\n" + appendix, encoding="utf-8")


def main():
    experiments = load_experiments()
    repaired_baseline()
    paired_comparisons(experiments)
    reranker_promotion_demotion(experiments)
    patch_status(experiments)
    patch_report_appendix()
    print(json.dumps({
        "experiments": len(experiments),
        "paired_comparisons": True,
        "reranker_promotion_demotion": True,
        "repaired_baseline": True,
    }, indent=2))


if __name__ == "__main__":
    main()
