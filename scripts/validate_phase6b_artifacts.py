from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase6b_structural_experiments import OUT, validate_experiment


def main():
    dataset_checksum = None
    checks = []
    for directory in sorted(path for path in OUT.iterdir() if path.is_dir()):
        if not (directory / "raw_results.json").exists():
            continue
        raw = json.loads((directory / "raw_results.json").read_text(encoding="utf-8"))
        if raw and dataset_checksum is None:
            dataset_checksum = raw[0].get("dataset_checksum")
        issues = validate_experiment(directory, dataset_checksum, len(raw))
        checks.append({
            "experiment_id": directory.name,
            "valid": not issues,
            "issue_count": len(issues),
            "issues": issues,
        })
    payload = {
        "experiment_count": len(checks),
        "valid_count": sum(item["valid"] for item in checks),
        "invalid_count": sum(not item["valid"] for item in checks),
        "checks": checks,
    }
    (OUT / "integrity_validation_rerun.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("experiment_count", "valid_count", "invalid_count")}, indent=2))
    return 0 if payload["invalid_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
