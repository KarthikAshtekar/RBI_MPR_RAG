from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbi_rag.env_loading import load_project_dotenv
from rbi_rag.v2_experiments import V2_OUT, generate_reports

load_project_dotenv(ROOT)


def main() -> int:
    generate_reports(ROOT)
    payload = {
        "presentation_summary": str(ROOT / V2_OUT / "v2_results_for_presentation.md"),
        "final_report": str(ROOT / V2_OUT / "v2_unstructured_cohere_report.md"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
