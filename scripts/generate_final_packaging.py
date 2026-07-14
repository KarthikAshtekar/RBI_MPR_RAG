from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rbi_rag.final_packaging import generate_packaging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-status", default="not_run")
    parser.add_argument("--pip-check-status", default="not_run")
    args = parser.parse_args()
    result = generate_packaging(ROOT, tests_status=args.tests_status, pip_check_status=args.pip_check_status)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"completed", "completed_with_skips"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
