from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rbi_rag.env_loading import load_project_dotenv
from rbi_rag.v2_generation_evaluation import validate_generation_artifacts


def main() -> int:
    load_project_dotenv(ROOT)
    result = validate_generation_artifacts(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
