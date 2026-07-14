from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rbi_rag.env_loading import load_project_dotenv  # noqa: E402
from rbi_rag.final_mmr_generation import run_final_mmr_generation  # noqa: E402


def main() -> int:
    load_project_dotenv(ROOT)
    result = run_final_mmr_generation(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"completed", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
