from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rbi_rag.mmr_selection import run_mmr_experiments  # noqa: E402


def main() -> int:
    result = run_mmr_experiments(ROOT)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"completed", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
