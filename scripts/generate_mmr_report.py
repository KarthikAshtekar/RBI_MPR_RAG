from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rbi_rag.mmr_selection import generate_mmr_report  # noqa: E402


def main() -> int:
    result = generate_mmr_report(ROOT)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
