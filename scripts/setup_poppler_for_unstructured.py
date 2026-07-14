from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbi_rag.env_loading import load_project_dotenv
from rbi_rag.poppler_setup import setup_poppler


def main() -> int:
    load_project_dotenv(ROOT)
    payload = setup_poppler(ROOT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

