from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbi_rag.env_loading import load_project_dotenv
from rbi_rag.poppler_setup import apply_poppler_path_from_helper
from rbi_rag.v2_experiments import run_v2

load_project_dotenv(ROOT)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run V2 Unstructured + Cohere controlled retrieval experiments.")
    root.add_argument(
        "--post-final-heldout-diagnostic",
        action="store_true",
        help="Reserved for a clearly labelled reused-heldout diagnostic. Not run by default.",
    )
    root.add_argument(
        "--only",
        help="Comma-separated controlled experiment IDs to execute. Existing artifacts are reused for excluded arms.",
    )
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.post_final_heldout_diagnostic:
        raise SystemExit(
            "Post-final held-out diagnostic is prepared but intentionally not run by this command. "
            "Use a dedicated diagnostic runner only after confirming the reused-heldout caveat."
        )
    apply_poppler_path_from_helper(ROOT)
    only = [item.strip() for item in args.only.split(",")] if args.only else None
    payload = run_v2(ROOT, only=only)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
