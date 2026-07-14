from __future__ import annotations

import argparse
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
from rbi_rag.v2_generation_evaluation import run_v2_generation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2 development generation evaluation.")
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--retrieval-experiment", default="V2_COHERE_ONLY")
    parser.add_argument("--config", default="configs/v2_selected_retrieval.yaml")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--temperature", default=0.0, type=float)
    args = parser.parse_args()

    load_project_dotenv(ROOT)
    result = run_v2_generation(
        ROOT,
        split=args.split,
        retrieval_experiment=args.retrieval_experiment,
        config_path=Path(args.config),
        model_name=args.model,
        temperature=args.temperature,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
