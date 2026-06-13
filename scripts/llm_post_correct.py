from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_pipeline.config import ensure_project_dirs, load_config
from asr_pipeline.llm_correction import correct_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply constrained LLM post-correction to ASR predictions.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    result = correct_predictions(cfg, args.predictions)
    print(result["metrics"])


if __name__ == "__main__":
    main()

